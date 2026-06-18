# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.

中文学习提示：
这是 Skywork-OR1 训练的 Python 总入口。bash 训练脚本最终都会调用：
    python3 -m verl.trainer.main_ppo ...

你可以把这个文件理解成“装配器”：
1. 读取 Hydra 配置；
2. 初始化 Ray；
3. 根据配置选择 FSDP 或 Megatron worker；
4. 选择 reward manager；
5. 创建 RayPPOTrainer 并进入 trainer.fit() 主循环。
"""
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.reward_score.livecodebench import compute_score as code_compute_score

import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    # Hydra 会先加载 verl/trainer/config/ppo_trainer.yaml，
    # 再叠加命令行里的 algorithm.adv_estimator=grpo 等覆盖项。
    run_ppo(config, code_compute_score)


def run_ppo(config, compute_score=None):
    if not ray.is_initialized():
        # 单机调试时自动启动一个本地 Ray runtime。
        # 多机训练时通常先在各节点 ray start，再运行训练脚本接入 Ray 集群。
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

        # debug
        # ray.init(
        #     runtime_env={
        #         'env_vars': {
        #             'TOKENIZERS_PARALLELISM': 'true', 
        #             'NCCL_DEBUG': 'WARN', 
        #             "RAY_DEBUG":"1",
        #             "RAY_DEBUG_POST_MORTEM":"1"
        #             }
        #         },
        #     )

    ray.get(main_task.remote(config, compute_score))
    # main_task(config, compute_score)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def main_task(config, compute_score=None):
    from verl.utils.fs import copy_local_path_from_hdfs
    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # 如果模型路径是 HDFS/远端路径，这里会先复制到本地；普通本地/HF 路径则直接使用。
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # tokenizer 决定 chat_template、padding token、eos token，也是奖励函数解码 response 的基础。
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # 根据训练策略选择 worker 实现：
    # - fsdp: PyTorch FSDP，适合读懂和单机/少量机器复现；
    # - megatron: 更复杂的大规模张量/流水并行训练。
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError
    
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    # Role 表示 Ray 集群里不同“角色”的 worker。
    # Skywork 的 hybrid_engine 把 actor、rollout、ref 合并在 ActorRolloutRefWorker 里，
    # 这样同一套权重可以训练、生成和计算 logprob，减少权重同步成本。
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        # Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        # Role.RefPolicy: global_pool_id,
    }

    # reward 有两种来源：
    # 1. 训练出来的 reward model；
    # 2. rule-based verifier，例如数学答案校验、代码单测。
    # Skywork-OR1 脚本默认 reward_model.enable=False，主要依赖 rule-based verifier。
    #
    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    if reward_manager_name == 'naive':
        from verl.workers.reward_manager import NaiveRewardManager
        reward_manager_cls = NaiveRewardManager
    elif reward_manager_name == 'prime':
        from verl.workers.reward_manager import PrimeRewardManager
        reward_manager_cls = PrimeRewardManager
    elif reward_manager_name == 'yr':
        from verl.workers.reward_manager import YRRewardManager
        reward_manager_cls = YRRewardManager
    else:
        raise NotImplementedError
    # 训练脚本里 reward_model.reward_manager=yr，所以这里通常实例化 YRRewardManager。
    # compute_score 来自 livecodebench.compute_score，它会根据 ground_truth 类型判断是数学校验还是代码单测。
    reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score, is_long_penalty=config.reward_model.get("is_long_penalty", False),is_binary_reward=config.reward_model.get("is_binary_reward", True), is_power4_reward=config.reward_model.get("is_power4_reward", False))

    # 验证阶段也用函数式 verifier，num_examine=1 会打印少量样本，便于人工看生成质量。
    val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    # init_workers 会真正启动/初始化各个 Ray worker 的模型、FSDP、vLLM rollout 等组件。
    trainer.init_workers()
    # fit 是训练主循环：采样 -> 打分 -> 计算 advantage -> 更新 actor -> 验证/保存。
    trainer.fit()


if __name__ == '__main__':
    main()
