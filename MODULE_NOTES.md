# Skywork-OR1 Core Module Notes

## 1. `verl/protocol.py` - DataProto

- 职责：统一 Driver、Ray worker、actor、rollout、reward 之间的数据交换。
- 接口：`from_dict`、`from_single_dict`、`select`、`pop`、`union`、`repeat`、`reorder`、`chunk`、`concat`、`to`。
- 依赖：PyTorch、TensorDict、NumPy、Ray object refs。
- 被依赖：几乎所有 trainer 和 worker。
- 核心结构：`batch`、`non_tensor_batch`、`meta_info`。
- 隐含假设：只支持一个 batch 维；非张量数据必须是 `dtype=object` 的 NumPy 数组且首维对齐。
- 常见修改：新增字段、跨 worker 传输、batch 重排。
- 坑点：`__getitem__` 返回 `DataProtoItem` 而不是 `DataProto`；布尔索引后某些调用方需转回 `DataProto`。

## 2. `verl/utils/dataset/rl_dataset.py` - RLHFDataset

- 职责：读取 pkl/parquet，套 chat template，tokenize，过滤过长 prompt。
- 接口：`__getitem__`、`__len__`、`resume_dataset_state`、`collate_fn`。
- 依赖：pandas、Transformers tokenizer、`tokenize_and_postprocess_data`。
- 被依赖：`RayPPOTrainer._create_dataloader`。
- 核心结构：prompt 消息列表和保留的 reward metadata。
- 隐含假设：数据 prompt 格式适配 tokenizer chat template；prompt 长度过滤不会破坏索引。
- 常见修改：新增字段、改变截断策略、视觉输入扩展。
- 坑点：训练数据是 pkl，验证数据是 parquet；两条读取路径都要测试。

## 3. `verl/trainer/main_ppo.py` - 装配入口

- 职责：初始化 Ray、tokenizer、worker class、资源池和 reward manager。
- 接口：`main`、`run_ppo`、`main_task`。
- 依赖：Hydra、Ray、`RayPPOTrainer`、FSDP/Megatron worker。
- 被依赖：所有 `or1_scripts/train/*.sh`。
- 隐含假设：模型路径可被 `copy_local_path_from_hdfs` 解析；配置字段完整。
- 常见修改：新增 reward manager、worker role、训练后端。
- 坑点：Skywork 脚本使用 `reward_manager=yr`；默认 YAML 是 `naive`，读配置时必须区分默认与覆盖。

## 4. `verl/trainer/ppo/ray_trainer.py` - 训练编排器

- 职责：数据加载、worker 初始化、rollout、reward、rejection sampling、advantage、更新、验证、保存。
- 接口：`init_workers`、`fit`、`_validate`、`_save_checkpoint`。
- 依赖：`DataProto`、RayWorkerGroup、core_algos、RLHFDataset、MetricFunc。
- 被依赖：`main_ppo.py` 和 arithmetic E2E trainer。
- 核心结构：Role、ResourcePoolManager、batch UID、timing/metric dict。
- 隐含假设：rejection 后 batch 仍足以按 world size 分配；group 顺序在 balance 后仍可由 UID 恢复。
- 常见修改：新 advantage estimator、新 rejection 规则、新验证指标。
- 坑点：`dataprotoitem_to_dataproto` 在文件开头重复定义；全组无效时会 `continue`；实际有效 batch 会变化。

## 5. `verl/trainer/ppo/core_algos.py` - 算法核心

- 职责：GAE、GRPO、REINFORCE++、ReMax advantage，PPO loss、value loss、KL、entropy controller。
- 接口：`compute_grpo_outcome_advantage`、`compute_policy_loss`、`kl_penalty`、`EntController.update`。
- 依赖：PyTorch 和 masked tensor utilities。
- 被依赖：RayPPOTrainer、actor、critic。
- 核心结构：`[batch, response_length]` 的 reward/advantage/logprob/mask。
- 隐含假设：同组至少有多个样本；mask 中有效 token 数足够；epsilon 防除零但不能创造信息。
- 常见修改：advantage normalization、clip 方式、KL 类型、entropy 调度。
- 坑点：全对/全错组即使数学上可算，训练上几乎无信息，所以在 trainer 提前过滤。

## 6. `verl/workers/actor/dp_actor.py` - 策略计算与更新

- 职责：actor forward、logprob/entropy、PPO loss、反向传播和 optimizer step。
- 接口：`compute_log_prob`、`update_policy`。
- 依赖：FSDP、flash-attn remove-padding、Ulysses sequence parallel、core_algos。
- 被依赖：`ActorRolloutRefWorker`。
- 核心结构：response mask、old/new logprob、advantages、entropy controller。
- 隐含假设：CUDA + bf16；`ppo_mini_batch_size` 可被 micro batch 划分。
- 常见修改：loss 项、动态 batch、梯度累计、profiler。
- 坑点：`data.cuda()` 是硬编码 GPU 路径；CPU 不能完整执行更新。

## 7. `verl/workers/fsdp_workers.py` - GPU Worker

- 职责：构建 HF 模型、FSDP、optimizer、actor、vLLM rollout、ref/critic/RM。
- 接口：`init_model`、`generate_sequences`、`compute_log_prob`、`compute_ref_log_prob`、`update_actor`、checkpoint。
- 依赖：Transformers、FSDP、sharding manager、actor/rollout。
- 被依赖：RayPPOTrainer 经 RayWorkerGroup 远程调用。
- 核心结构：role 标志、device mesh、offload 状态。
- 隐含假设：world size 与 TP/SP/FSDP size 可整除；模型支持 remove-padding。
- 常见修改：新模型架构、并行策略、offload、权重加载。
- 坑点：训练模型和 vLLM 引擎权重布局不同；切换时显存峰值敏感。

## 8. `verl/workers/rollout/vllm_rollout/vllm_rollout.py`

- 职责：把 padded prompt 转成 vLLM 输入，多样本生成，重建 response mask/position ids。
- 接口：`generate_sequences`、`update_sampling_params`。
- 依赖：vLLM 0.6.3 适配层、SamplingParams、sharding manager。
- 被依赖：ActorRolloutRefWorker。
- 核心结构：prompt token list、response、logprob、KV cache。
- 隐含假设：模型上下文上限 >= prompt + response；TP <= world size。
- 常见修改：temperature/top-p、n、max batched tokens、cache 策略。
- 坑点：生成后 actor 仍要重算 logprob；`free_cache_engine` 与 eager/CUDA graph 配置有关。

## 9. `verl/workers/reward_manager/yr_code.py` 与 verifier

- 职责：decode response，选择 ground truth，并行调用数学/代码 verifier，构造 reward tensor。
- 接口：`YRRewardManager.__call__`、`parallel_compute_score`、`compute_score`。
- 依赖：tokenizer、ProcessPoolExecutor、math-verify、LiveCodeBench sandbox。
- 被依赖：main_ppo 和 main_generation。
- 核心结构：字符串 response、ground truth dict、最后有效 token reward。
- 隐含假设：response 非空；verifier 可序列化到子进程；Linux 提供 SIGALRM。
- 常见修改：连续奖励、超时、按数据源路由、新 verifier。
- 坑点：批量 reward 发生任意异常时，当前实现可能把整批 scores 置 0；`parallel_compute_score` 的 `timeout` 参数未直接控制 Future。

## 10. `verl/single_controller/` - Ray 控制层

- 职责：创建资源池、spawn actor、根据装饰器把方法分发到一个或多个 rank。
- 接口：`RayResourcePool`、`RayWorkerGroup`、`RayClassWithInitArgs`、`register`。
- 依赖：Ray placement group、DataProto dispatch decorator。
- 被依赖：trainer 和所有 worker。
- 核心结构：worker list、rank、world size、dispatch/collect mode。
- 隐含假设：placement group 资源满足配置，远程方法有正确 dispatch metadata。
- 常见修改：新 worker role、colocation、异步执行。
- 坑点：远程错误栈跨 Ray 边界后更难定位；需先区分 driver 和 worker 日志。

## 11. 文档概念到代码映射

| 论文/博客概念 | 代码 |
|---|---|
| MAGIC | `or1_scripts/train/*.sh` + `RayPPOTrainer.fit` |
| Model-aware difficulty | `download_and_filter_data_*.py::filter_fn` |
| Offline/online filtering | 数据脚本 + `fit` 的 rejection sampling |
| GRPO | `compute_grpo_outcome_advantage` |
| High-temperature sampling | `TRAIN_TEMPERATURE=1.0`、SamplingParams |
| Adaptive entropy | `EntController`、`dp_actor.update_policy` |
| No KL loss | Role.RefPolicy 未启用、KL coefficient 实际为 0 |
| Multi-stage context | 8K/16K/32K shell scripts |
| Math verifier | `math_verify_reward_function` |
| Code sandbox | `livecodebench/compute_score.py` |
| Avg@K | `main_generation.py` 多采样和 score reshape |

## 12. 文档没强调但代码很重要的细节

- 生成数据的顺序要 `reorder`，否则同题 group 会错位。
- rejection 后 batch 会向下取整到 world size 倍数，实际样本数可能减少。
- `_balance_batch` 按 token 数平衡 DP rank，不只是按样本数。
- 训练时 vLLM logprob 不直接用于 PPO，actor 会重新计算。
- 公开 `7b_8k.sh` 的英文注释说复制 math，实际 `train_files` 是 code 两次、math 一次。
- `main_generation.py` 同时计算首样本准确率、全部采样平均准确率和 Pass@K，字段名容易混淆。
