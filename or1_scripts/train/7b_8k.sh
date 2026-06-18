#!/bin/bash
set -ex

# 中文学习提示：
# 这是 Skywork-OR1-7B 的 8K 回复长度训练脚本，也是最适合新手先读懂的一条主线。
# 它没有把训练逻辑写在 bash 里，而是通过 Hydra 覆盖参数启动：
#   python3 -m verl.trainer.main_ppo
# 真正的训练循环在 verl/trainer/ppo/ray_trainer.py。

# Ray / torch distributed 的基础环境变量。单机时默认 WORLD_SIZE=1、RANK=0。
export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-29500}
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1

# Entropy Config
# 熵正则用于防止模型过早变得“太确定”，也就是技术报告里讨论的 entropy collapse。
# Skywork 这里启用自适应熵系数：当前熵低于目标值时增大 entropy coefficient。
ENTROPY_COEFF=0.0
USE_ADAPTIVE_ENT=True
TGT_ENTROPY=0.2
MAX_ENT_COEF=0.005
MIN_ENT_COEF=0
DELTA_ENT_COEF=0.0001

# ROLLOUT_BATCH_SIZE 是每步抽取多少个 prompt。
# GROUP_SIZE=16 表示每个 prompt 采样 16 条回答，这是 GRPO 的关键：
# 同一个题目下多条回答互相比较，奖励有高有低才有训练信号。
ROLLOUT_BATCH_SIZE=256
PPO_MINI_BATCH=256
MAX_PROMPT_LENGTH=2048
RES_LENGTH=8192
GROUP_SIZE=16
N_VAL_SAMPLES=8

# 训练采样温度。温度越高，采样越发散；GRPO 需要一定多样性，但太高会降低可学习性。
TRAIN_TEMPERATURE=1.0

# TP 是 vLLM 推理侧 tensor parallel size。
# SP 是 Ulysses sequence parallel size，长上下文训练时用于切分序列维。
TP=1
SP=1
MAX_TOKEN_LEN=$(((RES_LENGTH + MAX_PROMPT_LENGTH + 1000) / SP))

# Your Model Path
# MODEL_PATH 通常指向 DeepSeek-R1-Distill-Qwen-7B 或继续训练得到的 checkpoint。
# CODE_PATH 指向当前仓库根目录，脚本通过它找到 or1_data、保存 verl_ckpt。
MODEL_PATH=${MODEL_PATH:-}
CODE_PATH=${CODE_PATH:-}
if [ -z "$MODEL_PATH" ]; then
    echo "MODEL_PATH is not set"
    exit 1
fi
if [ -z "$CODE_PATH" ]; then
    echo "CODE_PATH is not set"
    exit 1
fi

# Since math queries are much more than code queries, we duplicate the math data when mixing the datasets
# README 这句英文和实际脚本不完全一致：这里重复的是 code 数据两次，再加 math 一次。
# 目的都是调节 math/code 混合比例，让代码样本在训练 batch 中有足够权重。
train_files="[\"$CODE_PATH/or1_data/train/train_7b_code.pkl\",\"$CODE_PATH/or1_data/train/train_7b_code.pkl\",\"$CODE_PATH/or1_data/train/train_7b_math.pkl\"]"
test_files="[\"$CODE_PATH/or1_data/eval/aime24.parquet\",\"$CODE_PATH/or1_data/eval/aime25.parquet\"]"

PROJECT_NAME=skywork-or1-train

EXP_NAME=7B_L$(($RES_LENGTH / 1024))k
MODEL_NAME=$(basename $MODEL_PATH)
EXP_NAME=$EXP_NAME-${MODEL_NAME}-bs${ROLLOUT_BATCH_SIZE}-minibs${ROLLOUT_BATCH_SIZE}-gs${GROUP_SIZE}-tgt${TGT_ENTROPY}-temp${TRAIN_TEMPERATURE}-${WORLD_SIZE}nodes
SAVE_DIR=$CODE_PATH/verl_ckpt/$PROJECT_NAME/$EXP_NAME
SAVE_STATS_DIR=${SAVE_DIR}/stats
mkdir -p $SAVE_DIR
mkdir -p $SAVE_STATS_DIR

export RAY_DEBUG=1

# 下面所有 key=value 都是 Hydra 覆盖项，会覆盖 verl/trainer/config/ppo_trainer.yaml。
# 新手读代码时建议按这个顺序追踪：
# 1. main_ppo.py 读取配置并创建 RayPPOTrainer；
# 2. ray_trainer.py 用 data.train_files 创建 RLHFDataset；
# 3. ActorRolloutRefWorker 用 vLLM 生成 GROUP_SIZE 条回答；
# 4. YRRewardManager 调 verifier 得到奖励；
# 5. core_algos.compute_grpo_outcome_advantage 计算 advantage；
# 6. update_actor 做 PPO-style policy update。
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.train_batch_size=$ROLLOUT_BATCH_SIZE \
    data.val_batch_size=13000 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$RES_LENGTH \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKEN_LEN \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.adaptive_entropy.enabled=$USE_ADAPTIVE_ENT \
    actor_rollout_ref.actor.adaptive_entropy.target_entropy=${TGT_ENTROPY} \
    actor_rollout_ref.actor.adaptive_entropy.max_ent_coef=${MAX_ENT_COEF} \
    actor_rollout_ref.actor.adaptive_entropy.min_ent_coef=${MIN_ENT_COEF} \
    actor_rollout_ref.actor.adaptive_entropy.delta_ent_coef=${DELTA_ENT_COEF} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$TRAIN_TEMPERATURE \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$GROUP_SIZE \
    actor_rollout_ref.rollout.n_val=$N_VAL_SAMPLES \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    reward_model.reward_manager=yr \
    trainer.critic_warmup=0 \
    trainer.rejection_sample=True \
    trainer.rejection_sample_multiplier=1 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$WORLD_SIZE \
    trainer.save_freq=20 \
    trainer.test_freq=20\
    trainer.stats_path=$SAVE_STATS_DIR \
    trainer.stats_save_freq=1 \
    trainer.default_local_dir=$SAVE_DIR \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=30 "${@:1}"
