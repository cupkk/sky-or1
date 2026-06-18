# Skywork-OR1 中文学习复现指南

## 1. 项目定位

Skywork-OR1 是一个面向数学和代码推理的 LLM 后训练项目。它不是多模态模型项目，也不包含图像/视频输入输出链路。对多模态大模型实习来说，它的价值在于展示你掌握了大模型后训练中的通用能力：

- 基于可验证奖励的强化学习训练。
- 数学与代码任务的 rule-based verifier 设计。
- GRPO/PPO 类策略优化。
- vLLM 高吞吐 rollout。
- Ray + FSDP 的分布式训练工程。
- 长上下文生成、KL 约束、熵坍塌缓解和 rejection sampling。

简历包装时不要写成“多模态项目”，而应写成“推理大模型后训练复现与源码解析项目”。多模态岗位同样会关心这些能力，因为 VLM 后训练也常用 RL、偏好优化、verifier、分布式训练和高吞吐推理。

## 2. 一句话讲清楚 Skywork-OR1

Skywork-OR1 在 DeepSeek-R1-Distill-Qwen 等强推理基座模型上，使用数学/代码题的可验证奖励进行大规模强化学习，让模型通过同题多采样、自动验题、相对优势优化来提升推理能力。

## 3. 本地文件状态

- 官方源码已拉到当前仓库，官方远端名为 `official`。
- 当前本地分支为 `official-main`，跟踪 `official/main`。
- 技术报告已下载到 `papers/2505.22312.pdf`。
- 重点源码已加入中文注释：
  - `or1_scripts/data_preprocess/download_and_filter_data_7b.py`
  - `or1_scripts/train/7b_8k.sh`
  - `verl/trainer/main_ppo.py`
  - `verl/trainer/ppo/ray_trainer.py`
  - `verl/trainer/ppo/core_algos.py`
  - `verl/workers/reward_manager/yr_code.py`
  - `verl/utils/reward_score/livecodebench/compute_score.py`
  - `verl/utils/dataset/rl_dataset.py`
  - `verl/trainer/main_generation.py`
  - `verl/workers/rollout/vllm_rollout/vllm_rollout.py`
  - `verl/workers/fsdp_workers.py`

## 4. 最小复现路线

### 4.1 推荐环境

官方 README 给了 Docker 和 Conda 两条路。更推荐 Docker，因为训练依赖 CUDA、vLLM、flash-attn、Ray、FSDP，Windows 原生环境很容易踩坑。

```bash
docker pull whatcanyousee/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te2.0-megatron0.11.0-v0.0.6
```

进入容器后：

```bash
git clone https://github.com/SkyworkAI/Skywork-OR1.git
cd Skywork-OR1
pip3 install -e .
```

如果使用当前已经注释过的本地仓库，把当前目录挂载进容器即可。

### 4.2 准备训练数据

7B 路线：

```bash
model_size=7b
python ./or1_scripts/data_preprocess/download_and_filter_data_${model_size}.py --local_dir ./or1_data/train
```

输出：

```text
or1_data/train/train_7b_math.pkl
or1_data/train/train_7b_code.pkl
```

这个脚本会从 Hugging Face 数据集 `Skywork/Skywork-OR1-RL-Data` 下载样本，并按 `DeepSeek-R1-Distill-Qwen-7B` 的难度标注过滤。

### 4.3 运行 7B 8K 训练

需要先准备基座模型路径，通常是 Hugging Face 模型名或本地模型目录。

```bash
export CODE_PATH=/path/to/Skywork-OR1
export MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
bash ./or1_scripts/train/7b_8k.sh
```

重要提示：

- 官方脚本默认 `trainer.n_gpus_per_node=8`，真实训练需要多卡大显存。
- 若只是学习代码，不建议一开始完整训练 7B。
- 新手可先读脚本和核心代码，再在小模型/小数据上改参数做冒烟实验。

### 4.4 运行评测

```bash
bash ./or1_scripts/eval/eval_7b.sh
```

默认评测：

- AIME24：每题采样 32 次。
- AIME25：每题采样 32 次。
- LiveCodeBench：每题采样 4 次，需要额外下载数据。

结果保存到：

```text
outputs/evalation/pass.csv
```

## 5. 核心代码阅读路线

按下面顺序读，不要一开始钻进 `verl/third_party/vllm`。

### 5.1 训练脚本

先读：

```text
or1_scripts/train/7b_8k.sh
```

重点理解：

- `algorithm.adv_estimator=grpo`
- `actor_rollout_ref.rollout.n=16`
- `reward_model.reward_manager=yr`
- `trainer.rejection_sample=True`
- `actor_rollout_ref.actor.adaptive_entropy.*`
- `actor_rollout_ref.rollout.name=vllm`

### 5.2 训练入口

再读：

```text
verl/trainer/main_ppo.py
```

它负责组装 tokenizer、Ray worker、reward manager 和 `RayPPOTrainer`。

### 5.3 主训练循环

核心文件：

```text
verl/trainer/ppo/ray_trainer.py
```

主线是：

```text
dataloader batch
-> generate_sequences
-> reward_fn
-> rejection sampling
-> compute old_log_probs
-> compute ref_log_prob
-> apply KL penalty
-> compute GRPO advantage
-> update_actor
-> validate/save/log
```

### 5.4 算法核心

```text
verl/trainer/ppo/core_algos.py
```

重点函数：

- `compute_grpo_outcome_advantage`
- `compute_policy_loss`
- `compute_entropy_loss`
- `kl_penalty`

### 5.5 奖励函数

```text
verl/workers/reward_manager/yr_code.py
verl/utils/reward_score/livecodebench/compute_score.py
```

核心思想：

- 模型生成完整回答。
- verifier 判断数学答案或代码单测是否正确。
- 每条回答得到一个标量 reward。
- reward 放到最后一个有效 response token 上。

### 5.6 生成与评测

```text
verl/trainer/main_generation.py
verl/workers/rollout/vllm_rollout/vllm_rollout.py
```

重点理解：

- 每题重复采样 `n_samples` 次。
- vLLM 负责高吞吐生成。
- 生成结果保存后再统一打分。
- Pass@K 看 K 次里是否至少有一次正确。
- Avg@K / average sample accuracy 看多次采样的平均正确率。

## 6. 底层原理速记

### 6.1 为什么用 rule-based reward

数学和代码任务天然可验证：

- 数学题可以抽取最终答案，做字符串、数值、符号等价判断。
- 代码题可以执行测试用例。

这比训练一个 reward model 更直接，奖励更客观，也更适合 reasoning model 后训练。

### 6.2 GRPO 和 PPO 的区别

传统 PPO 常需要 critic/value model 估计 baseline。GRPO 省掉 critic，而是同一个 prompt 采样多条回答，用组内平均奖励作为 baseline。

简化公式：

```text
advantage_i = (reward_i - mean(group_rewards)) / std(group_rewards)
```

所以 GRPO 特别依赖 `rollout.n > 1`。如果同一题所有回答都对或都错，这个题对训练几乎没有帮助。

### 6.3 为什么要 rejection sampling

同一 prompt 的所有采样都对：

```text
[1, 1, 1, 1, ...]
```

或都错：

```text
[0, 0, 0, 0, ...]
```

组内没有相对差异，GRPO 学不到“哪种回答更好”。所以训练循环会过滤这类 prompt 组。

### 6.4 为什么要 KL

只优化 verifier reward 可能导致模型学会投机格式、语言质量下降或偏离原模型分布。KL penalty 用参考模型约束当前策略：

```text
reward = verifier_score - beta * KL(current_policy, reference_policy)
```

### 6.5 为什么关注 entropy collapse

RL 训练中模型可能越来越确定，采样多样性下降，导致探索不足。Skywork 脚本加入 adaptive entropy：

- 当前 entropy 太低：增大 entropy coefficient。
- 当前 entropy 高于目标：减小 entropy coefficient。

## 7. 面试高频问答

### Q1：这个项目和 RLHF 有什么关系？

它属于大模型后训练中的强化学习路线，但更准确叫 RLVR：reinforcement learning with verifiable rewards。奖励不是人类偏好模型，而是数学/代码 verifier。

### Q2：为什么代码题可以自动打分？

模型输出 Python 代码，系统抽取代码块，然后运行隐藏/公开测试用例。全部通过给 1，否则给 0，也可以配置为按通过比例给连续分数。

### Q3：GRPO 为什么不需要 critic？

GRPO 用同一个 prompt 的多条采样回答构造相对 baseline。每条回答的好坏不是和 critic 估值比，而是和同组其他回答比。

### Q4：为什么每道题要采样多次？

训练时多采样用于构造 GRPO 组内优势。评测时多采样用于衡量模型在随机采样下的稳定性和上限，例如 Avg@32、Pass@32。

### Q5：vLLM 在这里解决什么问题？

RL 训练会频繁生成长回答，而且每个 prompt 要采样多条。vLLM 提供高吞吐推理和 KV cache 管理，降低 rollout 阶段成本。

### Q6：这个项目怎么和多模态岗位相关？

多模态岗位不仅要求会处理图像/视频，还要求理解大模型后训练。VLM 的 OCR、图表推理、视觉数学题、代码/工具调用、多模态 agent 也会用到 verifier、RL、采样评测、分布式训练和推理加速。

## 8. 简历写法草稿

项目名：Skywork-OR1 推理大模型后训练复现与源码解析

可写要点：

- 复现 Skywork-OR1 开源推理模型后训练流程，完成数据准备、vLLM rollout、rule-based reward、GRPO/PPO 更新和 AIME/LiveCodeBench 评测链路梳理。
- 阅读并注释核心源码，覆盖 Ray + FSDP 分布式训练、vLLM 高吞吐采样、数学/代码 verifier、KL 约束、adaptive entropy 和 rejection sampling。
- 分析 RLVR 在数学/代码推理任务中的训练信号构造方式，理解同题多采样下的 GRPO 组内优势归一化机制。
- 整理可迁移到多模态大模型后训练的经验，包括可验证奖励设计、长输出推理评测、模型分布约束和高吞吐生成工程。

面试时不要夸大成“独立训练出 32B 模型”。更稳妥的说法是“复现和解析训练链路，重点掌握后训练原理与工程实现；完整大规模训练受限于 GPU 资源”。

## 9. 下一步建议

1. 先精读已注释的 10 个文件，按本文第 5 节顺序走。
2. 在 Docker/Linux CUDA 环境中跑通 `pip install -e .`。
3. 先运行数据准备脚本，确认 `train_7b_math.pkl` 和 `train_7b_code.pkl` 能生成。
4. 若没有 8 卡资源，先改小模型、小 batch、小 response length 做冒烟实验。
5. 读 `papers/2505.22312.pdf`，重点看训练 pipeline、entropy collapse、数据过滤和 ablation。
