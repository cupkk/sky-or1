# Skywork-OR1 论文与源码对照笔记

论文文件：`papers/2505.22312.pdf`

论文题名：Skywork Open Reasoner 1 Technical Report

arXiv ID：2505.22312

页数：40 页

## 1. 论文主线

论文目标是给出一个可复现、可扩展的长 CoT 推理模型强化学习配方。Skywork 把这个配方称为 MAGIC：

```text
Multi-stage Adaptive entropy scheduling for GRPO In Convergence
```

它不是从零训练模型，而是在 DeepSeek-R1-Distill 系列长 CoT 模型上继续做 RL 后训练。

论文报告的主要增益：

- 32B：AIME24/AIME25/LiveCodeBench 平均准确率从 57.8% 提升到 72.8%。
- 7B：平均准确率从 43.6% 提升到 57.5%。
- Skywork-OR1-32B 在 AIME24 和 AIME25 上超过 DeepSeek-R1 与 Qwen3-32B。

源码入口：

```text
README.md
or1_scripts/train/7b_8k.sh
verl/trainer/main_ppo.py
verl/trainer/ppo/ray_trainer.py
```

## 2. GRPO 公式与源码对应

论文第 2 节写明：每个 prompt 采样 M 条回答，每条回答由 rule-based verifier 得到二值奖励，token-level advantage 为：

```text
A = (r - mean(group_rewards)) / std(group_rewards)
```

源码对应：

```text
verl/trainer/ppo/core_algos.py
  compute_grpo_outcome_advantage(...)
```

关键变量对应：

- `token_level_rewards`：每条 response 的奖励张量。
- `index` / `uid`：同一个 prompt 的多条 response 共享同一个 id。
- `id2score`：把同一 prompt 的 response 奖励聚到同一组。
- `scores = (scores - group_mean) / group_std`：GRPO 组内标准化。
- `scores.unsqueeze(-1).tile(...) * eos_mask`：把 outcome-level advantage 扩展到所有有效 response token。

训练循环中调用位置：

```text
verl/trainer/ppo/ray_trainer.py
  compute_advantage(...)
```

## 3. MAGIC 组件与源码对应

### 3.1 数据过滤

论文说训练前会过滤掉 base model 全对或全错的问题，训练中也会做在线过滤，保证训练集中持续存在有区分度的问题。

源码对应：

```text
or1_scripts/data_preprocess/download_and_filter_data_7b.py
or1_scripts/data_preprocess/download_and_filter_data_32b.py
or1_scripts/data_preprocess/download_and_filter_data_1p5b.py
```

这些脚本读取：

```text
extra_info.model_difficulty.DeepSeek-R1-Distill-Qwen-7B
```

并过滤出指定难度区间的问题。

训练中在线过滤对应：

```text
verl/trainer/ppo/ray_trainer.py
```

搜索：

```text
rejection_sample
solve_none
solve_all
valid_mask
```

含义：

- 同一 prompt 的多条 response 全错：`solve_none`。
- 同一 prompt 的多条 response 全对：`solve_all`。
- 这些组被过滤，因为 GRPO 组内 advantage 没有有效差异。

### 3.2 多阶段训练

论文强调先用短上下文训练，再逐步增加上下文长度，可以降低计算成本并保留 scaling 能力。

源码对应：

```text
or1_scripts/train/7b_8k.sh
or1_scripts/train/7b_16k.sh
or1_scripts/train/7b_32k.sh
or1_scripts/train/32b_16k.sh
or1_scripts/train/32b_32k.sh
```

关键参数：

```text
MAX_PROMPT_LENGTH
RES_LENGTH
data.max_prompt_length
data.max_response_length
actor_rollout_ref.rollout.response_length
```

论文训练配置：

- Skywork-OR1-7B：Stage 1 为 16K，Stage 2 为 32K。
- Skywork-OR1-32B：Stage 1 为 16K，Stage 2 为 24K。
- Skywork-OR1-Math-7B：8K -> 16K -> 32K -> 32K/更大 group size。

### 3.3 高温采样

论文选择 rollout temperature = 1.0 来增强探索，避免低温采样快速进入低熵状态。

源码对应：

```text
or1_scripts/train/7b_8k.sh
  TRAIN_TEMPERATURE=1.0
  actor_rollout_ref.rollout.temperature=$TRAIN_TEMPERATURE
```

vLLM 使用位置：

```text
verl/workers/rollout/vllm_rollout/vllm_rollout.py
  SamplingParams(...)
  update_sampling_params(...)
  inference_engine.generate(...)
```

### 3.4 Adaptive Entropy Control

论文第 3.2.5 节提出自适应熵控制：

```text
如果当前 entropy < target entropy，则增大 entropy loss coefficient。
如果当前 entropy > target entropy，则减小 entropy loss coefficient。
只有 entropy 低于目标时启用 entropy loss。
```

源码对应：

```text
or1_scripts/train/7b_8k.sh
  USE_ADAPTIVE_ENT=True
  TGT_ENTROPY=0.2
  MAX_ENT_COEF=0.005
  DELTA_ENT_COEF=0.0001

verl/trainer/ppo/core_algos.py
  class EntController
  compute_entropy_loss(...)
```

注意：

论文示例中提到 `tgt-ent=0.2` 和某些实验里的 `delta=0.005`。当前公开脚本里 `DELTA_ENT_COEF=0.0001`，这是源码配置与论文实验描述可能存在阶段/脚本差异的地方，面试时不要说死所有实验都用同一个 delta。

### 3.5 No KL Loss

论文第 3.2.6 节报告：在某些阶段 KL loss 会把 actor 拉回 reference policy，导致 AIME24 性能提升受限。因此最终发布模型设置 beta=0。

源码对应：

```text
or1_scripts/train/7b_8k.sh
  ENTROPY_COEFF=0.0
  actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF
```

需要谨慎区分：

- 论文讨论的是 actor loss 里的 KL loss。
- `ray_trainer.py` 里仍然有 `apply_kl_penalty(...)` 的通用实现。
- 当前脚本没有显式设置 `actor_rollout_ref.actor.use_kl_loss=True`，且 role mapping 默认没有启用单独 `Role.RefPolicy`，所以实际 7B 脚本主线更接近 No KL 训练。

相关源码：

```text
verl/trainer/ppo/ray_trainer.py
  apply_kl_penalty(...)
  if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):

verl/trainer/ppo/core_algos.py
  kl_penalty(...)
```

## 4. 数据准备与源码对应

论文第 6 节给出数据选择标准：

- Verifiable：题目必须能自动验证。
- Correct：答案或测试用例必须可靠。
- Challenging：base model 不能全部答对或全部答错。

数学数据来源包括 NuminaMath-1.5、DeepScaleR、STILL、Omni-MATH 和 2024 年前 AIME 等。

代码数据主要来自 LeetCode 和 TACO。

源码对应：

```text
or1_scripts/data_preprocess/download_and_filter_data_7b.py
```

当前开源脚本不是完整重跑论文的数据清洗 pipeline，而是下载官方已经整理好的：

```text
Skywork/Skywork-OR1-RL-Data
```

然后按模型难度过滤。

面试讲法：

> 我复现的是官方开源仓库里的训练数据准备入口。完整论文里的去重、质量筛选、LLM-as-a-judge 等离线数据工程已经体现在官方发布的数据集中；开源脚本主要负责按目标模型难度切分出适合训练的 math/code pkl。

## 5. Math & Code Verifier 与源码对应

论文第 7 节讲 verifier。

### 5.1 数学 verifier

论文最终采用 Math-Verify 路线：

1. 抽取 reasoning 后的最终答案。
2. 用 Math-Verify parser 解析。
3. 先做字符串匹配。
4. 再把 gold answer 包进 `boxed{}` 做语义 verify。

源码对应：

```text
verl/utils/reward_score/livecodebench/compute_score.py
  math_verify_reward_function(...)
```

也有另一路 deepscaler/prime math verifier：

```text
verl/utils/reward_score/deepscaler_math/math_reward.py
verl/utils/reward_score/prime_math/grader.py
```

但 Skywork 训练脚本传入的 `compute_score` 来自：

```text
from verl.utils.reward_score.livecodebench import compute_score as code_compute_score
```

### 5.2 代码 verifier / sandbox

论文第 7.2 节说代码执行 sandbox 支持：

- 标准输入输出测试。
- 函数单元测试。
- assertion-based tests。
- AST 语法检查。
- subprocess 执行。
- 多进程并行。

源码对应：

```text
verl/utils/reward_score/livecodebench/compute_score.py
  compute_score(...)
  convert_function_to_class_method(...)
  timeout_run(...)

verl/workers/reward_manager/yr_code.py
  parallel_compute_score(...)
```

重要限制：

- 当前代码使用 `signal.alarm`，适合 Linux 环境。
- Windows 原生不适合直接跑代码 sandbox；复现建议使用 Docker/Linux CUDA 环境。

## 6. 评测指标与源码对应

论文第 8 节评测：

- 最大生成长度：32768 tokens。
- AIME24/25：avg@32。
- LiveCodeBench：avg@4。
- temperature = 1，top-p = 1。

源码对应：

```text
or1_scripts/eval/eval_7b.sh
or1_scripts/eval/eval_32b.sh
verl/trainer/main_generation.py
```

当前 README 的 eval 脚本中 AIME 使用：

```text
data.n_samples=32
rollout.response_length=32768
rollout.temperature=0.6
```

这里与论文第 8 节的 `temperature=1` 存在公开脚本/论文配置差异。面试时建议说：

> 论文报告设置和开源脚本默认设置可能有差异，我复现时会记录具体命令和温度，不把 README 默认脚本直接等同于论文全部实验设置。

## 7. 面试可讲的技术闭环

建议按这个闭环讲，不要散讲文件名：

```text
数据准备
-> 选可验证且有难度的问题
-> 模型对每题采样多条长 CoT
-> 数学/代码 verifier 自动打分
-> GRPO 用同题组内奖励算相对 advantage
-> PPO clipping 稳定更新 actor
-> vLLM 提升 rollout 吞吐
-> adaptive entropy 防止过早熵坍塌
-> 多阶段上下文扩展提高训练效率
-> AIME/LiveCodeBench 多采样评测
```

对应源码路径：

```text
or1_scripts/data_preprocess/download_and_filter_data_7b.py
or1_scripts/train/7b_8k.sh
verl/utils/dataset/rl_dataset.py
verl/workers/rollout/vllm_rollout/vllm_rollout.py
verl/workers/reward_manager/yr_code.py
verl/utils/reward_score/livecodebench/compute_score.py
verl/trainer/ppo/core_algos.py
verl/trainer/ppo/ray_trainer.py
verl/trainer/main_generation.py
```

## 8. 需要避免的夸大

不要说：

- “我训练了 Skywork-OR1-32B。”
- “这是多模态大模型项目。”
- “我完整复现了论文所有数据清洗和所有大规模实验。”

可以说：

- “我复现并解析了 Skywork-OR1 的开源后训练链路。”
- “我重点掌握了 RLVR/GRPO、verifier、vLLM rollout、Ray+FSDP、entropy collapse 缓解等后训练关键技术。”
- “受限于算力，我把大规模训练拆成源码级复现和小规模冒烟实验，完整训练配置和论文配置已对齐分析。”
