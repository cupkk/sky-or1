# Skywork-OR1 研究进展日志 20260617

## 总体研究进展

### 项目目标

围绕 SkyworkAI/Skywork-OR1 项目做可复现、可理解、可用于简历与面试讲解的学习工程。当前阶段目标是：

1. 拉取并整理官方源码。
2. 下载项目相关论文或技术报告到本地。
3. 建立从环境配置、推理复现、训练/后训练流程到核心代码原理的学习路径。
4. 给新手需要重点掌握的核心代码添加详细中文注释，便于理解和面试复盘。

### 当前研究方向

- 先以官方仓库 `SkyworkAI/Skywork-OR1` 为真相源，不直接依赖本地空仓库已有的非官方远端。
- 优先理解 README、项目结构、模型/数据/训练脚本、推理入口、评测入口和论文链接。
- 先复现最小可运行流程，再逐步拆解底层原理。

### 已完成工作

- 确认当前工作目录为 `D:\github\sky-or1`。
- 确认本地目录目前只有 `.git`，尚无源码文件。
- 确认本地 git 远端当前指向 `https://github.com/cupkk/sky-or1.git`，不是用户指定的官方仓库。

### 关键发现

- 当前仓库还没有任何提交，`git status` 显示 `No commits yet on main...origin/main [gone]`。
- 需要先把官方仓库源码拉到本地，再进行代码阅读、复现和注释。

### 当前阻塞

- 暂无技术阻塞。下一步需要从官方仓库获取源码并确认论文入口。

### 下一步

1. 添加官方仓库远端并拉取源码。
2. 梳理项目结构和入口文件。
3. 查找并下载论文或技术报告。
4. 运行最小验证命令，确认环境依赖和复现路径。
5. 选择核心文件添加中文注释。

## 2026-06-17 更新

### 初始化研究日志

- 新建本日志文件，用于后续交接。
- 记录了本地仓库初始状态和官方仓库对齐计划。

### 拉取官方源码

- 保留原有 `origin=https://github.com/cupkk/sky-or1.git` 不改，新增远端 `official=https://github.com/SkyworkAI/Skywork-OR1.git`。
- 执行 `git fetch official` 成功，官方远端包含 `main`、`add_train_scripts`、`fix_eval_script` 等分支。
- 从 `official/main` 新建并切换到本地分支 `official-main`。
- 当前官方 `main` 对应提交为 `64e96afa213ae89d0ad21932106d3b8aafe9ace2`。
- 下一步读取 README、依赖文件和目录结构，确定复现路径与论文下载入口。

### 第一轮项目盘点

- 已读取 `README.md`、`requirements.txt`、`pyproject.toml`、`setup.py`、`or1_scripts` 和核心 `verl` 入口文件。
- 项目本质：Skywork-OR1 是数学/代码推理大模型的后训练项目，不是多模态模型项目。它适合在多模态大模型实习简历中包装为“LLM 后训练、强化学习、可验证奖励、分布式训练与推理工程”相关项目。
- 论文已下载到 `papers/2505.22312.pdf`，对应 arXiv ID `2505.22312`，题名为 `Skywork Open Reasoner 1 Technical Report`。
- 训练入口：
  - 数据准备：`or1_scripts/data_preprocess/download_and_filter_data_7b.py` 等脚本从 Hugging Face 数据集 `Skywork/Skywork-OR1-RL-Data` 下载并按模型难度过滤。
  - 训练脚本：`or1_scripts/train/7b_8k.sh` 等脚本通过 Hydra 参数覆盖启动 `python3 -m verl.trainer.main_ppo`。
  - 评测脚本：`or1_scripts/eval/eval_7b.sh` 通过 `python3 -m verl.trainer.main_generation` 生成多次采样并计算 AIME / LiveCodeBench 指标。
- 核心代码链路：
  - `verl/trainer/main_ppo.py`：训练总入口，初始化 Ray，选择 worker 类型，装配 reward manager。
  - `verl/trainer/ppo/ray_trainer.py`：PPO/GRPO 主训练循环，负责 rollout、奖励计算、rejection sampling、KL、advantage、actor 更新、验证和保存。
  - `verl/trainer/ppo/core_algos.py`：PPO/GRPO/REINFORCE++/ReMax 的优势函数、KL、policy loss 等核心算法。
  - `verl/workers/reward_manager/yr_code.py`：Skywork 训练脚本实际指定的 `reward_model.reward_manager=yr`，负责把生成文本交给数学/代码 verifier 得分。
  - `verl/utils/reward_score/livecodebench/compute_score.py`：数学答案与代码题单测奖励的统一入口。
  - `verl/utils/dataset/rl_dataset.py`：读取 pkl/parquet 数据、套 chat template、截断/过滤 prompt 并生成张量。
  - `verl/workers/rollout/vllm_rollout/vllm_rollout.py`：vLLM 生成封装，用于训练和评测阶段的高吞吐采样。
- 当前决定：优先给以上文件添加中文注释；`verl/third_party/vllm` 是第三方兼容层，暂不作为新手首轮逐行学习重点。

### 核心代码中文注释与复现指南

- 已给新手需要重点理解的核心文件添加中文注释，未改变训练逻辑：
  - `or1_scripts/data_preprocess/download_and_filter_data_7b.py`：解释官方 RL 数据下载、模型难度过滤、math/code 拆分和 ground truth 清理。
  - `or1_scripts/train/7b_8k.sh`：解释 7B/8K 训练脚本的 Hydra 覆盖参数、GRPO 组采样、adaptive entropy、vLLM rollout、rejection sampling、模型路径和保存路径。
  - `verl/trainer/main_ppo.py`：解释训练入口如何初始化 Ray、选择 FSDP/Megatron worker、装配 reward manager 并启动 `RayPPOTrainer.fit()`。
  - `verl/trainer/ppo/ray_trainer.py`：解释训练主循环：生成、打分、拒绝采样、重算 logprob、KL、advantage、actor update、验证和保存。
  - `verl/trainer/ppo/core_algos.py`：解释 GRPO advantage、PPO clipping、entropy loss、KL penalty。
  - `verl/workers/reward_manager/yr_code.py`：解释 Skywork 实际使用的 `YRRewardManager` 如何解码 response、调用 verifier、把 outcome reward 放到最后一个有效 token。
  - `verl/utils/reward_score/livecodebench/compute_score.py`：解释数学答案校验和代码单测 verifier 的分支逻辑，并把原先乱码的少量中文提示修为清晰中文。
  - `verl/utils/dataset/rl_dataset.py`：解释 pkl/parquet 读取、chat template、left padding、非张量字段保留和 uid/index 用途。
  - `verl/trainer/main_generation.py`：解释评测生成、每题多采样、结果保存和 Pass@K/Avg@K 指标计算。
  - `verl/workers/rollout/vllm_rollout/vllm_rollout.py`：解释 vLLM rollout、去除 left padding、n 条采样、EOS mask、KV cache 释放。
  - `verl/workers/fsdp_workers.py`：解释 FSDP worker 同时承担 actor/rollout/ref、权重同步、offload、生成和 actor 更新。
- 新增中文学习与复现文档 `docs/skywork_or1_中文学习复现指南.md`，内容包括项目定位、最小复现路线、代码阅读顺序、底层原理、面试问答和简历写法。
- 验证中发现并修复两个问题：
  - `core_algos.py` 有一段中文解释误放在 docstring 外导致 `SyntaxError`，已修正。
  - `or1_scripts/train/7b_8k.sh` 在 Windows 环境下混入 CRLF，`bash -n` 报 `fi` 附近语法错误，已将该脚本规范为 LF 行尾。
- 已通过：
  - `bash -n or1_scripts/train/7b_8k.sh`
  - `python -m compileall -q` 覆盖本次修改的 Python 文件
- 还需要做：
  - 最终运行 `git diff --check`，确保无尾随空白。
  - 如果后续进入真实复现，优先在 Docker/Linux CUDA 环境中执行，Windows 原生不适合跑代码 verifier 和 vLLM 训练链路。

### 论文阅读与源码对照

- 使用 `pypdf` 成功读取 `papers/2505.22312.pdf`，确认论文共 40 页。
- 论文目录重点：
  - 第 3 节：MAGIC 训练配方。
  - 第 4 节：policy entropy collapse 经验研究。
  - 第 6 节：Dataset Preparation。
  - 第 7 节：Math & Code Verifiers。
  - 第 8 节：实验设置和结果。
- 新增 `docs/skywork_or1_论文源码对照笔记.md`，把论文观点和源码路径对齐：
  - GRPO 公式对应 `verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage`。
  - 数据难度过滤对应 `or1_scripts/data_preprocess/download_and_filter_data_7b.py`。
  - rejection sampling 对应 `verl/trainer/ppo/ray_trainer.py` 中 `valid_mask/solve_none/solve_all` 逻辑。
  - adaptive entropy 对应 `or1_scripts/train/7b_8k.sh` 与 `verl/trainer/ppo/core_algos.py::EntController`。
  - verifier 对应 `verl/workers/reward_manager/yr_code.py` 和 `verl/utils/reward_score/livecodebench/compute_score.py`。
  - 评测生成对应 `or1_scripts/eval/eval_7b.sh` 和 `verl/trainer/main_generation.py`。
- 关键风险记录：
  - 项目不是多模态项目，应定位为推理 LLM 后训练项目。
  - 论文设置和公开脚本默认设置存在一些差异，例如某些 eval 温度和 adaptive entropy delta，复现时要以实际命令为准，不要混说。
  - Windows 原生环境不适合直接跑代码 sandbox 和 vLLM 训练，建议 Docker/Linux CUDA。

### 根据用户目标调整交付方向

- 用户明确表示现阶段不需要真正完整复现，而是需要：
  - 整体流程整理成文档。
  - 底层原理，包括代码怎么写、代码逻辑、数学原理。
  - 后续由 agent 模拟面试官提问，并给出浅显易懂的标准答案。
  - 列出项目中需要掌握的知识点、理论和八股文。
  - 多用绘图和可视化帮助理解。
  - 最终整理为可放在简历上的内容，按“任务背景 -> 方法/实现 -> 实验结果/产出”组织。
- 新增主文档 `docs/Skywork-OR1_项目学习总纲.md`，作为后续学习和面试训练的总入口。
- 该文档包含：
  - 项目一句话总结。
  - 总体流程 Mermaid 图。
  - 单次训练步 sequence diagram。
  - 数学原理：RLVR、GRPO、PPO clipping、KL、entropy collapse、rejection sampling、Avg@K/Pass@K。
  - 代码逻辑图和关键文件对应表。
  - 必须掌握的基础概念、工程概念、训练技巧。
  - 18 个模拟面试问题与浅显答案。
  - 按“任务背景 -> 方法/实现 -> 实验结果/产出”的简历模板版内容。
- 下一步建议：如果用户要求继续，应围绕该主文档扩展更细的“面试官追问版题库”和更丰富的图解，而不是继续做真实训练复现。

### 三层面试题库

- 根据用户要求新增 `docs/Skywork-OR1_三层面试题库.md`。
- 题库按面试官风格分为三层：
  - 基础题：确认项目定位、整体流程、RLVR、GRPO、verifier、vLLM、Ray/FSDP、Avg@K 等基础概念。
  - 追问题：深入追问 GRPO advantage、critic、rejection sampling、outcome reward、PPO clipping、KL、entropy collapse、多阶段训练、数据过滤、verifier 细节、代码路径。
  - 反拷打题：处理“你到底做了什么”“是否完整复现”“这不就是跑开源代码吗”“和多模态岗位有什么关系”“verifier 不可靠怎么办”“reward hacking 怎么办”等真实性和边界问题。
- 文档包含 Mermaid 面试追问地图和 60 个问题，每题包含标准答案和考察点。
- 新增高频八股速记版和最后背诵版，便于用户短时间准备面试。
