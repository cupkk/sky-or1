# Skywork-OR1 研究进展日志 20260715

## 总体研究进展

### 项目目标

围绕 Skywork-OR1 建立一套可供新手真正理解、可用于浙江大学李环老师面试的学习材料。此前已经完成仓库地图、代码主链路、模块笔记和轻量验证，但用户反馈这些材料偏代码文档，无法形成对原理和流程的直观理解。本轮目标因此调整为：只保留必要代码查证入口，以“一个 prompt 如何经过 rollout、verifier、GRPO、PPO 变成模型更新”为主线，统一讲清基础知识、公式、系统瓶颈、多模态迁移和面试回答。

### 当前研究方向

- 算法主线：自回归语言模型、SFT、策略梯度、baseline、GRPO、PPO clipping、entropy、KL、on/off-policy。
- Skywork 配方：model-aware data filtering、rejection sampling、multi-stage context、高温采样、自适应熵控制和 no-KL。
- 系统主线：Ray 编排、FSDP 训练、vLLM rollout、KV Cache、continuous batching、长尾序列调度和 speculative decoding。
- 多模态迁移：视觉数学、图表问答和 GUI agent 等可构造 verifier 的任务。
- 面试定位：承认 Skywork-OR1 是文本 reasoning LLM 项目，用后训练、数据质量和高效推理方法与多模态岗位建立可辩护的联系。

### 已完成工作

- 阅读本地 40 页技术报告的 Preliminaries、MAGIC、entropy collapse、resource allocation、dataset、verifier 和 experiments 章节。
- 核对 `compute_grpo_outcome_advantage`、`compute_policy_loss`、`EntController`、`RayPPOTrainer.fit`、`YRRewardManager`、vLLM rollout 和评测脚本。
- 通过浙江大学教师主页检索确认李环老师为计算机学院百人计划研究员、博士生导师。
- 核对李环老师和 SuDIS Lab 的公开方向：Data-centric AI、Efficient AI、LLM/Multimodal LLM/Agentic AI 推理优化、speculative decoding、KV Cache、动态序列调度、异构资源优化、vLLM/verl 定制和 RL 数据流程。
- 新建统一讲义 `docs/Skywork-OR1_原理精讲与李环老师面试准备.md`。

### 关键结论

- 用户当前最需要的不是更多代码注释，而是建立稳定的因果链：为什么 SFT 不够，为什么同题多采样，为什么用组内 advantage，为什么还要 clipping 和 entropy，为什么长 CoT 的主要成本是 rollout。
- 对 `[1,0,0,1]`，总体标准差会给出 `+1/-1`；仓库使用 PyTorch 默认样本标准差，实际得到约 `+0.866/-0.866`。该差异已在讲义中解释。
- PPO clipping 与 KL penalty 约束的参照不同：前者约束相对 rollout old policy 的单步变化，后者约束相对固定 reference policy 的长期漂移。
- temperature 是人为设置的采样旋钮，entropy 是策略概率分布的统计量；top-p 控制候选集合。这三个概念必须分开。
- Skywork-OR1-32B 的论文统计中，1000 步总耗时 309 小时，rollout 223 小时，占 72.1%；policy update 27 小时，占 8.7%。这与李环老师的 Efficient AI 和推理优化方向直接相关。
- 技术报告摘要的三项平均增量与表 13 逐项数值无法直接对齐：摘要报告 7B/32B 为 `+13.9/+15.0`，表 13 直接求均值约为 `+13.4/+9.8`。讲义已要求面试时逐 benchmark 报分并明确口径。
- 对李环老师问题的预测必须标明依据和不确定性，不能描述为老师本人提供的题库。

### 当前边界

- 本轮没有执行完整 7B/32B 训练，也没有新增这类复现声明。
- speculative decoding、PagedAttention 和多模态扩展用于解释公开研究方向与可行迁移，不是 Skywork-OR1 仓库已经实现的功能。
- 项目陈述继续使用已验证边界：完成源码链路分析、中文解释和 verifier 到 GRPO advantage 的最小闭环；本地硬件不支持官方多卡长 CoT 训练。

### 下一步

1. 对统一讲义做 Markdown、公式、Mermaid 和事实一致性校验。
2. 提交并推送到用户个人仓库 `origin/main`。
3. 用户阅读后进入一问一答模拟面试；每次只问一个问题，根据回答指出理解漏洞。

## 2026-07-15 更新

### 论文和代码复核

本轮重新阅读 `papers/2505.22312.pdf`，确认技术报告的关键实验数据和配置：7B/32B 基于 DeepSeek-R1-Distill-Qwen；最终配方采用 temperature 1.0、clip ratio 0.2、target entropy 0.2、rejection sampling 和 no-KL；7B 的正式配方从 16K 扩展到 32K，32B 从 16K 扩展到 24K。论文表 13 使用 temperature 1、top-p 1，而仓库 `or1_scripts/eval/eval_7b.sh` 默认 temperature 0.6，讲义已明确两者不能混用。

代码核对确认：

- `core_algos.py::compute_grpo_outcome_advantage` 按 UID 聚合同题 reward，使用 `torch.std`，再把序列 advantage 复制到全部有效 response token。
- `core_algos.py::compute_policy_loss` 使用新旧 log probability 的指数比和 PPO clipping。
- `EntController` 在当前熵低于目标值时增加系数，高于目标值时降低系数，并把系数限制在脚本配置范围内。
- `ray_trainer.py::fit` 先 rollout 和 verifier，再过滤全对/全错组，随后由 actor 重算 old log probability、计算 advantage 并更新策略。
- `YRRewardManager` 将 outcome reward 写在最后一个有效 response token。
- `main_generation.py` 直接计算首样本正确率、所有样本平均正确率和至少一次成功率；讲义同时补充了代码生成文献中的标准 Pass@K 无偏估计式。

最终数值验证时，第一次把 `compute_grpo_outcome_advantage` 的 `index` 传成 `torch.Tensor`，触发 `KeyError: tensor(0)`。生产链路实际传入 `data.non_tensor_batch['uid']`，即 NumPy 字符串数组；改用四个相同 UID 后，函数输出 `[0.866024, -0.866024, -0.866024, 0.866024]`，与讲义推导一致。这说明该函数虽然主要张量参数是 Tensor，但分组索引依赖可稳定比较的非张量 UID。

### 统一讲义内容

新讲义共约 2.4 万中文字符，使用连续解释而非碎片化清单。主要内容包括：

- 从 token、自回归概率和 SFT loss 开始建立 RLVR 动机。
- 用最小伪代码和 Mermaid 图讲完整训练闭环。
- 用 `[1,0,0,1]` 手算 GRPO，并解释样本标准差细节。
- 推导策略梯度、GRPO advantage、PPO ratio/clipping、entropy、adaptive entropy、KL、on/off-policy 和截断目标。
- 解释 MAGIC 的数据、训练策略和 loss 设计及其消融证据。
- 解释 Ray、FSDP、vLLM、DataProto、KV Cache、PagedAttention、continuous batching 和 speculative decoding。
- 解释 Avg@K、仓库 Pass@K 与标准无偏 Pass@K 的差异。
- 补充 Transformer、attention、cross-entropy、并行策略和多模态架构基础知识表。
- 根据李环老师公开方向设计 16 个高概率专业问题及浅显但可深入追问的回答。
- 提供 90 秒陈述、一句话陈述、诚实简历表述和面试前自检方法。

### 李环老师公开信息依据

- 浙江大学教师主页：`https://person.zju.edu.cn/lihuan`
- 个人学术主页：`https://longaspire.github.io/`
- SuDIS Lab：`https://sudis-zju.github.io/`

公开主页显示老师当前关注 data-centric、resource-efficient、scalable AI，个人主页列出 Efficient LLM inference 和 fine-tuning 方向；浙江大学主页的实习课题进一步明确投机解码、KV Cache、动态序列调度、异构资源、sgLang/vLLM/verl，以及领域模型数据准备。这些公开内容用于决定面试题重心，没有推断私人面试偏好。

### 保留与清理

- 保留新讲义、既有五份代码学习文档、论文和历史日志。新讲义作为明天面试前的主入口，旧文档作为需要查代码时的辅助材料。
- Playwright 浏览浙江大学主页生成的 `.playwright-mcp/` 是临时工具输出，不应提交。
- 不提交任何虚拟环境、浏览器缓存或本机依赖文件。

### 最终校验

- 主讲义约 2.5 万字符、554 行。
- 4 个代码块/图示对应 8 个 fence，23 个公式块对应 46 个 `$$`，行内公式分隔符均成对。
- 所有本地论文和代码查证路径存在。
- `git diff --check` 无空白错误。

### 下一位 Agent 交接

后续不要继续扩展一整套新文档。应以 `docs/Skywork-OR1_原理精讲与李环老师面试准备.md` 为唯一面试主讲义，根据用户反馈做局部补充。模拟面试必须每次只问一个问题，先听用户回答，再指出缺失的因果关系、公式含义和系统权衡。
