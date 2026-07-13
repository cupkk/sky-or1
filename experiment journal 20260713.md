# Skywork-OR1 研究进展日志 20260713

## 总体研究进展

### 项目目标

从第一性原理掌握 Skywork-OR1 的完整技术链路，形成可供代码学习、实验复盘、简历表达和面试训练使用的材料。当前重点不是在本机强行复现 7B/32B 多卡训练，而是先完成以下可验证闭环：

1. 建立仓库全局地图和高价值阅读顺序。
2. 解释数据、rollout、verifier、GRPO/PPO、Ray/FSDP/vLLM 的关系。
3. 追踪真实训练主链路，并把论文概念映射到代码。
4. 在当前 Windows/RTX 4060 8GB 环境中跑通轻量测试和最小算法闭环。
5. 保留完整训练所需的 Linux/CUDA 环境说明，避免把 smoke test 误写成完整复现。

### 当前研究方向

- 研究主线：LLM 推理后训练、RLVR、GRPO/PPO、可验证奖励、长 CoT rollout 和分布式训练。
- 岗位迁移：该仓库本身不是多模态项目，但 verifier 驱动的 RLVR 方法可迁移到视觉数学、图表问答、OCR 推理和多模态 agent。
- 学习策略：先读 Skywork 的配方差异和生产主链路，再读 verl 通用基础设施；首轮跳过多版本 vLLM 兼容层和 Megatron 细节。

### 已完成工作

- 重新扫描仓库目录、入口、配置、依赖、测试和示例。
- 创建五份根目录学习文档：
  - `PROJECT_MAP.md`
  - `FIRST_PRINCIPLES.md`
  - `EXECUTION_TRACE.md`
  - `MODULE_NOTES.md`
  - `EXPERIMENT_LOG.md`
- 在文档中加入项目架构图、数据流图、生产主执行链路图和核心模块依赖图。
- 识别并说明十个核心模块，建立论文概念、配置参数与函数实现的映射。
- 设计三个递进练习：GRPO 分组日志、reward 诊断、rollout backend 抽象。
- 创建本地 `venv/`，安装轻量依赖并完成导入、DataProto、Hydra 和最小 verifier/GRPO 验证。
- 确认论文 `papers/2505.22312.pdf` 和既有中文总纲、三层面试题库均在本地。

### 关键结论

- Skywork-OR1 的核心不是新模型结构，而是面向数学/代码推理的 RLVR 训练配方和工程系统。
- 同题多次采样提供组内相对信号；GRPO 用组内 reward 均值和标准差构造 advantage，从而省去 critic。
- 长序列、多样本的自回归 rollout 是主要计算瓶颈；vLLM、KV cache、动态 batch 和 FSDP 主要解决规模与效率问题。
- verifier 的可靠性决定 reward 的可靠性。verifier 错误会直接导致 reward hacking 或把执行异常误判为模型错误。
- 训练时不会直接信任 vLLM 生成阶段的 logprob，actor 会重新计算 `old_log_probs`，保证 PPO 比率使用训练策略的概率口径。
- rejection sampling 会过滤同题全对或全错的组，并改变实际有效 batch size；随后还需按 world size 对齐。

### 当前阻塞与边界

- 当前机器为 Windows + RTX 4060 Laptop 8GB，不能运行官方 7B、8 GPU、长 CoT 训练配置。
- vLLM 0.6.3 和 flash-attn 依赖 Linux/CUDA；Docker CLI 可用，但 Docker Desktop Linux engine 当前未启动。
- Python 3.12 会触发旧依赖 `pyext` 使用已移除 `inspect.getargspec` 的问题。完整环境应按 README 使用 Python 3.10。
- 因此本轮只声明轻量测试、入口加载和最小算法闭环通过，不声明完整 7B 训练复现完成。

### 下一步

1. 进入一问一答的面试训练，逐题检查对项目的真实理解。
2. 将当前临时最小闭环固化为正式 pytest，覆盖 group reward 和 advantage。
3. 在 Linux 多卡环境可用时，按 tiny arithmetic E2E、小模型 GRPO、7B 长 CoT 的顺序扩大验证范围。
4. 完成学习材料后，把简历项目表述严格限定为本人确实参与或复现验证过的工作。

## 2026-07-13 更新

### 仓库全局体检

- 当前分支：`main`。
- 当前基线提交：`5aaf786`，与 `origin/main` 对齐。
- 用户个人远端：`origin=https://github.com/cupkk/sky-or1.git`。
- 官方远端：`official=https://github.com/SkyworkAI/Skywork-OR1.git`。
- 关键入口确认：
  - 数据准备：`or1_scripts/data_preprocess/download_and_filter_data_*.py`
  - 训练：`or1_scripts/train/*.sh -> verl.trainer.main_ppo`
  - 评测：`or1_scripts/eval/*.sh -> verl.trainer.main_generation`
  - 主循环：`verl/trainer/ppo/ray_trainer.py::RayPPOTrainer.fit`
  - 算法：`verl/trainer/ppo/core_algos.py`
  - 奖励：`verl/workers/reward_manager/yr_code.py`

### 文档产出

- `PROJECT_MAP.md`：仓库地图、目录职责、入口、依赖、配置、测试、阅读顺序和四类架构关系。
- `FIRST_PRINCIPLES.md`：输入、输出、中间状态、稀缺资源、核心抽象、最贵步骤、正确性、性能、GRPO/PPO 数学和最小实现。
- `EXECUTION_TRACE.md`：本轮实际最小闭环、生产训练调用链、数据 shape 变化、输出位置、断点和 profiler 建议。
- `MODULE_NOTES.md`：十个核心模块的职责、接口、依赖、数据结构、隐含假设、修改点和坑点，以及论文到代码映射。
- `EXPERIMENT_LOG.md`：环境命令、测试结果、失败根因、Linux 完整环境和三项递进练习。

### 本地环境与修复

- 使用 `python -m venv --system-site-packages venv` 创建被 `.gitignore` 忽略的本地环境。
- 安装 PyTorch `2.4.0+cpu`、TensorDict `0.5.0`、Ray `2.38.0`、Transformers `4.47.1`、Hydra `1.3.2`、math-verify 等轻量依赖，并以 editable 模式安装本项目。
- 发现 TensorDict 0.5 与基础环境 PyTorch 2.7.1 不兼容，降到仓库预期的 PyTorch 2.4.0 后解决。
- Windows PyTorch CPU wheel 加载 `fbgemm.dll` 时缺少 `libomp140.x86_64.dll`。仅在忽略的 `venv/` 内把 Intel OpenMP 的 `libiomp5md.dll` 复制为兼容名称；没有修改仓库源码。
- `pyext` 在 Python 3.12 使用 `inspect.getargspec`，未修改旧依赖，记录为完整环境使用 Python 3.10 的依据。

### 实际验证结果

执行：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
venv\Scripts\python -m pytest -q tests\sanity\test_import.py
venv\Scripts\python -m pytest -q tests\utility\test_tensor_dict_utilities.py
venv\Scripts\python -m verl.trainer.main_ppo --help
```

结果：

- `tests/sanity/test_import.py`：`2 passed`。
- `tests/utility/test_tensor_dict_utilities.py`：`12 passed, 1 warning`；warning 为基础环境 metadata 弃用提示。
- Hydra 训练入口成功加载并打印配置。
- digit-completion 最小闭环成功：两组 prompt 的 reward 均为 `[1,0,0,1]`，GRPO sequence advantage 为 `[+0.866,-0.866,-0.866,+0.866]`。
- `git diff --check` 通过，无空白错误。

### 代码级发现

- `verl/trainer/ppo/ray_trainer.py` 中 `dataprotoitem_to_dataproto` 重复定义。
- 训练脚本注释称重复 math 数据，但实际 `train_files` 是 code 两份、math 一份。
- `YRRewardManager` 把 outcome reward 写在 response 最后一个有效 token；response 为空时需要注意索引为 `-1` 的风险。
- `parallel_compute_score` 接收 `timeout` 参数，但该参数没有直接约束每个 Future 的等待时间。
- LiveCodeBench sandbox 使用 `signal.SIGALRM`，说明代码 verifier 路径面向 Linux。
- `main_generation.py` 同时统计首样本准确率、全部采样平均准确率和 Pass@K，阅读结果字段时不能混用。

### 保留与清理说明

- 保留：五份根目录学习文档、论文 PDF、既有中文总纲和三层面试题库、两份日期日志。
- 不提交：`venv/`、本机 DLL workaround、临时缓存和任何模型权重。
- 不修改：第三方 vLLM 兼容层和旧依赖源码，避免为了 Windows smoke test 改变正式训练实现。

### 下一位 Agent 交接

1. 开始前先读本日志，再按 `PROJECT_MAP.md` 的顺序阅读代码。
2. 不要把当前测试结果表述为完整模型训练复现；真实边界是 CPU 导入、数据协议、Hydra 入口和 verifier/GRPO 最小闭环。
3. 面试训练必须每次只问一个问题，等待用户回答后指出概念漏洞、给出标准答案，再进入下一题。
4. 后续发布时只推送用户个人仓库 `origin/main`，不要推送官方 `official/main`。
