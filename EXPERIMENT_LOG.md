# Skywork-OR1 Execution and Experiment Log

## 1. 环境快照

- 日期：2026-07-13。
- OS：Windows + WSL Ubuntu 20.04。
- GPU：RTX 4060 Laptop，8 GB。
- Driver：576.52，驱动支持 CUDA 12.9。
- 基础 Python：3.12.3。
- Docker CLI：28.1.1；Docker Desktop Linux engine 未启动。
- 仓库：`main` / `origin/main`，提交 `5aaf786`。

## 2. 环境结论

官方完整训练环境应使用 Linux、Python 3.10、PyTorch 2.4.0 CUDA 12.4、vLLM 0.6.3 和 flash-attn。当前 Windows/8GB GPU 不能运行官方 7B × 8 GPU 长 CoT 配置。

本轮创建被 `.gitignore` 忽略的 `venv/`，用于 CPU 算法闭环和导入测试。

## 3. 安装命令

```powershell
python -m venv --system-site-packages venv
venv\Scripts\python -m pip install -e . --no-deps
venv\Scripts\python -m pip install hydra-core==1.3.2 tensordict==0.5.0 ray==2.38.0 math-verify
venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.0
venv\Scripts\python -m pip install intel-openmp
venv\Scripts\python -m pip install transformers==4.47.1 accelerate codetiming peft pybind11 pylatexenc wandb
```

Windows CPU wheel 的 `fbgemm.dll` 寻找 `libomp140.x86_64.dll`。本轮仅在 `venv/` 内使用：

```powershell
Copy-Item venv\Lib\site-packages\torch\lib\libiomp5md.dll `
  venv\Lib\site-packages\torch\lib\libomp140.x86_64.dll
```

这是本机 workaround，不应提交，也不是 Linux 训练环境步骤。

## 4. 实际测试结果

### 4.1 导入测试

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
venv\Scripts\python -m pytest -q tests\sanity\test_import.py
```

结果：`2 passed`。

### 4.2 DataProto/TensorDict 测试

```powershell
venv\Scripts\python -m pytest -q tests\utility\test_tensor_dict_utilities.py
```

结果：`12 passed, 1 warning`。warning 来自基础环境 metadata deprecation，不是断言失败。

### 4.3 Hydra 训练入口

```powershell
venv\Scripts\python -m verl.trainer.main_ppo --help
```

结果：成功打印完整 `ppo_trainer.yaml` 配置，证明入口、Hydra、Ray、trainer 和 verifier 导入链路可加载。

### 4.4 最小 verifier + GRPO 闭环

输入两组 digit-completion prompt，每组四条回答，reward 为 `[1,0,0,1]`。

结果：

```text
sequence_advantages = [0.866, -0.866, -0.866, 0.866, ...]
```

验证函数：

- `tests/e2e/envs/digit_completion/task.py::generate_ground_truth_response`
- `tests/e2e/envs/digit_completion/task.py::compute_reward`
- `verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage`

## 5. 失败记录和根因

### 失败 1：Hydra 缺 `omegaconf`

原因：第一次把 `--no-deps` 错误作用到所有轻量包。修复：重新安装轻量包及传递依赖。

### 失败 2：TensorDict 0.5 与 PyTorch 2.7.1 不兼容

错误：无法从 `torch.multiprocessing.reductions` 导入 `ForkingPickler`。

原因：仓库锁定 `tensordict<0.6`，实际预期 PyTorch 2.4.0。修复：venv 内安装 PyTorch 2.4.0 CPU。

### 失败 3：PyTorch CPU wheel 缺 OpenMP DLL 名称

错误：加载 `fbgemm.dll` 时 WinError 126。

修复：安装 `intel-openmp`，在 venv 内复制兼容 DLL 名称。Linux 环境无此步骤。

### 失败 4：`pyext` 不支持 Python 3.12

错误：`inspect.getargspec` 已移除。

结论：实际推荐 Python 3.10；没有为旧包修改仓库源码。

### 失败 5：完整 GPU 栈不可用

- vLLM 0.6.3 和 flash-attn 不支持当前 Windows 原生环境。
- Docker Desktop 引擎未启动。
- 8GB 单卡无法满足官方 7B/8 GPU 配置。

## 6. 完整 Linux/CUDA 环境命令

推荐官方镜像：

```bash
docker pull whatcanyousee/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te2.0-megatron0.11.0-v0.0.6
docker run --runtime=nvidia -it --rm --shm-size=10g --cap-add=SYS_ADMIN \
  -v /path/to/Skywork-OR1:/workspace/Skywork-OR1 IMAGE_TAG
cd /workspace/Skywork-OR1
pip install -e .
```

环境变量：

```bash
export CODE_PATH=/workspace/Skywork-OR1
export MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
export LIVECODEBENCH_DATA_PATH=$CODE_PATH/or1_data/eval/livecodebench/livecodebench_2408_2502
export WANDB_API_KEY=...   # 使用 WandB 时才需要
```

数据、训练、评测：

```bash
python or1_scripts/data_preprocess/download_and_filter_data_7b.py --local_dir or1_data/train
bash or1_scripts/train/7b_8k.sh
bash or1_scripts/eval/eval_7b.sh
```

## 7. 三个递进练习

### 入门：观察 GRPO 分组

- 修改：`verl/trainer/ppo/core_algos.py` 或新建独立测试，不改训练逻辑。
- 任务：记录每个 UID 的 reward、mean、std 和 advantage。
- 预期：看见正确回答为正 advantage、错误回答为负。
- 验证：添加 pytest，构造 `[1,0,0,1]`，断言符号和组内均值接近 0。
- 问题：全相同 reward 的 std 为 0；应验证 rejection sampling 或 epsilon 行为。

### 中级：增加 reward 诊断

- 修改：`verl/workers/reward_manager/yr_code.py`、对应测试。
- 任务：按 `data_source` 统计 verifier 成功、超时、异常、平均耗时，避免整批异常只得到全 0。
- 预期：能区分“模型答错”和“verifier 失败”。
- 验证：mock 一个成功 verifier、一个异常 verifier、一个超时 verifier。
- 问题：ProcessPool 中函数必须可 pickle；Windows spawn 和 Linux fork 行为不同。

### 高级：抽象 Rollout Backend

- 修改：`verl/workers/rollout/base.py`、`vllm_rollout.py`、`fsdp_workers.py`、测试。
- 任务：统一 HF/vLLM rollout 的输出契约，并增加一个 CPU toy backend。
- 预期：tiny arithmetic model 可在无 vLLM 环境跑端到端 smoke test。
- 验证：相同 prompt 在三种 backend 下都返回同样的 DataProto keys、shape、mask 语义。
- 问题：采样语义、logprob、padding、EOS、n>1 排序必须一致。

## 8. 下一步

1. 把最小 demo 固化为正式 pytest，避免仅靠临时命令。
2. 在 WSL/Docker 启动后验证 Linux sandbox。
3. 有多卡环境时先跑 tiny arithmetic E2E，再尝试小模型 GRPO，最后才是 7B。
