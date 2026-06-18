# Skywork-OR1 三层面试题库

这份题库用于模拟面试官追问，按难度分三层：

- 基础题：确认你知道项目做什么、流程是什么。
- 追问题：确认你不是背概念，而是真的理解原理和代码。
- 反拷打题：面试官质疑你、挑战你、问边界和细节时用。

建议练习方式：

1. 先自己口头回答。
2. 再看标准答案。
3. 最后看“考察点”，确认自己没有答偏。

---

## 0. 面试追问地图

```mermaid
flowchart TD
    A[项目介绍] --> B[数据]
    A --> C[训练流程]
    A --> D[数学原理]
    A --> E[工程实现]
    A --> F[实验评测]
    A --> G[简历真实性]

    B --> B1[数据过滤]
    B --> B2[可验证奖励]
    B --> B3[math/code 混合]

    C --> C1[rollout]
    C --> C2[reward]
    C --> C3[advantage]
    C --> C4[policy update]

    D --> D1[GRPO]
    D --> D2[PPO clipping]
    D --> D3[KL]
    D --> D4[entropy collapse]

    E --> E1[Ray]
    E --> E2[FSDP]
    E --> E3[vLLM]
    E --> E4[sandbox]

    F --> F1[AIME]
    F --> F2[LiveCodeBench]
    F --> F3[Avg@K]

    G --> G1[你到底做了什么]
    G --> G2[没有完整训练怎么办]
    G --> G3[和多模态岗位关系]
```

---

# 第一层：基础题

基础题要求回答简洁，重点是讲清楚项目主线。

## 1. 你做的 Skywork-OR1 项目是什么？

**标准答案：**

Skywork-OR1 是一个推理大模型后训练项目，目标是提升模型在数学和代码任务上的长链路推理能力。它不是普通监督微调，而是用数学答案校验和代码单测作为可验证奖励，再用 GRPO/PPO 类强化学习方法继续训练模型。

**考察点：**

- 能不能一句话讲清楚。
- 是否知道它是 reasoning LLM 后训练项目。
- 是否避免说成多模态项目。

---

## 2. 这个项目的整体流程是什么？

**标准答案：**

整体流程是：先准备数学和代码训练数据，然后用模型对每个 prompt 采样多条回答，再用 verifier 判断回答对错，得到 reward。之后用 GRPO 做组内相对 advantage，最后用 PPO-style loss 更新 actor 模型，并周期性在 AIME 和 LiveCodeBench 上评测。

**考察点：**

- 数据 -> rollout -> reward -> advantage -> update -> eval。
- 能不能把训练闭环讲完整。

---

## 3. 这个项目为什么适合写进大模型实习简历？

**标准答案：**

因为它覆盖了大模型后训练的核心能力，包括 RLVR、GRPO、verifier 奖励、vLLM 高吞吐生成、Ray + FSDP 分布式训练、长 CoT 评测和 entropy collapse 分析。这些能力对多模态大模型岗位也有迁移价值。

**考察点：**

- 不要只说“我会跑代码”。
- 要强调后训练、工程和原理。

---

## 4. RLVR 是什么？

**标准答案：**

RLVR 是 Reinforcement Learning with Verifiable Rewards，也就是用可自动验证的奖励做强化学习。比如数学题可以比较最终答案，代码题可以运行测试用例，所以不需要人工偏好模型也能给 reward。

**考察点：**

- 知道 RLVR 和 verifier 的关系。
- 能举数学/代码例子。

---

## 5. RLVR 和 RLHF 有什么区别？

**标准答案：**

RLHF 通常依赖人类偏好数据或 reward model；RLVR 依赖自动验证器。Skywork-OR1 中，数学题用答案校验，代码题用单测执行，所以 reward 更客观，也更适合推理任务。

**考察点：**

- RLHF: human preference。
- RLVR: automatic verifier。

---

## 6. GRPO 是什么？

**标准答案：**

GRPO 是一种组内相对优化方法。它会对同一个 prompt 采样多条回答，然后比较这些回答的 reward。比组内平均更好的回答 advantage 为正，模型会提高它的概率；更差的回答 advantage 为负，模型会降低它的概率。

**考察点：**

- 同一 prompt 多采样。
- 组内 reward 标准化。
- 不依赖单独 critic。

---

## 7. 为什么同一道题要采样多条回答？

**标准答案：**

因为 GRPO 需要组内比较。如果只采样一条，就没法知道这条回答相对同题其他回答是好是坏。多采样能提供更稳定的相对 advantage。

**考察点：**

- 知道 `rollout.n` / group size 的意义。

---

## 8. verifier 是什么？

**标准答案：**

verifier 是自动判断模型回答是否正确的模块。数学题会抽取最终答案并和标准答案做等价判断；代码题会抽取模型生成的代码，然后运行测试用例。

**考察点：**

- 数学 verifier。
- 代码 sandbox / unit tests。

---

## 9. vLLM 在项目里负责什么？

**标准答案：**

vLLM 负责 rollout 阶段的高吞吐生成。因为 RL 训练每一步都要生成大量长回答，普通 `transformers.generate` 效率不够，vLLM 可以更好地管理 KV cache 和批量生成。

**考察点：**

- rollout 是生成阶段。
- vLLM 是推理加速，不是训练算法。

---

## 10. Ray 和 FSDP 分别是什么？

**标准答案：**

Ray 负责分布式任务调度，把 actor、rollout、reward 等 worker 管起来。FSDP 负责模型参数分片，把大模型参数拆到多张 GPU 上，降低单卡显存压力。

**考察点：**

- Ray: orchestration。
- FSDP: memory/sharding。

---

## 11. AIME 和 LiveCodeBench 分别测什么？

**标准答案：**

AIME 主要测数学竞赛推理能力，LiveCodeBench 主要测代码生成和程序解决能力。Skywork-OR1 用这两个 benchmark 分别评估数学和代码推理。

**考察点：**

- benchmark 和能力对应关系。

---

## 12. Avg@K 是什么？

**标准答案：**

Avg@K 是同一道题采样 K 次后的平均正确率。它比只看 Pass@1 更能反映模型在随机采样下的稳定性。

**考察点：**

- 不要把 Avg@K 和 Pass@K 混淆。

---

# 第二层：追问题

追问题通常在你答完基础题后出现。面试官想确认你是不是只背了关键词。

## 13. GRPO 的 advantage 具体怎么算？

**标准答案：**

对同一个 prompt 的多条回答，先拿到每条回答的 reward，然后算这一组 reward 的均值和标准差。每条回答的 advantage 就是：

```text
advantage = (reward - group_mean) / group_std
```

这样模型学到的是这条回答相对同题其他回答好不好。

**考察点：**

- 会说公式。
- 会解释公式含义。

---

## 14. GRPO 为什么可以不训练 critic？

**标准答案：**

传统 PPO 需要 critic 估计 value 作为 baseline。GRPO 直接用同一个 prompt 的多条采样结果构造 baseline，也就是组内平均 reward，所以可以省掉单独 critic。

**考察点：**

- critic/value baseline。
- group baseline。

---

## 15. 如果同一个 prompt 的 16 条回答全对，会发生什么？

**标准答案：**

这组样本的 reward 没有差异，GRPO 很难得到有效相对 advantage。源码里会用 rejection sampling 把这类全对或全错的组过滤掉，避免浪费训练步。

**考察点：**

- 全对/全错都没有区分度。
- rejection sampling 的原因。

---

## 16. reward 为什么放在最后一个有效 token 上？

**标准答案：**

因为这是 outcome reward，整段回答只看最终是否正确。中间每个 token 没有单独标注，所以先把标量 reward 放到最后一个有效 token，再在 advantage 计算里扩展到 token-level 训练。

**考察点：**

- outcome reward vs process reward。
- token-level loss。

---

## 17. PPO clipping 解决什么问题？

**标准答案：**

PPO clipping 防止策略一次更新太大。它会比较新旧策略概率比，如果这个 ratio 超出范围，就截断掉，让更新更稳定。

**考察点：**

- ratio。
- clip。
- stable policy update。

---

## 18. KL penalty 和 PPO clipping 有什么区别？

**标准答案：**

PPO clipping 限制当前更新步不要偏离旧策略太远，是局部更新约束。KL penalty 则约束当前策略不要偏离参考模型太远，更像是保持模型整体分布和语言能力。

**考察点：**

- clipping: new vs old policy。
- KL: actor vs reference policy。

---

## 19. 为什么论文会讨论 No KL Loss？

**标准答案：**

因为 KL 约束太强会把模型拉回参考模型，导致 RL 后训练提升受限。论文发现某些阶段去掉 KL loss 反而能让模型继续探索和提升。

**考察点：**

- KL 不是越大越好。
- 约束和探索之间有 tradeoff。

---

## 20. entropy collapse 为什么会影响性能？

**标准答案：**

熵塌陷说明模型输出越来越单一，探索能力下降。RL 训练需要探索不同推理路径，如果模型过早固定在一种模式上，后面就很难发现更好的解法，所以测试性能会受影响。

**考察点：**

- entropy = diversity/exploration。
- premature convergence。

---

## 21. adaptive entropy control 怎么工作？

**标准答案：**

它会监控当前策略 entropy。如果 entropy 低于目标值，就增大 entropy loss 系数，鼓励模型探索；如果 entropy 高于目标，就降低系数，避免过度随机。

**考察点：**

- target entropy。
- dynamic coefficient。

---

## 22. 为什么高温采样能缓解熵塌陷？

**标准答案：**

高温采样会让输出分布更平滑，模型更容易采到不同解法。这样 rollout 数据更有多样性，训练时不容易过早收敛到单一路径。

**考察点：**

- temperature affects diversity。
- rollout diversity。

---

## 23. 多阶段训练为什么有效？

**标准答案：**

长 CoT 训练很费算力。一开始用较短上下文可以更快训练和降低成本；模型学到基础推理能力后，再逐步增加上下文长度，让它处理更长推理链。

**考察点：**

- context length。
- training efficiency。
- scaling to long CoT。

---

## 24. 数据过滤为什么重要？

**标准答案：**

如果题太简单，模型全答对，没有学习信号；如果题太难，模型全答错，也没有学习信号。过滤出中等难度、有区分度的问题，GRPO 才能学到相对优势。

**考察点：**

- model-aware difficulty。
- zero-advantage groups。

---

## 25. 数学 verifier 怎么判断两个答案等价？

**标准答案：**

先抽取模型最终答案，然后做字符串匹配；如果不完全相同，再用数学解析和等价验证，比如分数、小数、latex 表达式或 boxed 答案的语义等价。

**考察点：**

- extract answer。
- parse。
- verify。

---

## 26. 代码 verifier 怎么工作？

**标准答案：**

它会从模型回答中提取 Python 代码块，先用 AST 检查语法，然后把代码和测试用例拼起来，在子进程或 sandbox 中执行。如果测试全过，就给正确奖励。

**考察点：**

- extract code。
- AST validation。
- subprocess/unit tests。

---

## 27. 为什么代码执行要用 sandbox？

**标准答案：**

模型生成的代码可能死循环、占用大量内存或有危险操作。sandbox 可以限制执行时间和资源，避免训练过程被异常代码拖垮。

**考察点：**

- timeout。
- memory/resource control。
- robustness。

---

## 28. `RayPPOTrainer` 的核心职责是什么？

**标准答案：**

它是训练流程编排器，不直接写所有模型计算，而是调 Ray worker 完成生成、logprob、reward、actor update 等步骤。它负责把这些步骤组织成完整 PPO/GRPO 训练循环。

**考察点：**

- driver/orchestrator。
- worker RPC。

---

## 29. `ActorRolloutRefWorker` 为什么能有多个角色？

**标准答案：**

因为训练中同一套模型权重既要作为 actor 更新，又要作为 rollout 模型生成，还可能作为 reference policy 计算 KL。把这些角色放到一个 hybrid worker 里，可以减少权重同步和显存浪费。

**考察点：**

- actor / rollout / ref。
- hybrid engine。

---

## 30. 为什么生成后还要重新计算 old logprob？

**标准答案：**

vLLM 主要负责高效生成，训练 loss 需要和 actor 训练模型精确对齐的 logprob。所以生成后会用 actor 模型重新计算 old_log_probs，保证 PPO loss 的数据一致。

**考察点：**

- generation engine vs training actor。
- logprob correctness。

---

## 31. `RLHFDataset` 做了什么？

**标准答案：**

它读取 pkl 或 parquet 数据，把 prompt 套 chat template，再 tokenize 成 input_ids、attention_mask、position_ids。同时保留 reward_model、ability、extra_info 这些非张量字段给 reward manager 使用。

**考察点：**

- chat template。
- tensor/non-tensor batch。

---

## 32. 为什么要 left padding？

**标准答案：**

decoder-only 模型批量生成时，不同 prompt 长度不同。left padding 能让有效 token 在右侧对齐，方便后续生成 response 和计算 position_ids。

**考察点：**

- decoder-only batching。
- prompt alignment。

---

# 第三层：反拷打题

反拷打题是面试官怀疑你没真做、没真懂，或者想看你边界意识时会问的。

## 33. 你说你参与过这个项目，那你到底做了什么？

**标准答案：**

我主要参与的是源码梳理、训练流程分析和文档沉淀。我把数据准备、rollout、reward verifier、GRPO advantage、PPO 更新、vLLM 推理和 Ray + FSDP 分布式训练链路串起来，并给核心代码加了中文注释，整理成可复盘的项目文档。

**更硬一点的回答：**

我没有夸大自己完整训练了 32B 模型。我的贡献更偏向工程理解和复现准备，包括读代码、对齐论文和源码、拆解训练链路、整理面试和交接材料。

**考察点：**

- 真实性。
- 不吹大规模训练。
- 能说清楚自己的实际贡献。

---

## 34. 你真的完整复现了论文结果吗？

**标准答案：**

没有完整复现 32B 大规模训练，因为这需要大量 H800/GPU 资源。我做的是源码级复现和流程拆解，重点掌握训练链路、核心算法和工程实现。如果有算力，下一步可以按官方脚本跑小规模实验或完整训练。

**考察点：**

- 诚实。
- 知道资源限制。
- 不把“读懂源码”说成“复现 SOTA”。

---

## 35. 如果没完整训练，你凭什么说你理解这个项目？

**标准答案：**

因为我把训练链路从数据到评测完整对齐到了源码：数据脚本、Hydra 配置、RayPPOTrainer、vLLM rollout、YRRewardManager、GRPO advantage、PPO loss、评测生成都能对应到具体文件和函数。理解项目不等于必须重训 32B。

**考察点：**

- 源码路径熟悉度。
- 能把理解落到具体文件。

---

## 36. 这个项目和多模态岗位有什么关系？

**标准答案：**

虽然 Skywork-OR1 本身不是多模态项目，但它覆盖的是大模型后训练通用能力。多模态模型也需要 RL、verifier、长输出评测、分布式训练和高吞吐推理，比如视觉数学题、图表问答、OCR 推理和多模态 agent 都能迁移这些方法。

**考察点：**

- 不硬说多模态。
- 强调可迁移能力。

---

## 37. 面试官说“这不就是跑开源代码吗”，你怎么回答？

**标准答案：**

我不是只跑命令，而是把论文方法和源码实现对齐了。比如 GRPO 公式对应 `compute_grpo_outcome_advantage`，verifier 对应 `YRRewardManager` 和 `compute_score`，vLLM rollout 对应 `vllm_rollout.py`，训练循环对应 `ray_trainer.py`。我能讲清楚每一步为什么这样设计。

**考察点：**

- 区分跑代码和读懂系统。
- 能举源码例子。

---

## 38. 为什么不用 SFT 继续训练，而要用 RL？

**标准答案：**

SFT 只能模仿已有答案，难以通过试错发现更好的推理路径。数学和代码任务有自动 verifier，适合让模型自己探索多条解法，再用 reward 反馈优化，所以 RL 更适合继续提升推理能力。

**考察点：**

- imitation vs exploration。
- verifier 让 RL 成本降低。

---

## 39. GRPO 相比 PPO 的风险是什么？

**标准答案：**

GRPO 依赖组内 reward 差异。如果数据太简单或太难，同组全对/全错，就没有有效信号。它也依赖 verifier 的质量，如果 verifier 有误，模型会学到错误方向。

**考察点：**

- 不是只讲优点。
- 能讲限制。

---

## 40. verifier 会不会有问题？

**标准答案：**

会。数学 verifier 可能解析不了非标准格式，也可能对某些等价表达式判断错误；代码 verifier 可能受测试用例覆盖不足影响，也可能遇到多解输出、超时或资源问题。所以数据清洗和 verifier 稳定性非常关键。

**考察点：**

- false positive / false negative。
- test coverage。

---

## 41. 如果模型学会钻 verifier 漏洞怎么办？

**标准答案：**

这属于 reward hacking。可以通过提高 verifier 质量、增加隐藏测试、混合多种 verifier、人工抽检、限制输出格式、加入 KL/entropy 约束等方式缓解。

**考察点：**

- reward hacking。
- mitigation。

---

## 42. 为什么同一题全对也要过滤？全对不是好事吗？

**标准答案：**

从评测角度全对是好事，但从训练角度没有区分度。GRPO 学的是组内相对优势，如果 16 条都对，模型不知道哪条更值得提高概率，所以训练信号很弱。

**考察点：**

- 训练信号视角。
- 区分 evaluation 和 optimization。

---

## 43. 为什么同一题全错也过滤？

**标准答案：**

全错也没有相对正样本。所有 reward 都是 0，组内标准化后没有稳定的正向学习信号，容易浪费训练计算。

**考察点：**

- 全错没有正反馈。

---

## 44. 如果 group size 越大越好，为什么不用 128？

**标准答案：**

group size 大确实能提供更稳定的组内估计，但计算成本也会线性增加。每个 prompt 采样越多，rollout 显存、时间和 verifier 成本越高。所以要在性能和资源之间折中。

**考察点：**

- compute tradeoff。
- rollout budget。

---

## 45. temperature 越高越好吗？

**标准答案：**

不是。高温能增加探索，但太高会让输出质量下降、噪声变多。Skywork 使用高温是为了避免过早熵塌陷，但也需要配合 verifier、entropy control 和训练稳定性。

**考察点：**

- exploration vs quality。

---

## 46. 为什么 Avg@K 比 Pass@K 更稳定？

**标准答案：**

Pass@K 只要 K 次里一次对就算过，容易高估模型能力。Avg@K 看 K 次平均正确率，更能反映模型每次采样的稳定表现。

**考察点：**

- metric interpretation。

---

## 47. 如果 Avg@32 高但 Pass@1 低，说明什么？

**标准答案：**

说明模型有一定解题能力，但单次采样不稳定。多采样能碰到正确答案，但模型还没有稳定地把正确路径排在高概率位置。

**考察点：**

- stability vs best-of-N。

---

## 48. 为什么代码 reward 不一定完全可靠？

**标准答案：**

因为测试用例可能不完整，代码题可能有多个合法输出，sandbox 也可能因为超时、环境差异或资源限制误判。所以代码 verifier 的质量直接影响训练质量。

**考察点：**

- unit test incompleteness。
- multiple valid outputs。

---

## 49. FSDP 和 vLLM 同时用会有什么工程难点？

**标准答案：**

FSDP 训练侧参数是分片的，而 vLLM 推理侧需要可用于生成的权重布局。因此需要 sharding manager 在训练和推理之间同步或重组权重，同时还要管理显存和 KV cache。

**考察点：**

- training/inference weight format mismatch。
- sharding manager。

---

## 50. 为什么要 offload？

**标准答案：**

大模型训练显存压力很大，参数、梯度和优化器状态都占显存。offload 可以把部分状态放到 CPU 内存，牺牲一些速度换取更低 GPU 显存占用。

**考察点：**

- memory tradeoff。

---

## 51. 如果线上训练崩了，你会先看哪里？

**标准答案：**

我会先看日志定位阶段：是数据读取、rollout 生成、reward verifier、logprob 计算还是 actor update 崩了。然后看 Ray worker 日志、GPU 显存、vLLM cache、sandbox 超时和数据格式字段。

**考察点：**

- debugging flow。
- 能按模块定位。

---

## 52. 如果 reward 全是 0，你怎么排查？

**标准答案：**

先检查 verifier 是否正常，ground_truth 是否格式正确，模型输出是否包含 expected answer/code block，再检查 `reward_model` 和 `extra_info` 字段有没有丢。数学题看答案抽取，代码题看测试用例执行和 sandbox。

**考察点：**

- reward pipeline。
- data schema。

---

## 53. 如果训练中 entropy 很快降到接近 0，你怎么处理？

**标准答案：**

可以提高采样 temperature，启用或调大 adaptive entropy，减少 off-policy 数据复用，检查学习率和 PPO 更新步数，也可以调整 group size 和 batch size 增加 rollout 多样性。

**考察点：**

- entropy collapse mitigation。

---

## 54. 如果模型输出越来越长但准确率不升，说明什么？

**标准答案：**

可能模型学会了生成更长 CoT，但没有学到更有效推理。需要检查 reward 是否只鼓励最终正确、是否存在长度偏置、截断比例、训练数据难度和 verifier 质量。

**考察点：**

- length bias。
- token efficiency。

---

## 55. 为什么论文强调长 CoT？

**标准答案：**

复杂数学和代码题往往需要多步推理，长 CoT 给模型更多中间思考空间。但长 CoT 也会带来训练成本、截断和稳定性问题，所以项目用了多阶段上下文训练。

**考察点：**

- long reasoning benefits and costs。

---

## 56. 如果面试官问你“这个项目最核心的创新是什么”？

**标准答案：**

我会说它的核心贡献不是单个算法，而是一套可扩展的长 CoT RL 后训练配方：高质量可验证数据、GRPO 组内优化、多阶段上下文训练、高温采样、自适应熵控制、以及高吞吐分布式训练工程。

**考察点：**

- 不是只说 GRPO。
- pipeline thinking。

---

## 57. 如果面试官问“你能手写 GRPO advantage 吗”？

**标准答案：**

可以：

```python
scores = rewards.sum(dim=-1)
for uid in unique_uids:
    group = scores[uids == uid]
    mean = group.mean()
    std = group.std()
    scores[uids == uid] = (group - mean) / (std + 1e-6)
advantages = scores[:, None].repeat(1, response_len) * eos_mask
```

**考察点：**

- 能把数学公式落到伪代码。

---

## 58. 如果让你把它迁移到多模态任务，你会怎么做？

**标准答案：**

我会先找可验证任务，比如图表问答、OCR 数学题、视觉代码生成、定位/计数任务。然后设计对应 verifier，比如答案匹配、程序执行、检测框 IoU、OCR 文本匹配。训练框架仍然可以用 rollout -> verifier reward -> GRPO update。

**考察点：**

- 能迁移，不硬套。
- 知道多模态 verifier 怎么设计。

---

## 59. 如果视觉问答没有标准答案怎么办？

**标准答案：**

那就不适合直接 RLVR，可能需要人工偏好、LLM-as-a-judge、VLM-as-a-judge 或构造更可验证的子任务。RLVR 的前提是 reward 尽量可验证。

**考察点：**

- 知道适用边界。

---

## 60. 你怎么向非技术同学解释这个项目？

**标准答案：**

就是让模型做很多数学题和编程题，每题让它尝试多种解法，然后用自动判卷系统判断对错。模型会逐渐增加好解法的概率，减少坏解法的概率。

**考察点：**

- 能通俗解释。

---

# 4. 高频八股速记版

## 项目主线

```text
Skywork-OR1 = 长 CoT 推理模型 + 可验证奖励 + GRPO 强化学习 + 分布式高吞吐训练
```

## GRPO

```text
同题多采样 -> 组内 reward 均值/方差 -> 相对 advantage -> 更新策略
```

## RLVR

```text
用自动 verifier 给 reward，不依赖人工偏好
```

## verifier

```text
数学: 抽答案 + 等价判断
代码: 抽代码 + 单测 / sandbox
```

## entropy collapse

```text
模型过早变得确定 -> 探索下降 -> 性能受限
```

## vLLM

```text
负责 rollout 高吞吐生成，不是训练算法
```

## FSDP

```text
负责模型参数分片，降低训练显存
```

## Ray

```text
负责分布式 worker 调度和远程调用
```

---

# 5. 最后背诵版

如果时间很少，至少背下这段：

> 我参与的是 Skywork-OR1 推理大模型后训练项目。它的核心是 RLVR，也就是用数学和代码任务的自动 verifier 给 reward。训练时每个 prompt 采样多条回答，用 GRPO 在组内做 reward 标准化得到 advantage，再用 PPO-style loss 更新模型。工程上用 vLLM 做高吞吐 rollout，用 Ray 和 FSDP 做分布式训练。项目还重点处理了 entropy collapse、rejection sampling、多阶段上下文训练等问题。我主要负责源码梳理、训练链路分析、论文和代码对齐，以及沉淀面试和复盘文档。
