# 流式采样收益测试报告

> 配套方案：`docs/designs/streaming_bench_plan.md`
> 配套脚本：`cookbook/rl/reward_loop/bench_streaming_local.py`
> 数据：`results/bench_timeline.jsonl`、`results/bench_summary.csv`

## 0. 当前状态

- **v2 实跑已完成**（训练机，本地 ray 模式，2026-08-27/28），本报告第 3~5 节为实测数据。
- 脚本纯逻辑部分通过桩模块验证；脚本已修复的问题见第 6 节。
- `results/bench_summary.csv` 共 61 行数据行 + 1 行 level1 汇总行；`bench_timeline.jsonl` 为逐事件时间线。

## 1. 运行方法

```bash
# 冒烟验证（极小配置，验证链路与正确性检查可用）
cd cookbook/rl/reward_loop
BENCH_RUNS=smoke python bench_streaming_local.py

# 完整测试（基础配置 + 单变量扫描，约 1-2 小时，需 ≥2 卡）
python bench_streaming_local.py
```

产物：`results/bench_summary.csv`、`results/bench_timeline.jsonl`，
以及 **`results/level1_texts.json`**（Level-1 逐对文本：每条序列的 A/B 文本、
reward、ground truth、首个分叉位置，用于人工核对文本差异形态）。

可用环境变量：`TWINKLE_MODEL_ID`（默认 `ms://Qwen/Qwen3.5-4B`，可传本地目录）、
`TWINKLE_DATASET_ID`（默认 `ms://modelscope/gsm8k`）、`TWINKLE_MODEL_GPUS` /
`TWINKLE_SAMPLER_GPUS`（默认各 1）、`TWINKLE_BATCH_SIZE`、`TWINKLE_MAX_STEPS`、
`TWINKLE_REWARD_DELAY_MS`、`TWINKLE_REWARD_NUM_WORKERS`、`BENCH_OUT_DIR`、
`BENCH_RUNS`（`all` 或 `smoke`）。

## 2. 测试环境

| 项 | 值 |
|---|---|
| 机器 / GPU | 训练机（本地 ray，无 twinkle-server）；昇腾 NPU ×4（`ASCEND_RT_VISIBLE_DEVICES=2,3,4,5`） |
| 部署模式 | `twinkle.initialize(mode='ray')`，model 1 卡 + sampler 1 卡 |
| 模型 | 默认 `ms://Qwen/Qwen3.5-4B`（可覆盖为本地路径） |
| 数据集 | GSM8K train（7473 行）；**本地 jsonl**（`messages` + `gold_answer` 列），经 `DatasetMeta(data=rows)` 内存路径加载，完全离线 |
| 采样参数 | temperature=1.0, top_p=0.95, logprobs=1, num_samples=1 |
| 每步序列数 | batch4 × gen（base/tok/d200/d1000=16，gen2=8，gen8=32） |

## 3. 正确性验证结果

### 3.1 Level 1（确定性：greedy + 固定 seed，训练前）

| 检查项 | 结果 |
|---|---|
| 样本总数一致（16=16） | ✅ 通过 |
| tokens 逐位一致 | ❌ 16 条中 12 条不一致（最大长度差 93 token） |
| logprobs 逐位一致 | ❌ 16/16 不一致（含 4 条 tokens 相同但 logprob 浮点值不同） |
| decoded 文本一致 | ❌ 12/16（跟随 tokens） |
| reward 值逐样本一致 | ✅ 通过（但平凡：见下方「ground truth 缺失」） |
| 优势（GRPOAdvantage）一致 | ✅ 通过（纯函数，且为 rewards 一致的逻辑推论） |

**文本差异形态（`results/level1_texts.json` 逐对核查）**：12 条发散全部为
**措辞/风格级**——首分叉位置散布于第 6~285 token，分叉后推理路径与最终答案
完全一致（72 / $10 / $5 / 42 全部同解同答案），**无一条改变最终答案**。
另发现：同 prompt 在路径 A 内部（同一批、greedy、同 seed）也出现 3+1 的文本
变体，说明引擎在请求级存在非确定性，"批形状数值差异"只是部分解释。

**⚠️ ground truth 缺失（影响 reward 类检查的解释）**：Level-1 的 prompt 来自
未 map 的原始行，诊断显示行里**没有 `answer` 字段**、`user_data` 也没有
`ground_truth`，导致默认奖励恒为 0（连答案正确的条目也判 0）——
`rewards_identical` / `advantages_identical` 是"全零"的平凡真，Level-2 的
奖励确定性检查同理平凡。叠加第二个问题：模型输出为 `**Final Answer:** 72
clips.` 形态，不含 `\boxed{}`/`####`，`GSM8KAccuracyReward` 本也无法提取。
**修复（已落地并验证）**：(1) 数据集行实为 `messages` + **`gold_answer`**
列（无 `question`/`answer`）；`_MessagesGSM8KPreprocessor` 从 `gold_answer`
/assistant 参考消息提取答案写入 `user_data`，并从 prompt 移除参考消息；
(2) system prompt 要求 `\boxed{}` 输出；(3) `_score` 容错提取兜底
（`\boxed{}` → `####` → 末尾数字）；(4) 本地 jsonl 经 `DatasetMeta(data=rows)`
内存路径加载——**完全离线，绕开 modelscope loader**。
纯 CPU 验证脚本 `verify_dataset_reward.py` 实测通过
（7473 行 map、GT 全部非空、对/错奖励 1.0/0.0）。

**Level-1 最终实测（数据适配 + 引擎级流式，base 档 16 对）**：
`all_ok: true` —— tokens / logprobs（max diff 0.0）/ decoded / rewards /
advantages 全部逐位一致。引擎级流式与整批采样在相同输入与 greedy 参数下
**位级等价**；此前 v2 的 12/16 token 发散（提示词无 system 指令、参考消息
混入 prompt）已随修复消失。`rewards_identical` 在真实 ground truth 下
成立，非平凡。

**结论（最终）**：Level-1 全部检查通过（位级等价达成）；reward/advantage
层面的一致建立在真实 ground truth 之上（非平凡真）。

### 3.2 Level 2（语义：随机采样）

| 检查项 | A（批量） | B（流式） |
|---|---|---|
| 无丢样本 / 无重复 item_id | ✅ 全部通过 | ✅ 全部通过 |
| handle 结果与 item_id 对齐 | ✅ | ✅ |
| reward 函数确定性（重打分一致） | ✅ | ✅ |
| 优势组均值归零 | ✅ | ✅ |

**结论：Level-2 全部通过（60 个 run-路径-步骤检查点均 True）。**

## 4. 收益测量结果（v4：引擎级流式，最终数据）

> 数值 = 该 run 该路径所有步骤的均值（秒）。`t_submit` 均在 0.01s 以下，省略。
> Path B 为引擎级流式（`vLLMSampler.sample_sequences_to_queue`，一次远程调用内
> 并发调度全部序列，事件经 Ray 队列逐条回传）。v2/v3（legacy N 调用）数据见
> 附录 D 对照表。

### 4.1 基础配置（batch=4, gen=4, max_tokens=1024, delay=0，6 步）

| 指标 | A（批量） | B（引擎流式） | 说明 |
|---|---|---|---|
| t_sample | 19.6 | 20.3 | **B ≈ A，合批恢复**（legacy 曾为 95.9，A 的 8×） |
| t_collect（奖励尾部等待） | 0.00 | 0.00 | delay=0 均接近 0 |
| t_train | 3.7 | 3.6 | 基本一致 |
| **t_total（s/步）** | **24.4** | **24.5** | **B/A = 1.00×** |
| seq/s | 0.65 | 0.65 | 持平 |
| reward_head_start | 19.7 | ~16 | 合批后序列几乎同时完成，重叠窗口变小（正常特征） |
| reward_tail_after_sample | 0.00 | -0.001 | B 最后一条奖励早于采样结束就绪（负值） |

### 4.2 全配置扫描（engine 流式，每 run 4 步）

| run | 变量 | A t_total | B t_total | B/A |
|---|---|---|---|---|
| base | — | 24.4 | 24.5 | **1.00×** |
| gen2 | gen=2 | 21.2 | 21.2 | 1.00× |
| gen8 | gen=8 | 32.7 | 33.0 | 1.01× |
| tok512 | max_tokens=512 | 17.4 | 18.2 | 1.05× |
| tok2048 | max_tokens=2048 | 44.6 | 44.1 | 0.99× |
| d200 | delay=200ms | A t_collect=0.21，B t_collect≈0.4~1.6 | 25.5 | 26.9 | 1.05× |
| d1000 | delay=1000ms | A t_collect=1.005，B t_collect≈1.7~7.9 | 26.3 | 31.5 | 1.20× |

**扫描结论**：除 d1000（奖励昂贵场景）外，B/A 全部落在 0.99~1.05×——引擎级流式
在采样层面与整批持平，且保留逐条事件语义。d1000 的残余代价来自奖励提交粒度
（16 个单 item handle 受 pipeline 2 线程池限制，collect 1.7~7.9s vs A 的 1.0s），
是分块粒度问题而非采样问题（见 §5 结论 4）。

## 5. 结论

1. **正确性**：Level-2（结构/对齐/奖励确定性）全部通过；**Level-1 在数据适配修复 +
   引擎级流式下首次整体通过**（tokens/logprobs/decoded/rewards/advantages 全部一致，
   `semantic_ok=True`）。此前 v2 的 12/16 token 发散源于提示词无 system 指令 +
   参考消息混入 prompt + 批形状数值差异，修复提示词后消失。
2. **引擎级流式合批恢复（本次核心成果）**：`vLLMSampler.sample_sequences_to_queue`
   一次远程调用内并发调度全部序列，vLLM 保持合批——B 的 t_sample 回到 A 的水平
   （base 20.3s vs 19.6s），**t_total B/A = 0.99~1.05×（除 d1000）**，与此前
   legacy N 调用实现的 3.6~7.1× 形成对照。逐条事件语义保留（Level-1 对齐验证通过）。
3. **流式重叠窗口随合批自然消失**：序列几乎同时完成时，「首条奖励早于采样结束」
   的窗口变小（head_start ≈ t_sample），tail 出现负值（奖励先于采样结束就绪）——
   这是引擎级流式的正常特征，整体耗时反而最优。
4. **残余代价：奖励提交粒度（分块）**：d1000 下 B 的 t_collect（1.7~7.9s）> A
   （1.0s），因逐序列提交产生 16 个单 item handle，受 pipeline 线程池（=num_workers=2）
   限制。与采样方式无关；若奖励昂贵且打分本身可并行，改为整批提交（大 chunk）可消除。
5. **实践建议**：本地 ray 模式下使用引擎级流式（`sample_sequences_to_queue`）即可在
   保留逐条事件语义的同时获得整批采样性能；奖励端按成本选择提交粒度
   （便宜奖励整批提交；昂贵且可并行的奖励逐条或小批量提交）。

## 6. 脚本已修复的问题

1. **DataLoader actor 重名崩溃**：twinkle 的 Ray actor 名含调用点行号，循环内
   同处新建多个 `DataLoader` 会报 `ActorAlreadyExistsError` —— 已为每个
   DataLoader 传唯一 `instance_id`；
2. **结果只在全部跑完后写盘**：中途崩溃会留下空 `results/` —— 已改为每完成
   一个 run 即落盘，异常时也会先保存已完成的 run 再抛错；
3. **输出目录依赖 CWD** —— 已改为默认输出到仓库根 `results/`（绝对路径，
   启动日志会打印实际路径）；
4. **Path B 奖励提交时机缺陷（v1）**：提交在采样循环结束后才执行，
   `reward_head_start ≈ t_sample`，重叠未发生 —— v2 已改为每条序列完成即提交，
   实测重叠生效（见 4.1）。
5. **ground truth 契约缺口**：原始行无 `answer`/`ground_truth`，奖励恒 0——
   已对齐 `GSM8KProcessor` 映射 + 容错答案提取（v3，待重跑验证非平凡结果）。

## 7. v3：本地引擎级逐序列流式（消除 B 的串行化代价）

**背景**：v2 的 Path B 用 N 次并发远程调用模拟流式，被 sampler actor 串行执行，
采样耗时 ≈ N × 单序列时间（8/16/32 序列时 3.6×/6.0×/7.1×）。结论是"本地模式
需要引擎级逐序列事件流"。

**实现（v3）**：库层新增 `vLLMSampler.sample_sequences_to_queue`
（`src/twinkle/sampler/vllm_sampler/vllm_sampler.py`）——一次远程调用
（`dispatch='all', execute='first'`）内，actor 事件循环并发调度全部序列
（与 `sample()` 相同的 gather 形态，vLLM 保持合批），完成一个即向 Ray 队列推
`(index, SampleResponse)` 事件，全部完成后推哨兵。bench 的 Path B 改用该接口：
driver 线程池持调用、主线程按完成顺序收事件 → 逐条提交奖励。

**预期效果**（待实测）：
- t_sample 恢复到整批水平（≈ Path A），B/A 收益比回归 1.0 附近；
- 保留逐序列完成即提交的重叠（`reward_head_start` 远小于 t_sample）；
- 备选回退：`BENCH_PATH_B_STREAM=legacy` 可回到 v2 的 N 调用实现用于对照。

**验证方法**：`BENCH_RUNS=base python bench_streaming_local.py`，
比较新 CSV 中 base 的 `t_sample`（B ≈ A）与 `reward_head_start`（B 仍远小于 t_sample）。

---

*报告由 `bench_streaming_local.py` v4（引擎级流式）运行产物汇总生成。*

## 8. RM（奖励模型）场景实测

> 配置：生成式 judge（`TWINKLE_REWARD_MODEL_ID`，独立 GPU，max_tokens=8 判词输出）；
> batch=2×gen=2（每步 4 条）、max_tokens=512、3 步；提交流粒度三档。
> 方案：`docs/designs/rm_bench_plan.md`。

### 结果（稳定步均值；首步含 judge 引擎 warm-up 一次性开销 46s，已剔除）

| run | 路径 | 粒度 | t_sample | t_collect | t_total |
|---|---|---|---|---|---|
| rm-whole | A | 整批（1 handle） | 8.5s | 0.96s | 11.3s |
| rm-b-whole | B | 整批 | 9.2s | 0.97s | 11.6s |
| rm-b-mini | B | 每 2 条一批 | 9.2s | 1.30s | 12.0s |
| rm-b-per | B | 逐条 | 9.3s | 1.27s | 11.9s |

### 结论（RM 场景）

1. **提交粒度差异从"数量级"缩至"噪声级"**：真实 judge 每条延迟仅 ~0.3s（8 token 判词），
   2 线程并行已吸收大部分串行代价——collect 差异 ~30%、t_total 差异 ~6%
   （与 d1000 模拟的 8 倍放大形成对照：模拟 sleep 每条 1s，串行代价被放大）。
   whole 仍略优（judge 合批 + 无 handle 开销），但三者实际可视为等价。
2. **采样路径（A vs B）在 RM 场景同样持平**（11.3 vs 11.6s，t_sample 8.5 vs 9.2s）。
3. **⚠️ judge 判别质量限制**：`reward_mean=0.0000` 且无解析失败告警——judge 全部判
   Incorrect（4B 验证器判别失效）。**延迟特征（真实推理耗时）有效，判别语义无效**；
   性能对比结论不受影响；如需真实判别需换更强 judge 或判别式 RM 权重。
4. 对实践的意义：在"奖励昂贵但单条延迟不高（秒级内）"场景，提交粒度几乎不影响
   总耗时；只有单条奖励延迟很大（数秒+）且打分并行受限时，整批/小批提交才有
   可测量的优势（d1000 模拟的边界情形）。

### 8.1 边界验证：长尾生成 + 昂贵 judge（采样 2048 / judge 200 token 判词）

> 目的：验证"奖励成为关键路径"时，流式与提交粒度的表现（`bench_streaming_local.py`，
> 环境变量 `BENCH_RM_MAX_TOKENS=2048`、`TWINKLE_REWARD_JUDGE_MAX_TOKENS=200`）。

| run | t_sample | t_collect | t_total（含 step0 抖动） |
|---|---|---|---|
| rm-whole / A | 27.8s | 12.1s | 46.7s（剔除 step0 后 ~41.5s） |
| rm-b-whole / B | 25.1s | 11.96s | 39.0s（剔除 step0 后 ~40.8s） |
| rm-b-mini / B | 25.2s | 17.4s | 44.3s |
| rm-b-per / B | 24.9s | 15.1s | 41.7s |

**边界结论**：
1. **奖励成为关键路径的幅度不足**：judge 合批 collect 12s，仍低于采样 25-28s——流式（B）
   与批量（A）持平（t_total 差 ~1s，噪声级），**流式未因奖励变贵而反超**；
2. **提交粒度首次显著分化**：整批提交（collect 11.96s）比小批（17.4s，-31%）与逐条
   （15.1s，-21%）更优——长判词下 judge 推理合批收益真实显现
   （4 条共享 prefill/decode，与 d1000 sleep 模拟的线程等待机制不同）；
3. **RM 模式下 `semantic_ok` 列不可用**（`reward_deterministic` 检查将 judge 分数与规则
   奖励比对，语义不再成立），应忽略；`reward_head_start/tail` 同理为空（batch_judge
   manager 未打奖励时间线点，如需可补）。
## 附录 D：legacy（N 次远程调用）与 engine（引擎级流式）对照

| run | legacy A t_total | legacy B t_total | legacy B/A | engine A t_total | engine B t_total | engine B/A |
|---|---|---|---|---|---|---|
| base | 16.70 | 100.68 | 6.03× | 24.4 | 24.5 | 1.00× |
| gen2 | 13.95 | 49.94 | 3.58× | 21.2 | 21.2 | 1.00× |
| gen8 | 29.40 | 208.14 | 7.08× | 32.7 | 33.0 | 1.01× |
| tok512 | 16.19 | 92.88 | 5.74× | 17.4 | 18.2 | 1.05× |
| tok2048 | 18.83 | 105.70 | 5.61× | 44.6 | 44.1 | 0.99× |
| d200 | 19.81 | 106.74 | 5.39× | 25.5 | 26.9 | 1.05× |
| d1000 | 19.68 | 108.14 | 5.49× | 26.3 | 31.5 | 1.20× |

> legacy 数据为 v2 实测；engine 数据为 v4 实测。
> 核心对照：legacy B 因 actor 串行化放大 3.6~7.1×；engine B 与 A 持平
> （0.99~1.05×，d1000 除外）。A 侧绝对值的差异源于提示词变更
> （system 指令 + 参考消息移除）与运行环境波动，不影响 B/A 比值结论。

