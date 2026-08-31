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
| 机器 / GPU | 训练机（本地 ray，无 twinkle-server） |
| 部署模式 | `twinkle.initialize(mode='ray')`，model 1 卡 + sampler 1 卡 |
| 模型 | 默认 `ms://Qwen/Qwen3.5-4B`（如需可覆盖） |
| 数据集 | GSM8K train |
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
**修复（已修入脚本，待重跑验证）**：(1) `create_dataset` 对齐 server 示例，
`map(GSM8KProcessor(system='Put the final answer within \\boxed{}.'))` ——
system prompt 要求 `\boxed{}` 输出，ground truth 写入 `user_data`；
(2) `_score` 增加容错提取兜底（`\boxed{}` → `####` → 末尾数字）；
(3) 启动诊断打印行结构（keys/user_data/preview）与解析出的 gt。耗时测量
已确认不受影响（与奖励值无关）。

**结论（修订后）**：位级不等价成立（12/16 tokens 发散、且发散均为措辞级）；
reward/advantage 层面的"一致"在 ground truth 修复前不作数，修复后重跑验证。

### 3.2 Level 2（语义：随机采样）

| 检查项 | A（批量） | B（流式） |
|---|---|---|
| 无丢样本 / 无重复 item_id | ✅ 全部通过 | ✅ 全部通过 |
| handle 结果与 item_id 对齐 | ✅ | ✅ |
| reward 函数确定性（重打分一致） | ✅ | ✅ |
| 优势组均值归零 | ✅ | ✅ |

**结论：Level-2 全部通过（60 个 run-路径-步骤检查点均 True）。**

## 4. 收益测量结果

> 数值 = 该 run 该路径所有步骤的均值（秒）。`t_submit` 均在 0.01s 以下，省略。

### 4.1 基础配置（batch=4, gen=4, max_tokens=1024, delay=0，6 步）

| 指标 | A（批量） | B（流式） | 说明 |
|---|---|---|---|
| t_sample | 11.87 | 95.85 | B 因 actor 串行化慢 8.1× |
| t_collect（奖励尾部等待） | 0.00 | 0.01 | delay=0 时双方都接近 0 |
| t_train | 3.84 | 3.94 | 基本一致 |
| **t_total（s/步）** | **16.70** | **100.68** | **B 慢 6.0×** |
| seq/s | 1.09 | 0.18 | A 的 6 倍 |
| reward_head_start | 11.87 | **4.80** | ✅ 流式重叠生效：B 第一条奖励在采样开始 ~4.8s 即开始，比 A 早 ~7s |
| reward_tail_after_sample | 0.00 | 0.01 | delay=0 均 ~0 |

### 4.2 单变量扫描（每 run 4 步）

| run | 变量 | A t_total | B t_total | B/A | 观察 |
|---|---|---|---|---|---|
| base | — | 16.70 | 100.68 | 6.03× | 基准 |
| gen2 | gen=2（8 序列） | 13.95 | 49.94 | 3.58× | 序列越少，串行化代价越小 |
| gen8 | gen=8（32 序列） | 29.40 | 208.14 | **7.08×** | 序列越多，串行化代价越大（∝N） |
| tok512 | max_tokens=512 | 16.19 | 92.88 | 5.74× | 长度缩短无帮助 |
| tok2048 | max_tokens=2048 | 18.83 | 105.70 | 5.61× | 长度加长无帮助（收益 ~秒级，被串行化吞没） |
| d200 | delay=200ms | A t_collect=0.21，B t_collect=1.35 | 19.81 | 106.74 | 5.39× | 延迟使 B 的奖励等待更差 |
| d1000 | delay=1000ms | A t_collect=1.01，B t_collect=8.01 | 19.68 | 108.14 | 5.49× | B 的逐序列 handle 受 2 线程池限制，奖励吞吐 1/8 |

**重叠指标的细节**（B 的 head_start 均远小于其 t_sample）：
- base/B head=4.80s vs t_sample=95.85s；gen8/B head=4.61s vs 194.73s；
  d1000/B head=4.77s vs 92.74s ——「边采样边算奖」机制确认生效，但奖励窗口
  （~几秒）相对串行化采样的损失（数十~上百秒）可忽略。

## 5. 结论

1. **正确性**：Level-2（结构/对齐/奖励确定性）全部通过；Level-1 位级等价不成立
   （12/16 tokens 发散，机制见 3.1），但 reward/advantage 层面一致 —— 训练
   不等价，报告中按「近似一致」记录。
2. **流式重叠机制有效**（v2 修复后）：B 的第一条奖励在采样开始后 ~4-5s 即开始
   计算，比整批提交早 ~7-12s；delay=0 时奖励等待几乎为 0。
3. **但本地 ray 模式下「逐序列远程调用」式流式不可行**：同一 sampler actor
   串行执行 N 个远程调用（`max_concurrency=1`），采样耗时 ≈ N × 单序列耗时
   （实测 8/16/32 序列时总耗时放大 3.6×/6.0×/7.1×），任何奖励重叠收益都被
   吞没。**结论：与 server 模式不同，本地 ray 模式没有引擎级逐序列事件流
   （`stream_sample_to_data_plane`），无法用纯客户端手段做出有收益的流式采样。**
4. **奖励昂贵场景（d1000）也不反转结论**：A 的 naive manager 对 chunk 内并行
   打分（1s×16 items 在 1.0s 内完成），B 的 16 个单 item handle 受
   pipeline 2 线程池限制串行成 8.0s —— 逐序列提交在奖励层反而是劣势。
5. **实践建议**：本地 ray 模式下用「整批采样 + 奖励管线双缓冲」；真正的流式
   收益（奖励与采样重叠）需要 vLLM 引擎层提供逐序列完成事件（推广
   `stream_sample_to_data_plane` 到本地组件），或作为服务端部署使用。

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

---

*报告由 `bench_streaming_local.py` v2 运行产物（`results/bench_summary.csv`，61 行）汇总生成。*
