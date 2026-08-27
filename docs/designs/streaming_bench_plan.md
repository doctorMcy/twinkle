# 流式采样收益测试方案（本地模式 GRPO + Reward Loop）

> 状态：已确认并进入实现。配套脚本：`cookbook/rl/reward_loop/bench_streaming_local.py`；
> 运行结果与分析见 `docs/designs/streaming_bench_report.md`。

## 1. 测试目标（按优先级）

1. **正确性（先验证）**——流式路径与批量路径结果一致，两级口径：
   - **Level 1 确定性**：greedy + 固定 seed 下，两条路径逐 token 输出完全相同，进而 reward、优势、loss 一致；
   - **Level 2 语义**：随机采样下无丢样本 / 无重复 / 无乱序，样本索引对齐，reward 函数与 `GRPOAdvantage`（纯函数）确定性一致。
2. **收益量化（再测）**——流式「逐序列完成即提交」带来的**奖励计算与采样的重叠**、端到端每步墙钟时间缩短、吞吐提升；并做单变量扫描，观察收益随参数的变化趋势。

## 2. 已确认约束

| 项 | 决定 |
|---|---|
| 对比方式 | 同一 harness 内批量 / 流式两条路径 A/B（同部署、同数据集、同参数、同权重状态） |
| 流式语义 | 同一批 N 个序列并发生成，任一完成立即提交该序列的奖励（本地版 true_stream） |
| 运行环境 | 仅本地 ray 模式（`twinkle.initialize(mode='ray')`，无需 twinkle-server） |
| GPU | ≥2 卡：model 1 卡 + sampler 1 卡（DeviceGroup 分卡） |
| 模型/数据 | 本地路径优先，环境变量可覆盖（回退 `ms://` 地址） |
| 时间预算 | 1–2 小时：基础配置 + 单变量扫描 |
| 交付物 | 测试脚本 + 时间线数据（JSONL/CSV）+ 本方案与测试报告（md） |

## 3. 实现方式

### 3.1 为什么流式路径用「并发远程调用」而不是 `sample_stream`

本地 ray 模式下，`vLLMSampler` 以 Ray actor 运行在独立进程中（driver 只持有
actor 句柄，`self.engine` 只在 actor 进程内存在）。未加 `@remote_function`
装饰的方法（`sample_stream`、`_sample_single`、`_iter_in_loop` 等）在 driver
侧调用会直接访问不存在的 engine，因此不可用。

流式路径因此实现为：**用线程池并发发起 N 个 `sampler.sample([单个输入])`
远程调用**，每个调用在对应序列生成完成时返回 —— 天然形成「逐序列完成事件」，
与 server 版 `stream_sample_to_data_plane` 的 `StreamSample` 事件语义等价。
vLLM 引擎本身支持并发请求动态合批，故生成效率与整批调用基本一致。

### 3.2 Harness 结构（`bench_streaming_local.py`）

```
main_benchmark()
├── 统一基建：TransformersModel(model 卡) + vLLMSampler(sampler 卡)
│   + AsyncRewardPipeline + GRPOAdvantage + GSM8K 数据集（两路径共用）
├── Level-1 正确性运行：temperature=0 + 固定 seed，A/B 各采样 1 步（不训练）
│   → 逐位对比 tokens / logprobs / decoded / reward
├── 扫描运行：每个 run 内先 Path A（整批采样 → 整批提交 → collect → 训练）
│   后 Path B（N 个并发调用，任一完成立即提交 → collect 全部 → 训练），各跑 N 步
└── 打点：每步记录 t_sample / t_submit / t_collect / t_train / t_total，
    奖励 worker 内按 item_id 记录 reward_start / reward_end → JSONL + CSV
```

关键点：

- **公平性**：同 run 内 A、B 使用同一份 prompt 序列、同一采样参数、同一权重
  状态（A 先跑，B 后跑，run 内不做其它扰动）；每步前 `sync_weights` 与
  `reset_prefix_cache` 与 `minimal_grpo_local` 一致。
- **奖励延迟注入**：`gsm8k_score` 内按环境变量 `TWINKLE_REWARD_DELAY_MS`
  sleep，模拟昂贵奖励（如 LLM-as-judge），用于放大/观察流式收益。
- **backlog**：流式路径每序列一个 handle，`AsyncRewardPipeline.backlog` 需 ≥
  最大每步序列数 + 余量（按 gen=8 时 batch×gen=32 计算，取 34）。
- **单缓冲调度**：两条路径都在「本步采样完成后 collect 本步奖励再训练」
  的调度下执行，避免双缓冲把奖励计算平移到下一步采样期而掩盖流式收益。

### 3.3 正确性验证设计

- **Level 1（确定性）**：`temperature=0`（greedy）+ 固定 seed，同一批输入
  分别走 A、B 路径采样（期间不训练），断言：
  - 两条路径每索引的 `tokens` 逐位相同、`logprobs` 逐位一致（允许浮点精度差）；
  - 相同 `solution_str` + `ground_truth` 经同一评分函数得到相同 reward；
  - `GRPOAdvantage` 输出一致（纯函数，构造相同输入直接断言）。
  若出现 token 级差异，记录最大偏差与差异计数，标记为「近似一致」而非硬失败。
- **Level 2（语义）**：随机采样（temperature=1.0, top_p=0.95）下对每条路径：
  - 样本总数 = batch × num_generations；`item_id` 无重复、无丢失；
  - 每个 handle 的 collect 结果与 item_id 对齐（pipeline 内部 `reorder_by_id`，
    外部再断言集合相等）；
  - reward 函数确定性：对若干样本用同一 solution 重新打分，两次结果一致；
  - 优势按索引组装后与纯函数结果一致。

### 3.4 扫描矩阵

基础配置：`batch=4, gen=4, max_tokens=1024, reward_delay=0`，跑 6 步；
每个扫描点 4 步、只变一个变量：

| 变量 | 取值 | 目的 |
|---|---|---|
| num_generations | 2 / 8 | 组内并发度对重叠量的影响 |
| max_tokens | 512 / 2048 | 生成越长，流式提前算奖的窗口越大 |
| reward_delay | 200ms / 1000ms（每序列） | 奖励越贵，流式收益越大（核心结论） |

预期结论形态：流式收益随 reward_delay 与 max_tokens 增大而放大，随
num_generations 增大而放大（组内尾部等待更长）。

## 4. 指标定义

| 指标 | 定义 |
|---|---|
| `t_sample` | 采样阶段墙钟（A：整批 sample 调用；B：从首个任务发起到最后一条序列完成） |
| `t_submit` | 提交阶段墙钟（A：整批 submit；B：逐序列 submit 的总耗时） |
| `t_collect` | collect 墙钟（等待本步全部奖励就绪的时间，即「奖励尾部等待」） |
| `t_train` | 训练（forward_backward + clip_grad_and_step）墙钟 |
| `t_total` | 单步总墙钟 = t_sample + t_submit + t_collect + t_train |
| `reward_head_start` | 本步第一条奖励开始计算相对采样开始的时间（B 中 >0 表示与采样重叠） |
| `reward_tail_after_sample` | 本步最后一条奖励就绪相对采样结束的时间（A 中 ≈ 全额奖励耗时） |
| 吞吐 | 每步序列数 / t_total（seq/s） |

数据落盘：`results/bench_timeline.jsonl`（逐事件）+ `results/bench_summary.csv`
（逐 run 逐步聚合）。

## 5. 风险与预案

| 风险 | 预案 |
|---|---|
| driver 侧无逐序列完成 API | 线程池 + N 个并发远程 `sample([单输入])` 调用（见 3.1） |
| vLLM greedy 确定性无法绝对保证 | 记录差异计数/最大偏差，报告中标注「近似一致」 |
| 并发远程调用与整批调用的调度差异 | 扫描中 gen 变量、max_tokens 变量天然覆盖该差异的影响 |
| 本地模型/数据集路径格式 | 环境变量 `TWINKLE_MODEL_ID`、`TWINKLE_DATASET_ID` 可覆盖，首步冒烟验证加载 |

## 6. 交付物

- `cookbook/rl/reward_loop/bench_streaming_local.py`（可重复运行，配置走环境变量）
- `results/bench_timeline.jsonl` + `results/bench_summary.csv`
- `docs/designs/streaming_bench_report.md`（阶段耗时对比表、收益%、重叠分析、两级正确性结论）
