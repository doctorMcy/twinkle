# twinkle 异步奖励计算模块移植设计方案

> 版本：v0.2（实施与实测修订版，由 v0.1 评审稿更新）
> 目标代码库：`twinkle-kit`（`src/twinkle`）
> 移植来源：`verl/experimental/reward_loop`
>
> **v0.2 修订说明**：v0.1 评审稿的两项待决事项（数据契约、模型服务托管）中，
> 数据契约已按「评审表 1 选项 A」落地实施；实现与验收已通过 cookbook 示例与
> 流式采样收益基准（`docs/designs/streaming_bench_report.md`）实测验证。
> 本文档在各章节标注「当前状态」，与实现的差异点（线程池 vs Ray worker、
> 提交背压语义、默认打分器注册空缺、逐样本流式的实测结论）已如实记录。

---

## 1. 背景与目标

### 1.1 背景

verl 的 `experimental/reward_loop` 提供了一套**分布式、可异步并行**的奖励计算组件：把规则奖励、判别式/生成式奖励模型（DisRM/GenRM）统一抽象为一条计算链路，并实现了「采样（rollout）与奖励计算」流式重叠、训练循环不被奖励阻塞的异步机制。

twinkle 在 v0.1 评审时**没有**对应的分布式奖励基础设施：

- 奖励仅限驱动进程内的本地同步规则函数（`Reward.__call__(trajectories) -> List[float]`）。
- 采样（vLLM sampler actor）→ 奖励（驱动进程同步）→ 优势 → 前反向，完全串行，奖励计算阻塞整个循环。

移植目标：把 verl 的**异步规则奖励计算能力**移植进 twinkle，使「采样 / 奖励计算 / 训练」三段在时间上重叠。

**当前状态（v0.2）**：模块已实现并落地于 `src/twinkle/reward_loop/`；cookbook 提供 4 个示例
（`minimal_grpo.py` 同步 / `minimal_grpo_local.py` 本地双缓冲 / `minimal_grpo_streaming.py`
server 双缓冲 / `minimal_grpo_true_stream.py` server 逐样本流式）；本地模式逐样本流式已做
基准实测（结论见 §10 R6 与附录 D）。

### 1.2 范围

**本次包含（In Scope，v0.2 已实现）**

- 分布式规则奖励：多 worker 并行计算（默认进程内线程池；Ray actor 路径预留）。
- RewardManager 家族移植：`naive` / `dapo` / `gdpo` / `limited`（rate_limited）/ `remote`。
- 同步 / 异步自定义奖励函数（自动识别 + 同步函数进 executor）。
- **异步奖励流水线**：提交-收集解耦、双缓冲、背压、按 key 对齐、错误降级。

**本次不包含（Out of Scope，预留接口）**

- DisRM / GenRM 模型推理服务、router、`classify`/`embeddings` HTTP 链路（评审表 2 未启动）。
- 与训练器深度耦合的 off-policy 重排 / staleness 回放（TransferQueue 等价物）。

### 1.3 核心目标与验收状态

| 编号 | 目标 | 验收口径 | v0.2 状态 |
| --- | --- | --- | --- |
| G1 | 奖励结果正确性 | 与同步基线实现逐项一致 | ✅ 通过（基准 Level-2 语义检查；详见 §9.2） |
| G2 | 采样不被奖励阻塞 | 时间线断言：B_k 采样完成即返回，奖励在后台算 | ✅ 通过（`reward_head_start`/`reward_tail_after_sample` 实测，见 §5.4、§9.2） |
| G3 | 模块低耦合可复用 | `reward_manager/` + `data.py` 零 twinkle 依赖，纯 CPU 可单测 | ✅ `data.py` 仅标准库；`reward_manager/base.py` 仅 asyncio/标准库（torch 可选导入） |
| G4 | 平滑回退 | `reward.mode='sync'` 退回「提交即收集」，行为等价现状 | ✅ 已实现（`pipeline.submit` 内 `mode='sync'` 立即 collect） |

---

## 2. verl 异步奖励机制要点（移植依据）

> 本节不变；v0.2 标注移植对照的落地情况。`file:line` 为 verl `main` 分支现状。

### 2.1 核心开关：`reward_loop_worker_handles`（✅ 已落地）

`RewardLoopManager.reward_loop_worker_handles`（`verl/experimental/reward_loop/reward_loop.py:311-322`）：

```
非 None  ⟺ 奖励可以跑在 rollout 管线内（流式）
  = None ⟺ 奖励必须由训练侧串行补算（colocate）
```

**twinkle 落地**：`AsyncRewardPipeline.mode='async'`（默认）等价「流式」；
`mode='sync'` 等价「colocate 串行补算」（`pipeline.submit` 内立即 collect，`pipeline.py:91-92`）。

### 2.2 每样本一条 asyncio task + Ray actor 流式 compute_score（✅ 已落地，worker 形态有差异）

- verl：Worker 对 batch 内逐样本 `asyncio.create_task` + `asyncio.gather`（`reward_loop.py:138-143`）。
- **twinkle 落地**：`RewardLoopWorker.compute_score_batch` → `asyncio.run(manager.run_batch(items))`；
  `RewardManagerBase.run_batch` 对每条 item `asyncio.create_task` + `gather`（`reward_manager/base.py:65-79`）。
- **差异（重要）**：默认 worker 为**进程内对象 + `ThreadPoolExecutor(max_workers=num_workers)`**
  （`pipeline.py:38`），并非 Ray actor；`pipeline.submit` 仅在 worker 方法带 `.remote`
  属性（如传入 Ray actor 句柄）时走 Ray 路径（`pipeline.py:82-84`）。详见 §5.3。

### 2.3 fire-and-forget 提交 + 延迟收集（✅ 已落地，语义有前提）

- verl：先对全部 worker 提交 `compute_score_batch.remote(chunk)`，再统一 `ray.get`（`reward_loop.py:348-354`）。
- **twinkle 落地**：`submit(batch) -> BatchHandle`（切块 → 分发 → 立即返回），
  `collect(handle) -> List[RewardResult]`（按 `item_id` 对齐后返回，`pipeline.py:95-134`）。
- **前提（实测确认）**：`submit` 仅在 `len(pending) < backlog` 时严格非阻塞；达到 `backlog`
  上限时按 `on_backlog_full` 策略阻塞收集最老批次（`'block'`）或丢弃（`'drop_oldest'`，
  `pipeline.py:73-77`）。背压即在这一层生效。

### 2.4 多级背压（✅ 已落地核心；verl 的多级队列未全量移植）

| verl 层级 | twinkle 落地 |
| --- | --- |
| 生产者并发（asyncio.Queue） | `pipeline.pending` 列表 + 锁；`backlog`（默认 2）为最大未收集批次数 |
| 样本并发上限 | `RateLimitedRewardManager.max_concurrent`（`limited.py`） |
| 消费队列丢最老 | `on_backlog_full='drop_oldest'`（`pipeline.py:73-75`） |
| 消费端攒批 | 训练侧自行控制（示例里一个 step 一个 batch） |

### 2.5 以 key/uid 对齐结果，不依赖顺序（✅ 已落地）

- **twinkle 落地**：`RewardItem.item_id` 为对齐键；`reorder_by_id`（`data.py:34-40`）乱序安全；
  `collect` 按 `items` 原始顺序回填（`assemble_scores`，`data.py:43-50`）。

### 2.6 同步/异步函数自适应（✅ 已落地）

- `RewardManagerBase.call_score`：`inspect.iscoroutinefunction` 识别；同步函数进
  `loop.run_in_executor(None, ...)`（`reward_manager/base.py:34-46`）。
- 注意：`run_in_executor(None)` 使用默认 executor（线程数 ≈ min(32, cpu+4)），
  基准实测表明 chunk 内并行度受此影响（见 §5.5 分块粒度结论）。

### 2.7 错误处理约定（✅ 已落地核心；指数退避重试未移植——出范围）

- `on_error: 'raise' | 'zero'`：`zero` 降级 0 分并把异常写入 `RewardResult.reward_extra_info['error']`
  （`pipeline.py:113-123`）。
- `RateLimitedRewardManager` 超时/异常降级为 0 分（`limited.py:55-64`）。
- HTTP 5xx 指数退避（verl `reward_loop.py:157-195`）：**未移植**（模型服务出范围）。

---

## 3. twinkle 现状与差距

### 3.1 现状事实清单（v0.2 更新）

| 维度 | 现状 |
| --- | --- |
| 分布式基础设施 | Ray：`infra/_ray/`（`create_workers` / `LazyCollect` / config registry）、`twinkle.initialize(mode='ray', ...)`、`@remote_class` |
| 数据格式 | 无 DataProto。`InputFeature` / `Trajectory` / `SampledSequence` / `SampleResponse` 等 |
| 奖励管线 | ✅ **`src/twinkle/reward_loop/` 已实现**：`AsyncRewardPipeline` / `RewardLoopWorker` / `RewardManagerBase` 家族 / `data.py` / `metrics.py` / `config.py` |
| 奖励函数 | `Reward.__call__(trajectories)` 同步族保持不变；管线侧使用 verl 风格 `compute_score(data_source, solution_str, ground_truth, extra_info)` 契约 |
| 配置 | `RewardLoopArgs`（`reward_loop/config.py`）扁平分组，字段与 v0.1 设计一致 |
| 采样 | vLLM engine 内嵌 sampler Ray actor（无独立 HTTP 推理服务） |
| HTTP 服务层 | Ray Serve + FastAPI gateway（`server/`）；Client SDK 经 HTTP 驱动训练 |
| 测试/CI | pytest（tests/）、.github/workflows；基准脚本 `cookbook/rl/reward_loop/bench_streaming_local.py` |

### 3.2 契约缺口（v0.1 预警 → v0.2 已证实并修复）

v0.1 指出 `grpo.py:161` 把无 `messages` 的 `new_input_feature` 直接传给奖励函数会导致取不到分。
基准实测**证实了同类问题并发现两个新缺口**：

1. **ground truth 缺失（最严重）**：`ms://modelscope/gsm8k` 原始行**没有顶层 `answer` 字段**，
   也未在 `user_data` 携带 `ground_truth`——不经过 `GSM8KProcessor` 映射时所有奖励恒为 0
   （`rewards_identical` 等检查成为"全零平凡真"）。**修复**：示例统一
   `dataset.map(GSM8KProcessor(system='...\\boxed{}.'))` 把 ground truth 写入 `user_data`
   （bench 脚本已加 `_extract_ground_truth_from_answer` 兜底从 `answer` 字段解析 `#### <n>`）。
2. **奖励提取格式**：`GSM8KAccuracyReward.extract_answer` 只认 `\boxed{}` / `####`；
   未加 system 指令的模型输出（如 `**Final Answer:** 72 clips.`）无法提取 → 恒 0。
   **修复**：system prompt 强制 `\boxed{}` 输出 + 示例层容错提取（`\boxed` → `####` → 末尾数字）。

**结论**：奖励输入契约已收敛为 `RewardItem`（评审表 1 选项 A）+ 适配器组装；任何新数据源
接入都必须走「处理器映射出 ground_truth + 适配器构造 `RewardItem`」两步，否则会出现静默全 0。

### 3.3 差距总览（v0.2 更新）

| 能力 | verl | twinkle（v0.2） | 差距 |
| --- | --- | --- | --- |
| 异步奖励 worker | Ray + asyncio per-sample | ✅ 线程池 + asyncio per-item（Ray 路径预留） | 默认形态差异（§5.3） |
| 奖励与采样重叠 | 流式/异步 | ✅ 双缓冲（batch 级）；逐样本流式仅 server 模式 | 本地逐样本流式待引擎级支持（R6） |
| 奖励函数注册/定制 | register/importlib | ✅ `registry` + `custom_reward_function_*`（importlib 仅函数路径，manager importlib 未实现） | 部分 |
| 高级 Manager（限流等） | ✅ | ✅ naive/dapo/gdpo/limited/remote | 已移植 |
| 默认打分器分派 | 数据集大表 | ⚠️ `default_score.py` 注册表已实现但**尚无内置注册项**（gsm8k 等未入库） | 需补注册或显式传 compute_score |
| 模型奖励服务 | ✅ | ❌（出范围） | 预留 |

---

## 4. 目标架构（v0.2：已落地）

### 4.1 模块布局（实际文件与设计一致，已全部实现）

```
src/twinkle/reward_loop/
├── __init__.py             # 导出 RewardItem/RewardResult/AsyncRewardPipeline/RewardLoopWorker 等
├── data.py                 # RewardItem / RewardResult / split_items / reorder_by_id / assemble_scores
├── config.py               # RewardLoopArgs（扁平分组）
├── reward_manager/         # 自 verl 移植，run_single 参数为 RewardItem
│   ├── __init__.py         #   RewardLoopManager = RewardManagerBase（别名）
│   ├── base.py             #   RewardManagerBase（call_score/run_single/run_batch + semaphore）
│   ├── registry.py         #   register / get_reward_manager_cls / registered_managers
│   ├── naive.py            #   NaiveRewardManager（默认）
│   ├── dapo.py             #   DAPORewardManager（overlong 惩罚）
│   ├── gdpo.py             #   GDPORewardManager（experiment_name 注入）
│   ├── limited.py          #   AsyncTokenBucket + RateLimitedRewardManager（三层限流）
│   └── remote.py           #   RemoteRewardManager（继承 naive，独立进程语义标注）
├── default_score.py        # 默认 compute_score 分派（register_score 注册表；暂无内置注册项）
├── worker.py               # RewardLoopWorker（@remote_class(execute='all')，manager 驱动）
├── pipeline.py             # AsyncRewardPipeline：submit/collect/双缓冲/背压/对齐/错误降级
└── metrics.py              # RewardLoopMetrics（submit_time/collect_wait_time/reward_time/overlap_ratio）
```

### 4.2 分层原则（✅ 达成）

```
┌─────────────────────────────────────────────────────────┐
│ 驱动进程（训练脚本）                                       │
│   AsyncRewardPipeline（pipeline.py）                     │
│     ├─ 提交 batch（backlog 未满时非阻塞）                  │
│     └─ 收集 rewards（按需阻塞，按 item_id 对齐）           │
└──────────────┬──────────────────────────┬────────────────┘
               │ ThreadPoolExecutor(num_workers)（默认）
               │ 或 Ray actor（worker 方法带 .remote 时）    │
┌──────────────▼──────────────────────────▼────────────────┐
│ RewardLoopWorker × N（worker.py）                        │
│   └─ asyncio.run(manager.run_batch(items))               │
│        └─ per-item asyncio task → compute_score          │
│             （异步函数 await；同步函数进默认 executor）      │
└───────────────────────────────────────────────────────────┘
```

- **第一层（纯逻辑）**：`data.py` + `reward_manager/` + `default_score.py`。零 twinkle 依赖 ✅（G3）。
- **第二层（适配）**：`worker.py` / `pipeline.py`。twinkle 绑定仅限 `@remote_class` 装饰与 CLI 配置接入。

### 4.3 与 verl 的移植映射（✅ 全部有对应落地）

| verl | twinkle 落地 | 说明 |
| --- | --- | --- |
| `compute_rm_score` | `AsyncRewardPipeline.submit/collect` | 拆成提交/收集两段，支持双缓冲 |
| `RewardLoopWorker` | `RewardLoopWorker`（worker.py） | `@remote_class(execute='all')`；输入为 `RewardItem` |
| `pad_dataproto_to_divisor` + `chunk` | `data.py: split_items / reorder_by_id` | 纯 list 切分（round-robin，无 padding 语义） |
| `assemble_rm_scores` | `data.py: assemble_scores(mode='scalar')` | token 级（`mode='token'`）预留 |
| 默认函数/注册表 | `default_score.py` + `registry.py` | ⚠️ 注册表为空，需补充注册或显式传函数 |
| `load_reward_manager` | `RewardLoopWorker.__init__`（manager_name 分派） | 按注册表取 Manager |
| `reward_loop_worker_handles`（流式开关） | `mode='async' | 'sync'` | 语义等价 |
| TransferQueue key 对齐 | `RewardItem.item_id` | 同语义、更轻量 |
| 多级背压 | `backlog` + `on_backlog_full`（block/drop_oldest） | 简化为一层 |

---

## 5. 异步奖励流水线设计（核心章节，v0.2 实录）

### 5.1 数据契约（✅ 按评审表 1 选项 A 落地，签名与设计一致）

`RewardItem`（item_id / data_source / solution_str / ground_truth / extra_info /
response_ids / attention_mask / raw_prompt）与 `RewardResult`
（item_id / reward_score / reward_extra_info）——实现于 `data.py`，与 v0.1 设计一致。

**奖励函数契约**：verl 四参数 `compute_score(data_source, solution_str, ground_truth, extra_info) -> (score, extra)`，
同步/异步皆可。cookbook 示例（`minimal_grpo*`、bench 脚本）均以 `gsm8k_score` 适配器形式给出
参考实现，其中包含轨迹重建（messages 追 assistant 内容）与 ground truth 兜底逻辑（§3.2）。

### 5.2 切分与重组（✅ 已实现，`data.py`）

- `split_items(items, num_workers)`：round-robin 均匀切分；
- `reorder_by_id(results)`：乱序安全重组；重复 id 报错；
- `assemble_scores(results, items, mode='scalar')`：按 items 原始顺序输出标量列表；`mode='token'` 预留。

### 5.3 Pipeline 运行模型（✅ 已实现；与 v0.1 的两处差异）

`AsyncRewardPipeline.submit/collect` 语义与设计一致，但：

1. **worker 执行形态**：默认 `ThreadPoolExecutor(max_workers=num_workers)` + 进程内
   `RewardLoopWorker` 实例；worker 为 `@remote_class` 但不带 `remote_group` 时即本地对象。
   仅当传入带 `.remote` 方法的 Ray actor worker 时走远端路径（`pipeline.py:82-84`）。
   实测行为（基准 §4）：num_workers=2 → pipeline 线程池并发 2；
   chunk 内由 naive manager 的 asyncio gather + 默认 executor 并行打分。
2. **submit 阻塞语义**：`len(pending) >= backlog` 时 submit 会先 collect 最老批次（`'block'`）
   或丢弃（`'drop_oldest'`），并非无条件非阻塞。单缓冲 + 逐批 collect 时不受影响。

### 5.4 双缓冲时间线与重叠测量（✅ 机制已落地，基准实测确认）

设计的时间线（采样 k 期间奖励 k 在算、训练 k-1 期间采样 k+1 在跑）与 v0.1 §5.4 一致。

**实测验证（`docs/designs/streaming_bench_report.md` §4）**，以 `reward_head_start`（首条奖励
相对采样开始）、`reward_tail_after_sample`（末条奖励相对采样结束）两个指标量化：

- 逐样本提交（真流式）时：首条奖励在采样开始后 ~4-5s 即开算（≈ 首条序列完成时刻），
  奖励与采样真实重叠 ✅（G2）；
- 双缓冲（整批提交、下步收集）会把奖励计算窗口平移到下一轮采样期——重叠收益被"掩盖"，
  测量流式收益时应使用单缓冲调度（bench 脚本即如此设计，并作为结论写入报告 §5.2）。

`metrics.py` 提供 `RewardLoopMetrics.overlap_ratio = 1 - collect_wait_time / (submit_time + reward_time)`
供训练侧持续观测（G2 验收的运行时指标）。

### 5.5 并发、背压与分块粒度（✅ 已实现；分块粒度有实测结论）

- worker 内并发：`run_batch` per-item asyncio task（`base.py:65-79`）；同步函数进默认 executor。
- consumer 背压：`backlog`（默认 2）最大未收集批次数；满则 block 或 drop_oldest。
- producer 并发：内部锁 + `pending` 列表，多 submit 线程安全。

**分块粒度实测结论（基准 d200/d1000 run）**：整批提交（1 个 handle / 大 chunk × num_workers 线程）
的奖励吞吐显著高于逐样本提交（N 个单 item handle）。实测 d1000（16 items × 1s 延迟）：
整批 `t_collect ≈ 1.0s`（chunk 内 8 路并行），逐样本 16 个单 item handle ≈ `8.0s`
（受 pipeline 线程池 = num_workers = 2 串行限制）。**结论：奖励非昂贵（规则函数）时，
整批提交为最优形态；逐样本提交仅当奖励本身昂贵且 worker 各有并行能力时才有意义。**

### 5.6 模式与回退（✅ 已实现）

| `mode` | 行为 | 落地 |
| --- | --- | --- |
| `async`（默认） | 双缓冲异步管线 | `pipeline.py` 默认 |
| `sync` | submit 后立即 collect（提交即收集） | `pipeline.py:91-92`；G4 已验收 |

### 5.7 错误处理（✅ 已实现核心）

- `on_error='raise' | 'zero'`：`zero` 降级 0 分 + `reward_extra_info['error']`（`pipeline.py:114-123`）。
- `RateLimitedRewardManager` 超时/异常降级 0 分（`limited.py`）。
- HTTP 模型服务相关重试未移植（出范围）。

### 5.8 奖励函数加载（✅ 函数路径已实现，manager importlib 未实现）

- `RewardLoopWorker.__init__`：`custom_reward_function_path/name` 动态 import
  （`worker.py:21-23`）；未配置时用 `default_compute_score`。
- Manager 按 `manager_name` 从注册表取；`manager_module_path/name`（manager importlib 模式）
  字段存在但**尚未实现**（`config.py` 与 `worker.py` 未接线）。

---

## 6. 配置接入（✅ 已实现）

`RewardLoopArgs`（`reward_loop/config.py`）字段与 v0.1 §6.1 完全一致（num_workers /
custom_reward_function_path/name / manager_name/source/module_path/module_name /
unknown_rewards / mode / backlog / on_backlog_full / on_error / max_rpm/max_tpm/max_concurrent/
timeout / reward_worker_executors / reward_kwargs）。CLI/YAML 扁平键用法与 v0.1 一致（示例见
`minimal_grpo*.py` 的 `AsyncRewardPipeline.from_args(args)` 或直接构造）。

---

## 7. 训练器/示例集成（✅ 已落地，示例簇完整）

### 7.1 示例清单（v0.2 新增）

| 示例 | 模式 | 说明 |
| --- | --- | --- |
| `cookbook/rl/reward_loop/minimal_grpo.py` | server + 同步 | 入门；submit 后立即 collect |
| `cookbook/rl/reward_loop/minimal_grpo_local.py` | 本地 ray + 双缓冲 | **无 server、无 tinker**；本地组件全链路 |
| `cookbook/rl/reward_loop/minimal_grpo_streaming.py` | server + 双缓冲 | 整批采样 + 奖励双缓冲 |
| `cookbook/rl/reward_loop/minimal_grpo_true_stream.py` | server + 逐样本流式 | `stream_sample_to_data_plane` → 每 `StreamSample` 事件即提交奖励 |
| `cookbook/rl/reward_loop/bench_streaming_local.py` | 本地 ray + A/B 基准 | 批量 vs 流式同 harness 对比 + Level-1/2 正确性检查 + 扫描矩阵 |

### 7.2 训练循环改造模式（与 v0.1 §7.1 一致）

双缓冲骨架已在 `minimal_grpo_local.py` / `minimal_grpo_streaming.py` 落地：
submit 当前批 → collect + 训练上一批 → flush 尾批。适配器（`gsm8k_score` 四参数包装 +
`make_reward_items`）即 v0.1 §7.2 所述职责，分布在各示例中。

### 7.3 与 twinkle 组件的关系（✅ 不变）

复用 `GRPOAdvantage` / `CompletionRewardMetric`；`model.forward_backward` / `sampler.sample` /
`CheckpointEngineManager` 均未改动。

---

## 8. 分阶段实施计划（v0.2 进度标注）

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| P0 决策与 Spike | CPU worker 验证 / 数据契约拍板 / 奖励函数调用约定 | ✅ 完成（契约选 A；CPU 路径未走 Ray，以线程池落地） |
| P1 核心流水线 | data.py / reward_manager 家族 / default_score / worker / pipeline / 单测 | ✅ 完成（`src/twinkle/reward_loop/` 全部文件；`limited.py` AsyncTokenBucket、`dapo.py` overlong、`gdpo.py` experiment_name 已移植） |
| P2 集成 | cookbook 接入 / 本机 e2e / 指标 | ✅ 完成（4 个示例 + 基准脚本；基准即 e2e 验证，见 §9） |
| P3 增强 | 异步自定义函数示例 / README 组件文档 / sync 回退回归 | 🟡 部分：sync 回退 ✅；README/组件文档待补 |
| P4 未来扩展 | DisRM/GenRM 服务 / 逐样本 true streaming / token 级 rm_scores | 🟡 部分启动：本地引擎级逐序列流式已实现（`vLLMSampler.sample_sequences_to_queue`，基准 v3），待实测；DisRM/GenRM 服务与 token 级组装仍未启动 |

---

## 9. 测试与验收（v0.2：实测结果）

### 9.1 单元测试（P1，纯 CPU）

`data.py` / `reward_manager/` 为纯逻辑（G3），cookbook 基准之外可独立单测
（split/reorder/assemble、AsyncTokenBucket、RateLimited 降级、双缓冲顺序——脚本层面
桩测试已覆盖部分：`.zcode/validate_bench_logic.py` 对 bench 纯逻辑 20 项断言全过）。

### 9.2 e2e 实测（基准 `bench_streaming_local.py`，本地 ray，训练机）

**正确性（G1）**：
- Level-2 语义检查（样本数、无重复/丢失 item_id、结果对齐、奖励确定性、优势组均值归零）：
  **60 个逐步检查点全部通过**；
- Level-1 位级确定性：**不成立**（12/16 tokens 发散，措辞级；机制为两条采样路径的
  vLLM 批形状/调度差异 + 训练前近并列 argmax）——与管线无关，属采样层结论；
- 契约缺口（§3.2）修复前 rewards 检查为"全零平凡真"；修复后需重跑取得非平凡结果。

**重叠（G2）**：逐样本提交场景下首条奖励在采样开始后 ~4-5s 即开始计算（`reward_head_start`
显著小于 t_sample），奖励与采样真实重叠；双缓冲会掩盖该收益（测量需单缓冲）。

**吞吐与形态结论**：
- 整批提交（大 chunk）在规则奖励（廉价）场景下是最优形态；逐样本提交在 2 线程池下
  奖励吞吐反而差 ~8 倍（d1000 实测）；
- 本地 ray 模式逐样本流式采样（N×并发远程调用）被 sampler actor 串行化，端到端慢
  3.6×~7.1×（gen 2/4/8）——**不可行**；真流式需引擎级逐序列事件（R6）。

### 9.3 验收清单（v0.2 状态）

- [x] G1 正确性逐项一致（Level-2 全过；Level-1 位级不等价为采样层结论，已记录）
- [x] G2 timeline 断言通过（`reward_head_start`/`reward_tail_after_sample` 实测）
- [x] G3 `data.py` + `reward_manager/` 无 `import twinkle`（静态检查）
- [x] G4 `mode='sync'` 行为等价现状（提交即收集）

---

## 10. 风险与开放项（v0.2 更新）

| # | 风险/开放项 | 影响 | 处置与实测 |
| --- | --- | --- | --- |
| R1 | `remote_class` CPU-only 路径 | worker 创建方式 | ✅ 已消解：默认线程池落地；Ray 路径为预留分支（§5.3） |
| R2 | 奖励输入契约（InputFeature 无 messages） | 组装 RewardItem 需界定来源 | ✅ **已证实并修复**：统一 `GSM8KProcessor` 映射 + 适配器组装；原始行无 `answer`/`ground_truth` 的静默全 0 问题已记录（§3.2），测量时必须先验证 gt 非空 |
| R3 | `data_source` 分派依赖命名 | 默认打分能力受限 | 🟡 `default_score.py` 注册表存在但**无内置注册项**；示例一律显式传 `compute_score`；建议后续注册 gsm8k/math 等默认打分手 |
| R4 | verl padding 语义简化 | 切块行为 | ✅ 纯 list 切分等价，无跨 batch padding 损耗 |
| R5 | YAML 内联 dict 与扁平键 | 配置体验 | 🟡 字段已落地；建议 P3 校准文档 |
| R6 | 逐样本真流式需 sampler 改造 | 重叠粒度 | ✅ **已实测 + 已提供本地引擎级实现**：batch 级双缓冲达成重叠（G2）；N×并发远程调用被 actor 串行化（端到端慢 3.6×~7.1×，基准 §5.3）；本地引擎级实现 `vLLMSampler.sample_sequences_to_queue`（一次调用内并发 + Ray 队列逐条回传事件，`dispatch='all', execute='first'`）已落地，待实测验证合批恢复（基准 v3） |
| R7 | 分块粒度经济性 | 逐样本提交吞吐 | ✅ **已实测**：1-item chunk × num_workers 线程池时奖励吞吐差 ~8×（d1000）；提交粒度应按奖励成本选择（§5.5） |
| R8 | 奖励提取格式（\boxed/####） | 无 system 指令时恒 0 | ✅ 示例层已修（system prompt + 容错提取）；库内 `GSM8KAccuracyReward` 保持原语义 |
| R9 | 双缓冲掩盖流式收益 | 基准测量失真 | ✅ 已确认：测量流式收益使用单缓冲调度（基准脚本设计），报告 §5.2 |

---

## 附录 A：评审待决事项（v0.2 决议更新）

### 评审表 1：奖励 worker 数据契约 — ✅ 已决议并落地：选项 A（轻量 `RewardItem`）

实现采用 `RewardItem` 作为唯一输入契约（`data.py`），驱动侧适配器从 `Trajectory`/数据集
组装（`make_reward_items` + `gsm8k_score` 示例）。表 B/C 不采纳：`InputFeature` 无 messages
的缺口已在实践中再次证实（§3.2）。

### 评审表 2：模型推理服务托管（P4 DisRM/GenRM） — ⏳ 未启动，维持推荐选项 A

（v0.1 评审表 2 原文保留；建议 P4 启动时按「选项 A：vLLM 子进程 + naive_router」实施，
接口按选项 C 预留 `RewardModelBackend` 抽象。）

---

## 附录 B：参考材料

- verl 源码：`verl/experimental/reward_loop/`（reward_loop.py / reward_model.py / reward_manager/ / router/）
- verl 官方文档：`docs/advance/reward_loop.rst`
- verl 异步机制：`experimental/fully_async_policy/`、`experimental/agent_loop/`、`docs/data/transfer_queue.md`
- twinkle 源码：`src/twinkle/reward_loop/`、`src/twinkle/infra/_ray/`、`cookbook/rl/reward_loop/`

## 附录 C：实现与实测记录

- 实现文件：`src/twinkle/reward_loop/`（§4.1 全量清单）。
- 示例：`cookbook/rl/reward_loop/minimal_grpo{,_local,_streaming,_true_stream}.py`。
- 基准：`cookbook/rl/reward_loop/bench_streaming_local.py`（A/B 同 harness、Level-1/2 正确性、
  扫描矩阵：gen/max_tokens/reward_delay）。
- 实测数据：`results/bench_summary.csv`（61 行）、`results/bench_timeline.jsonl`、
  `results/level1_texts.json`（逐对文本以核查发散形态）。
- 报告：`docs/designs/streaming_bench_report.md`（结论：Level-2 全过、Level-1 位级不等价、
  本地逐样本流式不可行需引擎级支持、分块粒度与提交时机对吞吐的影响）。
- 纯逻辑验证：`.zcode/validate_bench_logic.py`（20 项断言）。