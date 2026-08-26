# twinkle 异步奖励计算模块移植设计方案

> 版本：v0.1（评审稿）
> 目标代码库：`twinkle-kit`（`src/twinkle`）
> 移植来源：`verl/experimental/reward_loop`
> 本篇为**评审稿**：结尾「评审待决事项」两项选择（数据契约、模型服务托管）待评审确认后再进入实施。

---

## 1. 背景与目标

### 1.1 背景

verl 的 `experimental/reward_loop` 提供了一套**分布式、可异步并行**的奖励计算组件：把规则奖励、判别式/生成式奖励模型（DisRM/GenRM）统一抽象为一条计算链路，并实现了「采样（rollout）与奖励计算」流式重叠、训练循环不被奖励阻塞的异步机制。

twinkle-kit 目前**没有**对应的分布式奖励基础设施：

- 奖励仅限驱动进程内的本地同步规则函数（`Reward.__call__(trajectories) -> List[float]`）。
- 采样（vLLM sampler actor）→ 奖励（驱动进程同步）→ 优势 → 前反向，完全串行，奖励计算阻塞整个循环。

本次任务：把 verl 的**异步规则奖励计算能力**移植进 twinkle，使「采样 / 奖励计算 / 训练」三段在时间上重叠。

### 1.2 范围

**本次包含（In Scope）**

- 分布式规则奖励：多 Ray worker 并行计算。
- RewardManager 家族移植：`naive` / `dapo` / `gdpo` / `rate_limited` / `remote`。
- 同步 / 异步自定义奖励函数（自动识别 + 同步函数进 executor）。
- **异步奖励流水线**：提交-收集解耦、双缓冲、背压、按 key 对齐、错误降级。

**本次不包含（Out of Scope，预留接口）**

- DisRM / GenRM 模型推理服务、router、`classify`/`embeddings` HTTP 链路。
- 与训练器深度耦合的 off-policy 重排 / staleness 回放（TransferQueue 等价物）。

### 1.3 核心目标

| 编号 | 目标 | 验收口径 |
| --- | --- | --- |
| G1 | 奖励结果正确性 | 与同步基线实现逐项一致 |
| G2 | 采样不被奖励阻塞 | 时间线断言：B_k 采样完成即返回，奖励在后台算 |
| G3 | 模块低耦合可复用 | `reward_manager/` + `data.py` 零 twinkle 依赖，纯 CPU 可单测 |
| G4 | 平滑回退 | `reward.mode='sync'` 退回「提交即收集」，行为等价现状 |

---

## 2. verl 异步奖励机制要点（移植依据）

以下模式从 verl 现行实现中提取，作为移植的直接依据（`file:line` 为 verl `main` 分支现状）：

### 2.1 核心开关：`reward_loop_worker_handles`

`RewardLoopManager.reward_loop_worker_handles`（`verl/experimental/reward_loop/reward_loop.py:311-322`）：

```
非 None  ⟺ 奖励可以跑在 rollout 管线内（流式）
  = None ⟺ 奖励必须由训练侧串行补算（colocate）
```

- 规则奖励（无 RM）或「RM + 独立资源池」→ 非 None。
- trainer 侧据此分叉（`verl/trainer/ppo/v1/trainer_base.py:550-553`）：`handles is None` 才调用 `_compute_reward_colocate`，否则奖励早已在管线里算好。

**移植含义**：twinkle 的 `AsyncRewardPipeline` 即等价物——`enabled=True` 时奖励走管线；`enabled=False`（或 `mode='sync'`）时退回串行。

### 2.2 每样本一条 asyncio task + Ray actor 流式 compute_score

- Worker 批处理：逐样本 `asyncio.create_task` + `asyncio.gather`（`reward_loop.py:138-143`）。
- Agent loop 里每个样本完成后立即提交 reward：`random.choice(worker_handles).compute_score.remote(data)`（`verl/experimental/agent_loop/agent_loop.py:936-996`），同一 trajectory 内 reward 紧随其 rollout，不同样本间由 asyncio 并发。
- Worker 本身是 `@ray.remote` actor，可同时承接多个调用方。

### 2.3 fire-and-forget 提交 + 延迟收集

- `compute_rm_score`：先对所有 worker 提交 `compute_score_batch.remote(chunk)`，再统一 `ray.get([...])`（`reward_loop.py:348-354`）。
- `DataProtoFuture`：`concat(objrefs)` + 延迟 `get()`（`verl/protocol.py:1188-1228`）。
- twinkle 侧已有等价物 `LazyCollect`（`src/twinkle/infra/_ray/ray_helper.py`）——直接复用。

### 2.4 多级背压

| 层级 | 机制 | 参考 |
| --- | --- | --- |
| 生产者并发 | `asyncio.Queue(maxsize=128)` | `fully_async_rollouter.py:461` |
| 样本并发上限 | `max_concurrent_samples = replicas * concurrent_samples_per_replica`，超限 drain | `fully_async_rollouter.py:425, 970-978` |
| 消费队列 | `deque(maxlen=...)` 满则丢最老 + 暂停标志 | `message_queue.py:32-83`, `fully_async_rollouter.py:1142-1164` |
| 消费端 | 攒够 N 个样本才组 batch，天然解耦 | `fully_async_trainer.py:402` |

### 2.5 以 key/uid 对齐结果，不依赖顺序

- TransferQueue 以 `{uid}_{session_id}_{index}` 为键读写，reward 结果按**同一组 keys** 原样写回（`trainer_base.py:1411-1424`），batch 内顺序天然对应。
- 组内 GRPO advantage 依靠解析 key 尾部 `(uid, session_id, index)` 重排（`verl/trainer/ppo/v1/utils.py:178-217`），数据乱序不影响正确性。

**移植含义**：本设计每个 `RewardItem` 携带 `item_id`，结果按 id 回填——乱序安全。

### 2.6 同步/异步函数自适应

- `is_async_reward_score = inspect.iscoroutinefunction(compute_score)`；同步函数放进 `loop.run_in_executor`，避免阻塞事件循环（`reward_manager/*.py` 各实现一致）。
- 推荐外部 API/沙箱类奖励用异步函数（`docs/advance/reward_loop.rst` Rule-Based Reward 章节）。

### 2.7 错误处理约定

- 奖励模型 HTTP 请求：5xx 指数退避重试 16 次、4xx 直接抛（`reward_loop.py:157-195`）。
- `rate_limited` Manager 超时/异常**降级为 0 分继续训练**，异常信息写入 `reward_extra_info`（`reward_manager/limited.py`）。

---

## 3. twinkle 现状与差距

### 3.1 现状事实清单（已核验）

| 维度 | twinkle 现状 |
| --- | --- |
| 分布式基础设施 | Ray：`infra/_ray/resource_manager.py`（按 node 建 GPU/CPU placement group）、`ray_helper.py`（`create_workers` / `LazyCollect` / config registry）、`twinkle.initialize(mode='ray', groups=[DeviceGroup...])`、`@remote_class` |
| 数据格式 | 无 DataProto。TypedDict：`InputFeature`（input_ids/attention_mask/labels/completion_mask/length）、`Trajectory`（messages/tools/user_data/多模态）、`SampledSequence/SampleResponse/SamplingParams` |
| 奖励 | `Reward.__call__(trajectories, ground_truths) -> List[float]`，驱动进程内同步执行；`reward/` 下 gsm8k/format/math/mm/olympiad_bench；优点计算 `advantage/grpo.py` |
| 配置 | 扁平 dataclass 分组 + OmegaConf 读 YAML + ConfigRegistry（`{group}_{field}` 前缀映射、大小写/`-_` 不敏感、动态 key 进 `extra`） |
| 采样 | vLLM engine 内嵌在 sampler Ray actor（**无 HTTP 推理服务**） |
| HTTP 服务层 | Ray Serve + FastAPI gateway（`server/`），replica 注册 + sticky 路由，Client SDK 经 HTTP 驱动训练 |
| 测试/CI | pytest（tests/ 按模块分目录）、.github/workflows（citest/lint/citest_npu） |

### 3.2 已发现的契约缺口（重要）

`cookbook/rl/grpo/grpo.py:161` 把 `sequence.new_input_feature`（纯 token 字段，**无 `messages`**）传给 `GSM8KAccuracyReward` / `GSM8KFormatReward`——而这些奖励函数读取 `trajectory['messages']`（最后一条 assistant 消息）。当前该链路实际无法正确取分。

**结论**：twinkle 的「奖励输入契约」本身未收敛。移植方案必须显式定义输入契约（见 §5.1 与评审表 1），并在适配器层补齐。

### 3.3 差距总览

| 能力 | verl | twinkle（现状） | 差距 |
| --- | --- | --- | --- |
| 分布式奖励 worker | ✅ Ray + asyncio per-sample | ❌ 无 | 新建 |
| 奖励与采样重叠 | ✅ 流式/异步 | ❌ 串行阻塞 | 新建（本次核心） |
| 奖励函数注册/定制 | ✅ register/importlib | 部分（本地 Reward 类） | 增强 |
| 高级 Manager（限流/独立进程） | ✅ | ❌ | 移植 |
| 模型奖励服务 | ✅ | ❌（出范围） | 预留 |

---

## 4. 目标架构

### 4.1 新模块布局

```
src/twinkle/reward_loop/
├── __init__.py             # 导出 RewardLoopManager / AsyncRewardPipeline / register 等
├── data.py                 # RewardItem / RewardResult / 分块·重组工具（纯 Python，零框架耦合）
├── config.py               # RewardLoopArgs 扁平分组 + 配置解析（复用 twinkle CLI）
├── reward_manager/         # ⭐ 自 verl 移植，仅 run_single 参数 DataProto → RewardItem
│   ├── __init__.py         #   注意：import 副作用即完成 @register 注册
│   ├── base.py             #   RewardManagerBase（run_single 抽象 + assemble 工具）
│   ├── registry.py         #   register / get_reward_manager_cls
│   ├── naive.py            #   NaiveRewardManager（默认）
│   ├── dapo.py             #   DAPORewardManager（overlong 惩罚）
│   ├── gdpo.py             #   GDPORewardManager（experiment_name 注入）
│   ├── limited.py          #   RateLimitedRewardManager + AsyncTokenBucket（三层限流）
│   └── remote.py           #   RemoteRewardManager（独立 Ray 进程，CPU 密集隔离）
├── default_score.py        # 默认 compute_score 分派（简化版，不含 verl dataset 大表）
├── worker.py               # RewardLoopWorker：@remote_class，归属 CPU DeviceGroup 'reward'
├── pipeline.py             # AsyncRewardPipeline：提交/收集/双缓冲/背压/对齐（本次核心）
└── metrics.py              # 奖励耗时、重叠率、排队深度等流水线指标
```

### 4.2 分层原则

```
┌─────────────────────────────────────────────────────────┐
│ 驱动进程（训练脚本）                                       │
│   AsyncRewardPipeline（pipeline.py）                     │
│     ├─ 提交 batch（非阻塞）  ├─ 收集 rewards（按需阻塞）    │
└──────────────┬──────────────────────────┬────────────────┘
               │ remote                   │ remote
┌──────────────▼──────────────────────────▼────────────────┐
│ Ray Cluster                                               │
│   DeviceGroup('reward', device_type=CPU)                  │
│   RewardLoopWorker × N（worker.py，@remote_class）         │
│     └─ RewardManager.run_single(item)                     │
│          └─ compute_score（默认/自定义、同步/异步自适应）     │
└───────────────────────────────────────────────────────────┘
```

- **第一层（纯逻辑）**：`data.py` + `reward_manager/` + `default_score.py`。零 twinkle 依赖（仅 asyncio/torch/typing），可独立单测、可原样复用回其它框架。
- **第二层（适配）**：`worker.py` / `pipeline.py`。只做 twinkle 侧绑定点：`remote_class` 装饰、DeviceGroup 选择、LazyCollect/ray 收集、CLI 配置接入。

### 4.3 与 verl 的移植映射

| verl | twinkle 移植 | 改动要点 |
| --- | --- | --- |
| `RewardLoopManager.compute_rm_score` | `AsyncRewardPipeline.submit/collect` | 拆解成提交/收集两段，支持双缓冲 |
| `RewardLoopWorker` | `RewardLoopWorker`（`worker.py`） | `@remote_class`；输入从 DataProto 切片 → `RewardItem` |
| `pad_dataproto_to_divisor` + `chunk` | `data.py: split_items / reorder_by_id` | 纯 list 切块（分块取 chunks，无 padding 语义，逻辑等价） |
| `RewardManagerBase.assemble_rm_scores` | `data.py: assemble_scores` | 产出对齐 `item_id` 的 `List[RewardResult]` 或可选 token 级张量 |
| `get_default_compute_score` / 注册表 | `default_score.py` + `registry.py` | 按 `manager_name` 取默认函数；规则/自定义二选一 |
| `load_reward_manager` | `config.py: build_reward_manager` | 移除 verl 专有包装（sandbox_fusion 起步不移植） |
| `reward_loop_worker_handles`（流式开关） | `reward.mode='async'` + `pipeline.enabled` | 语义等价 |
| TransferQueue key 对齐 | `RewardItem.item_id` | 同语义、更轻量 |
| 背压（Queue/maxsize/pause） | `pipeline` 内 `asyncio.Queue` + `max_backlog` | 移植 |

---

## 5. 异步奖励流水线设计（核心章节）

### 5.1 数据契约

#### 5.1.1 输入 `RewardItem`（建议形态，最终见评审表 1）

```python
@dataclass
class RewardItem:
    item_id: str                    # 全局唯一对齐键，形如 "{uid}_{session_id}_{index}"
    data_source: str                # 数据集标识，默认函数按它分派（gsm8k/math/...）
    solution_str: str               # 解码后的有效响应文本（奖励函数主输入）
    ground_truth: str               # 标准答案
    extra_info: dict                # 奖励函数可能需要的任意上下文
    # 可选（供 token 级组装 / 高级 Manager 使用）
    response_ids: Any = None        # 原始 token ids（有效长度部分）
    attention_mask: Any = None      # 用于截取有效响应
    raw_prompt: str | None = None   # DisRM 预留（本期不用）
```

> 与 verl 奖励函数约定签名对齐：`compute_score(data_source, solution_str, ground_truth, extra_info, ...)`（见 verl `docs/advance/reward_loop.rst`）。
> 适配器（驱动进程内）负责从 twinkle 的 `SampledSequence.decoded` / `Trajectory.messages` / 数据集 `user_data` 组装 `RewardItem`——同时修复 §3.2 的契约缺口。

#### 5.1.2 输出 `RewardResult`

```python
@dataclass
class RewardResult:
    item_id: str
    reward_score: float
    reward_extra_info: dict = field(default_factory=dict)
```

### 5.2 负责切分与重组（`data.py`）

- `split_items(items, num_workers) -> list[list[RewardItem]]`：round-robin 均匀切分（等价 verl `chunk` 语义；twinkle 无 padding 需求，直接切即可）。
- `reorder_by_id(results) -> dict[item_id, RewardResult]`：乱序安全重组。
- `assemble_scores(results, items, mode) -> list[float]`：按 `items` 的原始顺序输出标量分列表（供 `GRPOAdvantage`）；`mode='token'` 时可选输出 token 级张量（对齐 verl `rm_scores` 约定，供未来扩展）。

### 5.3 Pipeline 运行模型

`AsyncRewardPipeline` 提供两个显式操作，把「计算」与「消费」解耦：

```python
class AsyncRewardPipeline:
    def submit(self, batch: list[RewardItem]) -> BatchHandle: ...
    def collect(self, handle: BatchHandle) -> list[RewardResult]: ...   # 仅此处阻塞
```

- `submit` **非阻塞**：切块 → 对各 `RewardLoopWorker` 发起 Ray 远程调用，持有 `ObjectRef` 列表；立即返回 `BatchHandle`。
- `collect` **按需阻塞**：`LazyCollect`/`ray.get` 拉取结果，按 `item_id` 回填对齐后返回。
- **warm-up**：首个 batch 提交后无先前批次可收集，训练 `collect` 返回空，下一迭代自动补上。

### 5.4 双缓冲时间线（重点）

```
迭代 k:
 ┌─────────────────────────────────────────────────────────────┐
 │ sampler.sample(B_k)          ← 采样（vLLM actor，异步引擎）   │
 │   └─ pipeline.submit(B_k)    ← 火抛：分发给 reward workers    │
 │ rewards(B_{k-1}) = pipeline.collect(k-1)                     │
 │                              ← 后台早已算完，此时近乎零阻塞    │
 │ advantage(rewards_{k-1}) → model.forward_backward(...)       │
 │ 指标上报（上一批真实 reward 均值/acc 等）                     │
 └─────────────────────────────────────────────────────────────┘
        │                    ▲
        │ 后台（Ray Cluster）
        ▼                    │
  RewardLoopWorker × N       │ 算完写回 ObjectRef，等待下次 collect
   └─ await run_single(item) │
       └─ asyncio.create_task per item
```

三段重叠：**采样 k 期间 → 奖励 k 在算；训练 k-1 期间 → 采样 k+1 已在跑**。带宽重叠率：

```
overlap_ratio = 1 - collect_wait_time / (sample_time + training_time)
```

由 `metrics.py` 持续统计（验收 G2 依据）。

### 5.5 并发与背压

- **worker 内并发**：透传 verl 模式——`compute_score_batch` 对 batch 内每条 item 建 asyncio task 并发；异步函数直接 `await`，同步函数进 `run_in_executor`。
- **consumer 背压**：`pipeline` 内部 `asyncio.Queue(maxsize=backlog)` 管理未收集批次；`backlog`（默认 2）为最大未收集批次数。训练消费过慢时 `submit` 阻塞、或按策略丢弃最老（`on_backlog_full: 'block' | 'drop_oldest'`）。
- **producer 并发**：多 `submit` 并发安全（内部锁 + 批次序号）。

### 5.6 模式与回退

| `reward.mode` | 行为 | 等价 verl 语义 |
| --- | --- | --- |
| `async`（默认） | 双缓冲异步管线，提交-收集解耦 | `reward_loop_worker_handles` 非 None（流式） |
| `sync` | submit 后立即 collect（提交即收集） | `_compute_reward_colocate`（串行补算） |

`sync` 模式保证：复杂场景、调试、或无 Ray 资源时可平滑回退到与现状等价的串行路径。

### 5.7 错误处理

- worker 异常携带 `item_id` 传播到 `collect`。
- 可配置 `on_error: 'raise' | 'zero'`：`zero` 降级为 0 分并记录 `reward_extra_info['error']`（对齐 verl `rate_limited` 行为），训练不中断。

### 5.8 奖励函数加载

- `reward.custom_reward_function_path/name`：动态 import（等价 verl `load_extern_object`）。
- 未配置自定义函数：按 `manager_name` 取默认打分（`default_score.py`，规则/数据集分派简化版）。
- 自定义 Manager：`manager_source='register' | 'importlib'`，支持用户注册 `@register` 或按模块路径加载类。

---

## 6. 配置接入

### 6.1 新增扁平分组 `RewardLoopArgs`（group 名 `reward`）

复用 twinkle CLI 的 ConfigRegistry（`{group}_{field}` 前缀映射、大小写与 `-_` 不敏感），新增 dataclass：

```python
@dataclass
class RewardLoopArgs:
    # 分布式
    num_workers: int = 8
    # 奖励函数
    custom_reward_function_path: str | None = None
    custom_reward_function_name: str = 'compute_score'
    # 奖励管理器
    manager_name: str = 'naive'                       # naive/dapo/gdpo/rate_limited/remote
    manager_source: str = 'register'                  # register | importlib
    manager_module_path: str | None = None            # importlib 模式
    manager_module_name: str = 'RewardLoopManager'    # importlib 模式
    unknown_rewards: str = 'warn'                     # 未匹配规则如何处置
    # 异步流水线
    mode: str = 'async'                               # async | sync
    backlog: int = 2                                  # 未收集批次上限
    on_backlog_full: str = 'block'                    # block | drop_oldest
    on_error: str = 'raise'                           # raise | zero
    # rate_limited 专用
    max_rpm: int | None = None
    max_tpm: int | None = None
    max_concurrent: int = 1
    timeout: float = 300.0
    # remote 专用
    reward_worker_executors: int | None = None
    # dapo 专用（overlong 惩罚）
    reward_kwargs: dict = field(default_factory=dict)
```

CLI/YAML 用法（`{group}_{field}` 扁平键）：

```yaml
reward_num_workers: 8
reward_manager_name: naive
reward_custom_reward_function_path: /path/to/rules.py
reward_custom_reward_function_name: compute_score
reward_mode: async
reward_backlog: 2
reward_on_error: zero
```

### 6.2 嵌套 verl 配置 → 扁平 twinkle 键 映射

| verl 配置（嵌套） | twinkle 扁平键 |
| --- | --- |
| `reward.num_workers` | `reward_num_workers` |
| `reward.custom_reward_function.path / name / reward_kwargs` | `reward_custom_reward_function_path / _name`（kwargs 并入 `reward_reward_kwargs`，JSON 串） |
| `reward.reward_manager.name / source / module.path / module.name` | `reward_manager_name / _source / _module_path / _module_name` |
| `reward.reward_manager.reward_kwargs`（dapo 的 overlong_buffer_cfg） | `reward_reward_kwargs`（JSON/YAML 内联 dict） |
| `reward.reward_model.*` | 本期不启用（预留 `reward_reward_model_*`） |
| `reward.sandbox_fusion.*` | 本期不移植（预留 `reward_sandbox_fusion_*`） |

> 说明：twinkle 配置解析器支持动态 key 落入 `args.extra`；内联 dict 字段（如 `reward_kwargs`）建议 YAML 内联对象或 JSON 字符串，P1 实现时以 ConfigRegistry 实际能力校准。

---

## 7. 训练器/示例集成

### 7.1 `cookbook/rl/grpo/grpo.py` 改造前后对照

**现状（串行）：**

```python
sample_responses = sampler.sample(expand_prompts, sampling_params)
# 组装 all_input_data / all_old_logps / all_completion_lengths
total_rewards, format_rewards, accuracy_rewards = compute_rewards(all_input_data)   # ← 阻塞
advantages = advantage_fn(total_rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()
for mb in mini_batches(model.forward_backward): ...
```

**改造后（异步双缓冲）：**

```python
pipeline = AsyncRewardPipeline.from_args(args)        # 自动建 CPU reward workers
last_handle = None
for k, batch in enumerate(dataloader):
    sample_responses = sampler.sample(expand_prompts, sampling_params)
    items = adapter.to_reward_items(sample_responses, k)   # SampledSequence → RewardItem（含 ground_truth/messages）
    handle = pipeline.submit(items)                         # 非阻塞
    results = pipeline.collect(last_handle) if last_handle else []   # 收集上一批（后台已算完）
    last_handle = handle
    rewards = adapter.to_reward_lists(results, k-1)         # 按 item_id 对齐 → 每组 rewards
    metrics.accumulate(...)                                 # 用上一批真实奖励指标
    advantages = advantage_fn(rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()
    for mb in mini_batches(model.forward_backward): ...
```

### 7.2 适配器（`adapter`）

职责（放置于 `cookbook` 或 `reward_loop/adapters/`）：

- 从 `SampledSequence` 取 `decoded`（或按 `InputFeature.input_ids` 解码）得到 `solution_str`；
- 从数据集 meta / `Trajectory.user_data` 取 `ground_truth` 与 `data_source`；
- 组装 `extra_info`（如 `messages`、问题文本、num_turns 等，按奖励函数需要补齐）；
- 生成 `item_id = f"{uid}_{session_id}_{index}"`，供对齐与 GRPO 组内重排。

### 7.3 与 twinkle 组件的关系

- 复用：`twinkle.advantage.GRPOAdvantage`（输入 `List[float]` 不变）、`twinkle.metric.*`（奖励指标不改接口）。
- 不侵入：`model.forward_backward` / `sampler.sample` / `CheckpointEngineManager` 均不改动。
- 引入点仅为：新模块 + 示例脚本中的 `compute_rewards` 调用替换。

---

## 8. 分阶段实施计划

### P0 决策与 Spike（0.5 周）

- [ ] 验证 `DeviceGroup(name='reward', device_type='CPU')` + `@remote_class` 能否创建纯 CPU 远程 worker（读 `infra/_ray/resource_manager.py` 的 CPU 分组路径；不成立则回退：`RayHelper` 低层 actor 封装或 `ray.remote` 直用）。
- [ ] 定数据契约（评审表 1 拍板）与 `RewardItem` 字段定稿。
- [ ] 定奖励函数调用约定：兼容 `Reward.__call__(trajectories)` 还是采用 verl 的 `compute_score(data_source, solution_str, ground_truth, extra_info)`（建议后者 + 兼容包装层）。

### P1 核心流水线（1.5 周）

- [ ] 移植 `data.py`（RewardItem/RewardResult/分块/重组）。
- [ ] 移植 `reward_manager/` 家族（base/registry/naive/dapo/gdpo/limited/remote），`run_single` 参数改为 `RewardItem`；`limited.py` 的 AsyncTokenBucket 原样移植。
- [ ] 移植 `default_score.py` 简化版。
- [ ] 实现 `worker.py`（@remote_class CPU worker）。
- [ ] 实现 `pipeline.py`（submit/collect/双缓冲/背压/错误降级/metrics）。
- [ ] 单元测试（纯 CPU）：对齐、切分重组、AsyncTokenBucket、rate_limited 降级、double-buffer 顺序、sync 模式等价。

### P2 集成（1 周）

- [ ] 接入 `cookbook/rl/grpo/grpo.py`（改造前后对照见 §7.1）。
- [ ] 本机 `mode='ray'` e2e：正确性（与同步基线逐项一致）+ 效率（timeline 断言采样不被奖励阻塞）。
- [ ] 指标与日志（`metrics.py`）。

### P3 增强（0.5~1 周）

- [ ] 异步自定义函数示例（aiohttp + 沙箱）。
- [ ] README / 组件文档（`docs/source_zh/组件/` 增补）。
- [ ] `mode='sync'` 回退路径回归。

### P4 未来扩展（不在本次，预留接口）

- [ ] DisRM/GenRM 服务（按评审表 2 决策）＋ DisRM 预处理 + `classify`/`embeddings` 请求路径。
- [ ] 每样本 true streaming（需 sampler 支持异步逐条产出）。
- [ ] token 级 `rm_scores` 张量组装（对齐 verl `assemble_rm_scores` 语义）。

---

## 9. 测试与验收

### 9.1 单元测试（P1，纯 CPU）

| 用例 | 断言 |
| --- | --- |
| `split_items` / `reorder_by_id` | 均匀切分；乱序输入按 id 还原 |
| `AsyncTokenBucket` | 速率/QPS 与突发语义（移植自 verl 测试） |
| `RateLimitedRewardManager` | RPM/TPM/并发限制；超时/异常降级 0 分 |
| `DAPORewardManager` | overlong 惩罚公式与日志字段 |
| pipeline 双缓冲 | 先提交后收集、warm-up、收集顺序稳定 |
| `mode='sync'` | 与直接同步调用结果一致 |

### 9.2 e2e 测试（P2）

- `mode='ray'` 小规模 grpo 跑通（CPU 或单卡 GPU）。
- **正确性**：随机种子固定，异步流水线总奖励 = 同步基线总奖励（逐项）。
- **效率（G2）**：在 `submit` 与 `collect` 处打点，断言 `sample → submit 返回` 与 `collect → rewards 得到` 之间时间差远小于奖励计算总耗时（即奖励已后台算完）。
- **背压**：用慢速 mock 奖励（sleep 放大耗时）验证 `backlog` 生效：消费慢时 submit 阻塞或按策略丢老。

### 9.3 验收清单

- [ ] G1 正确性逐项一致
- [ ] G2 timeline 断言通过
- [ ] G3 `reward_manager/` + `data.py` 无 `import twinkle`（静态检查）
- [ ] G4 `mode='sync'` 行为等价现状

---

## 10. 风险与开放项

| # | 风险/开放项 | 影响 | 缓解 |
| --- | --- | --- | --- |
| R1 | twinkle `remote_class` 的 CPU-only 路径未验证 | worker 创建方式未定 | P0 spike；回退 Ray Helper 低层封装 |
| R2 | 奖励函数输入契约现状不严谨（InputFeature 无 messages） | 组装 RewardItem 需界定来源 | 适配器统一组装；评审表 1 拍板 |
| R3 | `data_source` 分派依赖 verl 数据集命名 | 默认打分函数能力受限 | 本期仅规则奖励；`default_score.py` 文档化支持范围 |
| R4 | verl `pad_dataproto_to_divisor` 的 padding 语义在本设计被简化 | 不等长样本切块 | 纯 list 切分逻辑等价（无跨 batch padding 损耗） |
| R5 | YAML 内联 dict（reward_kwargs）与扁平键解析兼容性 | 配置体验 | P1 以 ConfigRegistry 能力校准；必要时 JSON 字符串 |
| R6 | 采样器为 batch 式返回，per-sample 真流式需 sampler 改造 | 重叠粒度 | 本期 batch 级双缓冲已达成重叠；flow 作为 P4 |

---

## 附录 A：评审待决事项

### 评审表 1：奖励 worker 数据契约

> 背景：twinkle 无 verl `DataProto`（batch 张量 + non_tensor_batch + meta_info）。奖励 worker 输入契约需拍板。

| 维度 | 选项 A：轻量 `RewardItem`（⭐ 推荐） | 选项 B：复用 twinkle `InputFeature`/`Trajectory` | 选项 C：引入 DataProto 等价物 |
| --- | --- | --- | --- |
| 形态 | 自建 dataclass：item_id/data_source/solution_str/ground_truth/extra_info | 直接传 `Trajectory` 列表 或 `InputFeature` + 附加并行列表 | 新 `DataProto`（TensorDict batch + non_tensor + meta） |
| 框架耦合 | 零耦合（纯 Python） | 与 twinkle 强耦合 | 与 twinkle 强耦合、且是整套新数据基建 |
| 移植/复用 | 可原样搬回 verl 或第三方 | 仅 twinkle 可用 | 仅 twinkle 可用 |
| 覆盖自定义字段 | 高（extra_info 任意 dict） | 中（依赖 Trajectory 字段约定） | 高 |
| 与现有奖励函数兼容 | 中（需适配器把 messages 等放进 extra_info） | 高（`Reward.__call__(trajectories)` 直接兼容） | 低 |
| 实现成本 | 低（1 个 dataclass + 适配器） | 低（但契约含糊、易再踩坑） | 高（改造成本远超奖励模块本身） |
| 风险 | 适配器需覆盖奖励函数所需字段 | 现状已出现 InputFeature/messages 缺口（§3.2）；治标不治本 | 过度设计 |

**建议**：选 A。以 `RewardItem` 为唯一输入契约；驱动侧适配器从 `Trajectory`/数据集组装；奖励函数按需从 `extra_info` 取 `messages` 等字段（或要求自定义函数遵循 verl 四参数签名）。

### 评审表 2：模型推理服务托管（P4 DisRM/GenRM 扩展用）

> 背景：twinkle 的 vLLM 内嵌于 sampler actor（无独立 HTTP 推理服务）；verl 用「vLLM 独立服务 + naive_router（FastAPI 代理）」。

| 维度 | 选项 A：vLLM 子进程 + 移植 naive_router（⭐ 推荐） | 选项 B：Ray actor 内嵌 vLLM engine + 复用 server 层 | 选项 C：可插拔后端抽象 |
| --- | --- | --- | --- |
| 实现忠实度 | 与 verl 一致，风险最低 | twinkle 风格，但需为 reward 模式扩展 sampler 的 embedding/pooling | 接口层先定，两种后端皆可 |
| 侵入面 | 独立进程，零改动既有 server/sampler | 侵入 vLLM 相关封装（sampler/engine） | 仅新增抽象层 + 默认实现 |
| GenRM 自定义函数调用 | 经 `reward_router_address`（HTTP）天然契合 | 需 gateway 暴露通道 | 依赖所选默认实现 |
| DisRM 取分 | `/classify`（vLLM pooling）、`/v1/embeddings`（SGLang） | actor 内 pooling 模式 | 依赖所选默认实现 |
| 负载均衡/确定性 | 移植 naive_router（least-loaded + crc32 确定性路由） | 复用 Ray Serve replica/sticky | 依赖所选默认实现 |
| 与 twinkle 一致性 | 低（独立 FastAPI 子进程） | 高（全部 Ray 化） | 中 |
| 实施成本 | 低 | 中高 | 中（需两次实现或一次双轨） |

**建议**：P4 启动时默认选 A（忠实惠移植、隔离风险），接口上按 C 预留 `RewardModelBackend` 抽象，后续需要 twinkle 原生化时切 B 不影响上层。

---

## 附录 B：参考材料

- verl 源码：`verl/experimental/reward_loop/`（reward_loop.py / reward_model.py / reward_manager/ / router/）
- verl 官方文档：`docs/advance/reward_loop.rst`（分布式、流式奖励、混合奖励用法）
- verl 异步机制：`verl/experimental/fully_async_policy/`、`verl/experimental/agent_loop/`、`docs/data/transfer_queue.md`
- twinkle 源码：`src/twinkle/infra/_ray/`、`src/twinkle/reward/`、`cookbook/rl/grpo/grpo.py`、`src/twinkle/cli/cli.py`