# 最小 GRPO + Reward Loop 示例

本目录展示如何用尽量少的组件构建一个 GRPO 训练循环，并接入 `twinkle.reward_loop`。

提供三个示例（另有本地模式示例见下文）：

- `minimal_grpo.py`：同步式（submit 后立即 collect），入门最直观。**需 twinkle-server（含 tinker）**。
- `minimal_grpo_streaming.py`：流式双缓冲（submit 当前 batch，后台算奖励，训练上一 batch），展示重叠。**需 twinkle-server（含 tinker）**。
- `minimal_grpo_true_stream.py`：真流式（每个样本一产出就提交 reward），展示逐样本粒度。**需 twinkle-server（含 tinker，且 `sampler_type: vllm_async`）**。
- `minimal_grpo_local.py`：本地组件模式，**无需 twinkle-server、无需 tinker**，见下文专属章节。

## 数据流

```text
Dataset / DataLoader
        ↓
Sampler rollout
        ↓
DataPlane decoded rows
        ↓
RewardItem
        ↓
AsyncRewardPipeline.submit / collect
        ↓
GRPOAdvantage
        ↓
DataPlane 写回 reward / advantage
        ↓
Model forward_backward + optimizer step
```

示例代码：`minimal_grpo.py`

> 这是教学用小规模示例，不包含生产环境常见的 partition staleness、失败恢复、checkpoint 管理和多阶段流水线编排。

## 前置条件

需要准备：

- Twinkle 及其 Python 依赖；
- 可访问的 GSM8K 数据集；
- Twinkle server（提供模型和 sampler 组件）；
- 可用的基础模型；
- `DataPlaneClient` 对应的数据平面服务。

启动服务的方式取决于当前部署。客户端默认连接：

```text
http://localhost:8000
```

也可以通过环境变量修改。

## 运行

```bash
python cookbook/rl/reward_loop/minimal_grpo.py
```

建议先使用小配置验证链路：

```bash
TWINKLE_MAX_STEPS=2 \
TWINKLE_BATCH_SIZE=2 \
TWINKLE_NUM_GENERATIONS=2 \
TWINKLE_REWARD_MODE=sync \
python cookbook/rl/reward_loop/minimal_grpo.py
```

Windows PowerShell：

```powershell
$env:TWINKLE_MAX_STEPS="2"
$env:TWINKLE_BATCH_SIZE="2"
$env:TWINKLE_NUM_GENERATIONS="2"
$env:TWINKLE_REWARD_MODE="sync"
python cookbook/rl/reward_loop/minimal_grpo.py
```

## 关键 API

### 1. RewardItem

每个待评分样本需要一个稳定且唯一的 `item_id`：

```python
RewardItem(
    item_id="step-0/prompt-0/sample-0",
    data_source="gsm8k",
    solution_str="模型生成的答案",
    ground_truth="标准答案",
    extra_info={
        "prompt": prompt,
        "messages": prompt.get("messages", []),
    },
)
```

`extra_info` 用于携带现有批量 Reward 适配器需要的上下文。GSM8K 奖励需要能够找到 prompt、messages 和 ground truth。

### 2. 奖励函数契约

Reward Loop 使用单样本四参数函数：

```python
def compute_score(data_source, solution_str, ground_truth, extra_info):
    return 1.0
```

也支持异步函数：

```python
async def compute_score(data_source, solution_str, ground_truth, extra_info):
    return 1.0, {"source": "remote"}
```

返回值可以是：

```python
1.0
(1.0, {"matched": True})
{"score": 1.0, "extra_info": {"matched": True}}
```

### 3. 提交与收集

```python
pipeline = AsyncRewardPipeline(
    num_workers=2,
    mode="sync",
    worker_kwargs={"compute_score": compute_score},
)

handle = pipeline.submit(items)
results = pipeline.collect(handle)
pipeline.close()
```

`collect()` 返回结果会按照原始 `items` 顺序排列，不依赖 worker 完成顺序。

## sync 和 async

### `sync`

```python
TWINKLE_REWARD_MODE=sync
```

提交后立即收集，最容易调试，适合：

- 第一次接入；
- 验证奖励正确性；
- 与旧同步实现对比。

### `async`

```python
TWINKLE_REWARD_MODE=async
```

适合把奖励计算与下一批 rollout 或训练重叠。异步 Python 代码中应将同步 pipeline API 放到线程中：

```python
handle = await asyncio.to_thread(pipeline.submit, items)
results = await asyncio.to_thread(pipeline.collect, handle)
```

本示例为了保持代码最小，仍然按单 batch 完成 submit/collect；生产级双缓冲可参考：

```text
submit(batch_k)
collect(batch_k-1)
train(batch_k-1)
```

## 流式双缓冲示例

`minimal_grpo_streaming.py` 实现双缓冲，奖励计算与训练、下一批 rollout 重叠。

核心时间线：

```text
iteration k:
  sampler.sample(batch_k)                  ← 采样
  pipeline.submit(batch_k)                 ← 火抛，不等待
  rewards(batch_k-1) = pipeline.collect    ← 后台早已算完，几乎不阻塞
  advantage(batch_k-1) → train(batch_k-1)
```

首个 batch 提交后没有上一批可收集，属于 warm-up；循环结束时对最后一个在飞 batch 做一次 flush（collect + 训练）。

运行：

```bash
python cookbook/rl/reward_loop/minimal_grpo_streaming.py
```

关键点：

- `REWARD_BACKLOG` 至少为 2（当前 batch + 上一 batch 同时在 pipeline 中）。
- 默认 `MAX_STEPS=3`、`BATCH_SIZE=2`、`NUM_GENERATIONS=2`，便于观察重叠。
- `submit` 与 `collect` 都通过 `asyncio.to_thread` 执行，避免阻塞事件循环。

与同步版的差异：

| 维度 | `minimal_grpo.py` | `minimal_grpo_streaming.py` |
| --- | --- | --- |
| 奖励时机 | submit 后立即 collect | submit(当前批) + collect(上一批) |
| 重叠 | 无 | 奖励与采样/训练重叠 |
| warm-up | 无 | 首个 batch 贡献 warm-up |
| 适用 | 入门/调试/验证正确性 | 理解重叠机制 |

> 该示例仍是教学用最小双缓冲，不含 partition staleness、失败恢复、checkpoint 管理和多阶段流水线编排。

## 真流式示例（逐样本提交）

`minimal_grpo_true_stream.py` 展示**每个样本一产出就提交**。

依赖链路（改造后的真流式能力）：

```text
VLLMSamplerTQ（per-sequence task 追踪）
  → HTTP NDJSON 流端点（逐样本 put_rows 增量落盘到预分配 DataRef）
  → client.stream_sample_to_data_plane() 异步生成器
  → 示例逐样本构造 RewardItem 并 pipeline.submit
```

事件流：

```text
{"event":"progress","index":3,"row":{...},"ref":{...},"done":4,"total":8}   # 每个样本一个
{"event":"ref","ref":{...}}                                                  # 全部完成，终态
{"event":"error","error":"..."}                                              # 失败
```

- `StreamSample`：单个已完成序列（`row` 与整组落盘行同构，`ref` 为预分配 DataRef，`index` 为全局序列索引 `prompt_index × num_generations + generation_idx`）。
- `StreamComplete`：携带写满全部行的 DataRef，后续 `aappend` / `forward_backward` 与旧流程一致。

运行：

```bash
python cookbook/rl/reward_loop/minimal_grpo_true_stream.py
```

关键点：

- 拿到 `StreamSample` 立即 `pipeline.submit([RewardItem(...)])`，组内其余样本还在采样。
- `pipeline` 的 `backlog` 必须 ≥ 一个 step 的样本总数（逐样本提交会产生等量 handle）。
- advantage 仍按组聚合：等 `StreamComplete` 后再统一 `collect` 各 handle、算组内优势。
- `item_id` 按全局 `index` 构造，收集结果乱序安全。

与双缓冲/同步的对比：

| 维度 | `minimal_grpo.py` | `minimal_grpo_streaming.py` | `minimal_grpo_true_stream.py` |
| --- | --- | --- | --- |
| 提交粒度 | 整批 | 整批 | **每样本** |
| 奖励开始时机 | 采样完成后 | 整批提交后 | **样本一产出即提交** |
| 重叠粒度 | 无 | batch 间 | batch 内 + batch 间 |
| backlog 需求 | 1 | ≥ 2 | ≥ 样本总数 |
| 适用 | 入门/调试 | 理解重叠 | 奖励昂贵 / 长生成 / 组内长尾 |

> 依赖真流式服务端改造（`sample_to_data_plane_stream` 端点）；当前示例仍是教学用最小实现，不含 partition staleness、失败恢复和 checkpoint 管理。

## 本地模式（免 server / 免 tinker）

`minimal_grpo_local.py` 使用 twinkle 的**本地组件**直接训练，完全不需要 twinkle-server 部署：

```text
twinkle.initialize(mode='ray', groups=[model, sampler])   # 自动起本地 Ray
  + 本地 vLLMSampler（GPU 组）
  + TransformersModel / MegatronModel（GPU 组）
  + AsyncRewardPipeline（纯 Python 线程池 worker）
```

### 为什么不需要 tinker

`tinker` 只被 twinkle-server 部署层（`src/twinkle/server/**`）引用；本地训练组件（`twinkle.sampler` / `twinkle.model` / `twinkle.reward` / `twinkle.advantage` / reward_loop）零 tinker 依赖。训练机装不了 tinker 时，本地模式是唯一且官方支持的路径（`cookbook/rl/grpo/grpo.py` 即此模式）。

### 运行

```bash
# 依赖：torch / transformers / vLLM（GPU），无需 twinkle-server、无需 tinker
python cookbook/rl/reward_loop/minimal_grpo_local.py
```

环境变量：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `TWINKLE_MODEL_ID` | `ms://Qwen/Qwen3.5-4B` | 基础模型 |
| `TWINKLE_TEMPLATE_CLS` | 按模型名自动选择 | template 类名（Qwen3.5/3.6 → `Qwen3_5Template`，其他 → `Template`） |
| `TWINKLE_MODEL_GPUS` / `TWINKLE_SAMPLER_GPUS` | `1` / `1` | model / sampler 各占 GPU 数 |
| `TWINKLE_USE_MEGATRON` | `0` | `1` 时用 MegatronModel |
| `TWINKLE_NUM_GENERATIONS` | `4` | 每个 prompt 生成数 |
| `TWINKLE_MAX_STEPS` | `20` | 最大训练步数 |
| `TWINKLE_BATCH_SIZE` / `TWINKLE_MINI_BATCH_SIZE` | `4` / `4` | batch 与 mini-batch |
| `TWINKLE_REWARD_MODE` | `async` | `async` 双缓冲 / `sync` 提交即收集 |
| `TWINKLE_REWARD_NUM_WORKERS` / `TWINKLE_REWARD_BACKLOG` | `2` / `2` | reward 并发与在飞批次数 |

**模型类不需要手动指定**：示例不传 `model_cls`，`TransformersModel` 会从模型的 `config.architectures` 自动解析类名（兜底 `AutoModelForCausalLM`），因此各种 Qwen / Llama / 其他架构都能直接跑。仅 template 需要按模型选择（见 `TWINKLE_TEMPLATE_CLS`）。

### 数据契约说明（重要）

示例的 dataset **不调用 `encode()`**：

- `Dataset.map(GSM8KProcessor(...))` 后的行是普通 dict（`messages` + `user_data` 含 ground truth），`DataLoader` 原样直通；
- `sampler.sample()` 检测到无 `input_ids` 会自动用已设置的 template 编码；
- 返回的 `SampledSequence.new_input_feature` 保留 `messages`/`user_data` 并追加 assistant 回答，因此：
  - reward 适配器（`gsm8k_score`）能直接取到回答与标准答案；
  - 同一份 feature 直接供 `model.forward_backward` 训练。

### 与 client/server 示例的定位

| 示例 | 架构 | 前置 |
| --- | --- | --- |
| `minimal_grpo.py` / `minimal_grpo_streaming.py` | twinkle-server + DataPlane | server（含 tinker） |
| `minimal_grpo_true_stream.py` | twinkle-server 真流式端点 | server（含 tinker，`vllm_async`） |
| `minimal_grpo_local.py` | 本地组件 | 仅训练机（无 tinker） |

## 替换为自定义奖励

将示例中的 `gsm8k_score` 替换为：

```python
def my_score(data_source, solution_str, ground_truth, extra_info):
    answer = solution_str.strip()
    target = ground_truth.strip()
    return float(answer == target), {"exact_match": answer == target}
```

然后：

```python
pipeline = AsyncRewardPipeline(
    num_workers=2,
    worker_kwargs={"compute_score": my_score},
)
```

也可以从模块路径加载：

```python
pipeline = AsyncRewardPipeline(
    num_workers=2,
    worker_kwargs={
        "custom_reward_function_path": "my_rewards",
        "custom_reward_function_name": "compute_score",
    },
)
```

## 常见问题

### 为什么不能直接传 `InputFeature`？

Reward Loop 的输入契约是 `RewardItem`。需要显式提供 `solution_str`、`ground_truth` 和 `item_id`，这样可以避免奖励函数拿不到消息内容或结果错位。

### 为什么一定要有 `item_id`？

不同 worker 可能以不同顺序完成。`item_id` 用于把结果重新对齐到原始 batch。

### 如何处理奖励函数异常？

默认建议：

```python
on_error="raise"
```

验证完成后，若训练确实允许异常样本降级，再考虑：

```python
on_error="zero"
```

### 是否需要 Ray？

示例依赖当前 client/server、sampler 和 DataPlane 组件。Reward Loop 本地逻辑可单独运行；分布式 reward worker 则需要按 Twinkle 的 Ray 资源配置启动 CPU worker。

## 环境变量

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `TWINKLE_MODEL_ID` | `Qwen/Qwen3.5-4B` | 基础模型 |
| `TWINKLE_TEMPLATE_MODEL_ID` | 与模型相同 | 模板模型 |
| `TWINKLE_TEMPLATE_CLS` | 自动选择 | 模板类 |
| `TWINKLE_SERVER_URL` | `http://localhost:8000` | Twinkle server 地址 |
| `TWINKLE_SERVER_TOKEN` | `EMPTY_TOKEN` | 服务 token |
| `TWINKLE_MAX_STEPS` | `2` | 最大训练步数 |
| `TWINKLE_BATCH_SIZE` | `2` | prompt batch size |
| `TWINKLE_NUM_GENERATIONS` | `2` | 每个 prompt 的生成数量 |
| `TWINKLE_REWARD_MODE` | `sync` | `sync` 或 `async` |
| `TWINKLE_REWARD_NUM_WORKERS` | `2` | reward worker 数量 |

## 与生产示例的区别

生产级异步 GRPO 通常还需要：

- rollout partition 和 staleness 控制；
- reward/rollout/trainer 背压；
- DataRef 异常清理；
- checkpoint 和恢复；
- 多 reward manager；
- Ray CPU placement group；
- 监控、日志和指标。

请将本示例作为最小接入模板，而不是直接用于大规模生产训练。
