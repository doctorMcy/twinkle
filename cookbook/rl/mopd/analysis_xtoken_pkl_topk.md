# NeMo-RL xToken 算法中 P-KL 的 top-k 机制分析

## 问题背景

NeMo-RL 中有一个 xToken（cross-tokenizer distillation）算法，其中计算 P-KL 时会用到 top-k。本文档解答两个核心问题：

1. P-KL 中 top-k 的具体实现代码在哪里？
2. 计算 P-KL 是对**异构词表**的两个模型进行的，使用 top-k 不会出现语义不一致吗？这是如何解决的？

---

## 1. 代码位置总览

P-KL 的 top-k 相关代码分布在两个核心文件中：

### 1.1 主文件：`nemo_rl/algorithms/loss/loss_functions.py`

| 方法 | 行号 | 功能 |
|------|------|------|
| `CrossTokenizerDistillationLossFn._compute_p_kl()` | 2255–2411 | **P-KL 完整主流程**：投影 → top-k 选取 → 双方切片 → chunk 对齐 → KL |
| `CrossTokenizerDistillationLossFn._direct_topk_kl()` | 1838–1895 | 同词表 teacher 的 top-k KL 路径（无投影，直接 teacher 侧选 top-k + 双方切片） |
| `CrossTokenizerDistillationLossFn._compute_teacher_kd()` | 1737–1808 | 调度器：按 teacher 是同词表还是跨词表分别调用上述两路径 |
| `CrossTokenizerDistillationLossFn._compute_gold()` | 2413–2625 | gold-loss 路径（对比参考：不打 top-k，打 exact token map 分区） |

### 1.2 辅助文件：`nemo_rl/algorithms/x_token/loss_utils.py`

| 函数 | 行号 | 功能 |
|------|------|------|
| `select_teacher_topk_indices()` | 235–259 | **核心 top-k 选取算子**：在 teacher 全词表 logits 上 per-vocab 取 max 后 topk |
| `project_student_to_teacher_vocab()` | 196–232 | 通过稀疏投影矩阵 `M [V_s, V_t]` 将 student 分布映射到 teacher 词表 |
| `chunk_average_log_probs()` | 132–160 | chunk 级平均聚合（处理 token 对齐后不同 chunk 长度） |
| `get_sparse_projection_matrix()` | 712–789 | 加载并缓存稀疏投影矩阵 |
| `parse_projection_file()` | 616–691 | 解析投影矩阵文件（支持 dense top-k 格式和 sparse multi-token 格式） |

---

## 2. P-KL 的完整执行流程

P-KL 的核心实现位于 `_compute_p_kl()` 方法（第 2255–2411 行）。其执行流程如下：

### Step 1：Student 侧 softmax + 投影

```python
# nemo_rl/algorithms/loss/loss_functions.py, 第 2294–2307 行
student_log_probs = vocab_parallel_log_softmax(
    student_logits, T, tp_group=tp_group       # temperature-scaled, TP-aware
)
student_probs = student_log_probs.exp()         # [B, T_s_local, V_s_local]

sparse_projection = get_sparse_projection_matrix(
    projection_matrix_path, device,
    student_vocab_size=self.student_vocab_size,
    teacher_vocab_size=teacher_vocab_size,
)                                                # [V_s, V_t] sparse COO, fp32

projected_full = project_student_to_teacher_vocab(
    student_probs, sparse_projection, tp_group=tp_group
)                                                # [B, T_s_local, V_t]
```

这一步将 student 的完整概率分布通过离线预计算的投影矩阵 M 映射到 teacher 的词表空间。

### Step 2：Teacher 侧选取全局 top-k

```python
# nemo_rl/algorithms/loss/loss_functions.py, 第 2334–2337 行
vocab_topk = min(self.vocab_topk, full_teacher_vocab_size)
global_top_indices = select_teacher_topk_indices(
    teacher_full_logits, vocab_topk, cp_group=cp_group
)   # [k]
```

`select_teacher_topk_indices` 的实现（第 235–259 行）：

```python
def select_teacher_topk_indices(
    teacher_logits: torch.Tensor,
    k: int,
    *,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    vocab_size = teacher_logits.shape[-1]
    with torch.no_grad():
        # 展平 (B * T_t)，每个 vocab 取 max 作为 importance
        teacher_flat = teacher_logits.reshape(-1, vocab_size)
        importance = teacher_flat.max(dim=0).values

        # CP 分片场景：all_reduce(MAX) 保证所有 rank 选到相同子集
        if cp_group is not None and torch.distributed.get_world_size(cp_group) > 1:
            torch.distributed.all_reduce(
                importance, op=torch.distributed.ReduceOp.MAX, group=cp_group
            )

        # 取 top-k 并排序，保证索引有序
        top_indices = torch.topk(importance, k=k, dim=-1).indices
        return top_indices.sort().values
```

**关键设计点：**
- 只在 teacher 侧做一次 top-k 选取
- 重要性度量 = 对展开的 `(B * T_t)` 维度取每个 vocab 的 max
- CP 分片时先 `all_reduce(MAX)` 保证一致
- 结果排序，避免因索引顺序导致的语义错位

### Step 3：双方切片到相同 k 列

```python
# nemo_rl/algorithms/loss/loss_functions.py, 第 2339–2346 行
# 双方都用同一个 global_top_indices 切片
projected_topk = projected_full[..., global_top_indices]  # [B, T_s, k]
teacher_topk_logits = teacher_full_logits[
    ..., global_top_indices
]                                                          # [B, T_t, k]

# 双方各自在 k 维子空间内 renormalize
target_log_probs = torch.log_softmax(
    teacher_topk_logits / T, dim=-1
)                                                          # [B, T_t, k]
```

### Step 4：Chunk 对齐 + KL 计算

```python
# nemo_rl/algorithms/loss/loss_functions.py, 第 2348–2409 行
# Chunk 平均
proj_chunks, proj_sizes = chunk_average_log_probs(
    projected_topk, student_chunk_id, max_chunks, cp_group=cp_group
)
tgt_log_chunks, tgt_sizes = chunk_average_log_probs(
    target_log_probs, teacher_chunk_id, max_chunks, cp_group=cp_group
)

# Student 侧重归一化
proj_chunks = proj_chunks / (proj_chunks.sum(dim=-1, keepdim=True) + eps)
proj_log_chunks = (proj_chunks + eps).log()

# KL 散度（forward 或 reverse）
if self.reverse_kl:
    per_chunk_kl = F.kl_div(
        tgt_log_chunks, proj_log_chunks, reduction="none", log_target=True
    ).sum(dim=-1)     # KL(student || teacher)
else:
    per_chunk_kl = F.kl_div(
        proj_log_chunks, tgt_log_chunks, reduction="none", log_target=True
    ).sum(dim=-1)     # KL(teacher || student) —— 默认 forward KL

# 全局归一化
kl_loss = (per_chunk_kl * valid).sum() / denom * (T * T)
```

---

## 3. 异构词表下 top-k 语义不一致问题的解答

### 3.1 核心问题

如果 student 和 teacher 各自独立在自己的词表上取 top-k，那么：
- Student 选出的 k 个 token ID 来自 student 词表（如 Llama tokenizer）
- Teacher 选出的 k 个 token ID 来自 teacher 词表（如 Qwen tokenizer）

这两个 k 维子空间完全是不同的语义空间，直接计算 KL 散度毫无意义。

### 3.2 NeMo-RL 的解决方案

NeMo-RL 通过以下三步设计彻底避免了这个问题：

```
                          ┌─────────────────────────────────────────────────────┐
                          │                   P-KL 流程图                        │
                          └─────────────────────────────────────────────────────┘

   Student logits [B,T_s,V_s]                  Teacher logits [B,T_t,V_t]
           │                                            │
           ▼                                            │
     log_softmax / T                                    │
           │                                            │
           ▼                                            │
       exp → prob                                       │
           │                                            │
           ▼                                            │
   ┌───────────────────┐                                │
   │  投影矩阵 M (稀疏)  │  离线预计算                    │
   │  [V_s, V_t]       │  student token → teacher token │
   └───────┬───────────┘  语义对应关系                   │
           │                                            │
           ▼                                            │
   projected [B,T_s,V_t]                                │
   (student 分布已在 teacher                             │
    语义空间中!)                                         │
           │                                            ▼
           │                   ┌─────────────────────────────────┐
           │                   │ select_teacher_topk_indices()   │
           │                   │                                 │
           │                   │  1. 展平 (B,T_t)                │
           │                   │  2. per-vocab max (重要性)      │
           │                   │  3. CP all_reduce(MAX)          │
           │                   │  4. topk(k) + sort              │
           │                   │                                 │
           │                   │  global_top_indices [k]         │
           │                   └───────────────┬─────────────────┘
           │                                   │
           │       双方切片到相同 k 列           │
           ├───────────────────────────────────┤
           ▼                                   ▼
   projected_topk [B,T_s,k]           teacher_topk [B,T_t,k]
           │                                   │
           ▼                                   ▼
   renormalize (subset softmax)      renormalize (subset softmax)
           │                                   │
           ▼                                   ▼
   ┌───────────────────────────────────────────────────────┐
   │              chunk 对齐 + KL(proj || teacher)          │
   │              per_chunk_kl.sum() / global_valid_chunks  │
   │              × T²                                     │
   └───────────────────────────────────────────────────────┘
```

### 3.3 三步设计详解

#### 第一步：投影矩阵统一语义空间

投影矩阵 M `[V_s, V_t]` 是在训练前离线预计算的，编码了 student token → teacher token 的语义对应关系。

**生成流程：**
```bash
# 1. 双向重 tokenize 生成原始映射
python -m tools.x_token.minimal_projection_via_multitoken \
    --student-model "meta-llama/Llama-3.2-1B" \
    --teacher-model "Qwen/Qwen3-4B" \
    --top-k 32 --enable-scale-trick --enable-reverse-pass

# 2. 精确映射重映射
python -m tools.x_token.reapply_exact_map ...

# 3. 排序裁剪到运行时 top_k
python -m tools.x_token.sort_and_cut_projection_matrix \
    --top_k 4 --output_path projection_matrix_top4.pt
```

例如，Llama tokenizer 的 `"hello"` 会被映射到 Qwen tokenizer 的 `["hello", "greet", "hi", ...]` 及其对应的权重。**Student 的分布经过 M 后已经完全进入了 teacher 的语义空间** (`[B, T_s, V_t]`)。

#### 第二步：只在 teacher 侧取一次 top-k

`select_teacher_topk_indices()` 仅对 teacher 的 logits 取 top-k，选出的 k 个 token ID 都是 **teacher 词表中的 token**。这是"权威方"选择——由教师模型决定在它的语义空间中哪些 token 是当前 batch 中最重要的。

#### 第三步：双方切片到完全相同的 k 维度

```python
projected_topk = projected_full[..., global_top_indices]   # Student 投影后的分布，切片
teacher_topk  = teacher_full_logits[..., global_top_indices] # Teacher 原始分布，切片
```

两组切片的 k 列是**完全相同的 k 个 teacher 词表 token ID**，因此：
- 双方在同一个 k 维语义子空间中
- 对应位置上的 token 含义完全一致
- KL 散度在语义上是良定义的

### 3.4 与 gold-loss 路径的对比

| 特性 | P-KL（`gold_loss=False`） | Gold-loss（`gold_loss=True`） |
|------|--------------------------|------------------------------|
| 词表划分 | 动态（每步选 top-k） | 静态（离线分 common/uncommon） |
| 选取依据 | 当前 batch teacher logit 重要性 | 离线精确 1:1 token 映射 |
| 投影矩阵用途 | 全量投影到 teacher 词表 | 仅用于构建 exact token map |
| 损耗函数 | KL on top-k subset | KL on common + L1 on uncommon |

两种路径的共同点：**都在统一的 teacher 词表语义空间中操作**，不存在各自取 top-k 导致的语义错位。

### 3.5 同词表 teacher 的处理

当 teacher 和学生共享 tokenizer 时（`_direct_topk_kl`，第 1838–1895 行），也遵循相同的单侧 top-k 逻辑：

```python
vocab_topk = min(self.vocab_topk, teacher.shape[-1])
topk_idx = select_teacher_topk_indices(teacher, vocab_topk, cp_group=cp_group)

# Student 先 gather 到完整词表（TP 场景），再切片
student_log_probs = vocab_parallel_full_log_softmax(student_logits, T, tp_group=tp_group)
student_log_probs_k = student_log_probs[..., topk_idx]
student_log_probs_k = student_gathered - torch.logsumexp(student_gathered, dim=-1, keepdim=True)

# Teacher 直接切片后 renormalize
teacher_log_probs_k = torch.log_softmax(teacher[..., topk_idx] / T, dim=-1)

# 计算 KL
per_pos = F.kl_div(student_log_probs_k, teacher_log_probs_k, reduction="none", log_target=True).sum(dim=-1)
```

---

## 4. 相关配置项

在 `CrossTokenizerDistillationLossConfig` 中，P-KL 相关的关键配置：

| 配置项 | 类型 | 说明 |
|--------|------|------|
| `gold_loss` | `bool` | `False` = P-KL 模式，`True` = gold-loss 模式 |
| `xtoken_loss` | `bool` | Gold-loss 下的 relaxed 修饰符，P-KL 模式下必须为 `False` |
| `vocab_topk` | `int` | P-KL 的 microbatch-global top-k 大小（gold_loss 模式下无效） |
| `temperature` | `float` | 对称应用于 student 和 teacher 的 softmax temperature |
| `reverse_kl` | `bool` | `True` = KL(student \|\| teacher)，`False` = KL(teacher \|\| student) |
| `exact_token_match_only` | `bool` | P-KL 下仅对精确对齐的 pair 计算 KL（过滤不匹配的 pair） |
| `dynamic_loss_scaling` | `bool` | 动态缩放 KD 项以匹配 CE 项的量级 |
| `kl_loss_weight` | `float` | 固定权重模式下 KD 项系数 |
| `ce_loss_scale` | `float` | 固定权重模式下 CE 项系数 |

YAML 示例片段：
```yaml
loss_fn:
  gold_loss: false
  xtoken_loss: false
  temperature: 1.0
  vocab_topk: 256
  reverse_kl: false
  exact_token_match_only: false
  dynamic_loss_scaling: true
```

---

## 5. 总结

NeMo-RL 的 P-KL 通过 **"投影矩阵统一语义空间 + teacher 单侧取 top-k + 双方切片到相同子集"** 的三步设计，彻底解决了异构词表下 top-k 可能导致的语义不一致问题：

1. **投影矩阵 M 离线预计算**，将 student 分布映射到 teacher 语义空间
2. **top-k 仅在 teacher 侧取一次**，由教师决定哪些 token 重要
3. **双方使用完全相同的 k 个 teacher token ID**，在同一语义子空间中做 KL

这使得跨分词器的知识蒸馏在数学上是严格良定义的。