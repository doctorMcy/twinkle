# NeMo X-Token 开源实现与 twinkle 实现对比分析

> 参考文件:
>
> - NeMo 主脚本:`NeMo/nemo_rl/algorithms/xtoken_off_policy_distillation.py`
> - NeMo 损失:`NeMo/nemo_rl/algorithms/loss/loss_functions.py`(`CrossTokenizerDistillationLossFn`)
> - NeMo 对齐/规范化:`NeMo/nemo_rl/algorithms/x_token/token_aligner.py`、`x_token/loss_utils.py`
> - NeMo 投影矩阵生成:`NeMo/tools/x_token/`(三阶段 CLI)
> - twinkle 损失:`src/twinkle/loss/cross_token.py`(`CrossTokenLoss`)
> - twinkle 训练脚本:`cookbook/rl/mopd/cross_token_trainer_on_policy.py`

## 0. 背景:本会话在 twinkle 上定位到的五个问题

在 twinkle 上做验证(同模型对照 + 跨模型 Qwen3 0.6B/1.7B on-policy 训练)时,
发现并修复了五个问题,本文以 NeMo 开源实现为参照,回答"这些问题是 twinkle
独有还是 NeMo 也有?NeMo 是怎么处理的":

| 编号 | 问题 | 现象(实测) | 状态 |
|---|---|---|---|
| A | 投影矩阵按 strip 后文本匹配,空格变体坍缩("的" vs " 的") | 同模型:90/186 位置 KL 触底 23.0259,kd=7.13 | 已修复(exact-first) |
| 2 | 教师 padding token 泄漏进 KD 有效掩码 | 同模型:valid=170 > num_tokens=128 | 已修复(标签掩码) |
| B | 教师侧 log 空间 chunk 几何均值压扁分布(学生侧恒单位置,不对称) | 同模型:1 个 t_n=2 位置 KL=22.78 | 未修,影响小 |
| C | 模板结构位置主导 loss(跨模型 on-policy) | 跨模型:top-20 异常 ~95% 为 \n/<\|im_start\|>/user/assistant;KD↔CE 互斥;total 平台 ~5 | 已修复(只蒸 response,labels 掩码) |
| D | 同文本多 id 坍缩(字节回退 token,多个 id 均解码为 U+FFFD) | 跨模型:每步 1~3 行,双侧一致仍 KL=23(238(�)→151560(�)) | 已修复(同 id 优先) |
| — | KL 方向疑问(reverse_kl 语义) | — | 结论:与 NeMo 一致,无需改 |

---

## 1. 总体架构对比

| | NeMo | twinkle |
|---|---|---|
| 三段式分工 | **CPU 侧 Collator+Aligner 完成全部跨词表工作**(投影矩阵加载、token 对齐),loss 只做损失数学 | 对齐(chunk 构建)与投影矩阵构建都在 **loss 内部**运行时完成 |
| 教师 logits 传输 | CUDA IPC 直传(不走 driver,常驻缓冲跨步复用) | Ray 回传 driver(每步拷贝) |
| 投影矩阵 | **离线生成的文件**(`projection_matrix_path`,`tools/x_token/` 三阶段 CLI:embedding 投影 + sinkhorn + top-k 裁剪 + exact-map 重映射) | 运行时按**文本匹配**构建(精确 weight 1.0 / 多 token β·γ^i) |
| 对齐算法 | **Needleman-Wunsch DP**(`token_aligner.py` ~1300 行,锚点加速,处理乱码字节/unicode 伪影/多 token 伪影) | **字符跨度重叠启发式**(`_build_chunk_ids`,~60 行) |
| 文本规范化 | **`canonical_token`**(空格前缀 ▁/Ġ/空格→Ġ,换行变体→\n,字节回退等) | 原实现 `.strip()`(已修复为 exact-first + strip 兜底) |

```
NeMo 数据流:
  数据集 → CrossTokenizerCollator(对齐+投影,DataLoader worker 常驻)
         → 教师前向(全词表 logits 经 CUDA IPC 直传,不经过 driver)
         → student.train(损失数学,TP/CP 感知)
         → 优化

twinkle 数据流:
  数据集 → 学生 vLLM 采样(on-policy)
         → decode → 教师 tokenizer 重编码
         → 教师前向(Ray 回传 logits 到 driver)
         → student.forward_backward(loss 内部:投影构建 + 字符跨度对齐 + KL)
         → 优化
```

---

## 2. 逐机制对比(含示例)

### 2.1 同词表路径 —— 最大的结构性差异

**机制**:学生与教师共享词表(相同 tokenizer,或不同模型共用同一分词器,如
Llama-3.2-3B 与 1B)时,教师前向**不需要重新编码**,位置天然一一对应。

**NeMo**:`projection_matrix_path=None` 即标记"同词表教师"——教师前向**直接复用
学生的 input_ids**(`export_teacher_logits_and_pack`),loss 走 `_direct_topk_kl`:
逐位置 top-k KL,**不投影、不对齐、不 chunk 平均**;并在 setup 时断言两侧词表
大小一致。

**twinkle**:**没有**同词表路径——任何情况都 decode→re-encode + 投影 + 对齐。

**示例**(真实日志中的重编码错位):

```
学生原始 token 序列(同词表,id 直接来自 vLLM 采样):
  [151644, 872, 198, 9370, ...]
    │       │    │    └─ decode → "的"
    │       │    └─ decode → "\n"
    │       └─ decode → "user"
    └─ <|im_start|>

twinkle 路径:decode 成文本再让教师重新 encode,同一文本的切分可能不同:
  [151644, 19060, 98878, 9370, ...]      ← 边界变了!
             ^^^^    ^^^^^
             " user"  空白族 token(8 个空格)
  872("user") 被重切成 19060(" user"),198("\n") 被并进空白族 token
  → 位置错位 → 需要字符跨度对齐 → 出现 t_n=2 等偶发异常

NeMo 同词表路径:教师直接用 [151644, 872, 198, 9370, ...] 前向
  → 位置一一对应,KL 逐位置直接算,零误差
```

**与本次会话的关系**:同模型验证中唯一的残留异常(`大` 行 t_n=2、KL=22.78)
正是重编码边界错位造成的——NeMo 的同词表路径从机制上绕开了它。

---

### 2.2 投影矩阵构建与文本坍缩(对应问题 A)

**机制**:跨词表时,学生 token 需要映射到教师词表才能比较分布。映射的质量决定
KL 是否失真。

**NeMo**:投影矩阵是**离线生成的文件**,主路径用**embedding 相似度**(token
embedding 投影 + sinkhorn 归一化,每行保留 top-4),精确文本匹配仅作为辅助
(`reapply_exact_map.py` 重映射);匹配前的文本统一走 `canonical_token` 规范化。

**twinkle**:运行时按文本匹配构建:精确匹配(weight 1.0)+ 多 token 编码
(β·γ^i)。原实现两侧都 `.strip()` 后匹配。

**示例**(问题 A 的完整链条,真实数据):

```
学生 token 9370 decode = "的";教师词表中 43589 decode = " 的"(带前导空格)

旧 twinkle 实现(.strip() 后匹配):
  "的".strip() == " 的".strip() == "的"   → 9370 被映射到 43589(weight 1.0)
  学生对自己生成的 token 9370("的")给出概率 1.0000
  投影后质量全部落到 43589(" 的")
  教师(同模型!)在同一位置对 43589 的概率 ≈ 0(它预测的是 9370)
  → KL = 1.0 × ln(1.0 / 1e-10) ≈ 23.0259 —— 触底
  实测:90/186 个位置全部如此,kd=7.13(同模型本应为 0)

NeMo canonical 化(不删除空格前缀,只归一化写法):
  canonical("的") = "的"      canonical(" 的") = "Ġ的"
  canonical("▁的") = "Ġ的"    canonical("Ġ的") = "Ġ的"
  两个 canonical 不同 → 各自独立映射 → 不坍缩
  (顺带把 SentencePiece 的 ▁ 与 BPE 的 Ġ/空格统一,跨分词器家族也能对上)

twinkle 修复后(exact-first + strip 兜底):
  9370 原始文本 "的" → 教师词表精确匹配到 "的"@9370 → 9370→9370 ✓
  43589(" 的") → " 的"@43589 ✓
  同词表下退化为恒等映射 → 同模型 KL ≈ 0(实测 kd=0.0000)
```

**结论**:NeMo 机制上不存在坍缩(canonical 保留空格前缀 + embedding 投影);
twinkle 已修复,若要更彻底可引入 canonical 归一化(见第 4 章)。

**残余变体:同文本多 id 坍缩(问题 D,已用"同 id 优先"修复)**

字节级 BPE 词表里存在大量 byte 回退 token(如 `<0xEF>`),单个字节单独解码时
大多渲染为替换字符 `�`——**几十上百个不同 id 的解码文本相同**。投影构建的
`text → id` dict 是"后者覆盖前者",所有解码为 `�` 的学生 token 都被映射到
最后一个同文 id(如 151560):

```
学生 id 238(�)、251(�)、233(�)… 全部 → 151560(�)   ← 坍缩成一个目标
```

实测(跨模型 Step 100,同词表):

```
23.0227 | b=0,pos=94 | 0.9585 → 0.9999 → 0.9996 → 0.9996 | 238(�) → 151560(�) [精确]
学生 p(238)=0.9585、教师 q(238)=0.9996 —— 两侧完全一致,KL 却触底 23.02
(学生质量被投影送到 151560,教师对 151560 概率≈0)
```

**修复**:匹配优先级改为 **同 id 优先**(学生 id < 教师词表且同 id 解码文本
相同 → 恒等映射 238→238)→ 上例 KL 23.02 → ~0.002;跨词表下同 id 解码文本
几乎必然不同,自然回落 exact-text,无回归。与问题 A 同属"基于文本的映射无法
区分解码文本相同的不同 id",彻底解法仍是 NeMo 的同词表直连路径(不投影)。

---

### 2.3 对齐与 padding 泄漏(对应问题 2)

**机制**:重编码后两侧 token 数量/边界不同,需要把教师位置映射到学生位置
(chunk/配对),映射时不能把填充位置当成有效内容。

**NeMo**:对齐是 **pair-based**(学生 token ↔ 教师 token 配对,DP 求解),对齐前
`_drop_padding` 剔除填充 token;有效性由数据级的 `pair_valid` 门控。

**twinkle**:字符跨度重叠启发式——教师位置取与其字符区间重叠最大的学生位置
作为 chunk;在**含 padding 的完整序列**上直接算跨度。

**示例**(padding 泄漏机制):

```
学生序列长 5,教师重编码后长 4,教师序列第 5 位用 id 0 填充:
  学生 spans: [t1][t2][t3][t4][t5]  ← 字符区间
  教师 spans: [t1][t2][t3][t4] [id=0 填充位] ← 填充 token 的字符区间
                                              延伸到文本末尾之后
  Qwen3 的 id 0 decode 为 "<|endoftext|>"(真实字符串,非空)

twinkle 的跨度对齐:教师填充 token 的跨度与学生的 padding chunk 重叠
  → 被判为"有效 chunk" → 计入 KD 均值
  实测现象:valid=170 而 num_tokens=128(多出 42 个填充位置),
  且该步 top-KL 的 20 个位置全部落在标签 -100 的填充 chunk 上(日志空行)

NeMo:对齐前剔除 padding,填充永远不会成为配对
  → 结构上不存在泄漏
```

**结论**:NeMo 结构上不存在该问题;twinkle 已用"chunk_valid 叠加标签掩码"修复
(修复后 valid ≤ num_tokens)。

---

### 2.4 chunk 平均:几何均值与对称性(对应问题 B)

**机制**:一个 chunk 内含多个位置时,把分布平均成 chunk 级分布。两边实现用的是
**log 空间平均**(`chunk_average_log_probs`,bmm 对 log 概率取均值 = 概率的
**几何均值**)——对"任一侧概率≈0"的 token 极其敏感,会把分布压扁。

**NeMo**:同样的几何均值,**但两侧都平均**(`proj_chunks` 与 `tgt_log_chunks`
都用学生/教师各自的 chunk id 平均);代码注释明确知道这是
"avg of log_softmaxes; not a true log of mean"。

**twinkle**:学生侧 chunk = 单个位置(恒等,不平均),教师侧可多位置 → **不对称**。

**示例**(根因 B 的完整链条,真实数据):

```
一个学生 chunk 里挤进 2 个教师位置 t1、t2(重编码边界不同):
  token:        A(大)       B(了)       C(很)
  t1 的分布:    1e-15       0.9         1e-15
  t2 的分布:    1e-15       1e-15       0.9

twinkle 教师侧几何均值 √(q1·q2):
  B: √(0.9 × 1e-15) ≈ 3e-8     ← 0.9 的质量被"蒸发"
  C: √(1e-15 × 0.9) ≈ 3e-8     ← 同样被蒸发
  → 教师分布被压成 ~1e-8 的"均匀摊开"(top-k 重归一化后每条 ≈0.016)
  学生侧:chunk 只有 1 个位置 → 保持 0.65 集中
  → KL(教师‖学生) ≈ 22.8 —— 实测正是 22.78(p_k=0.6504, q_k=2.46e-05)

NeMo:若 chunk 含 2 学生 + 2 教师位置,两侧都被同等压扁
  → 分布形态一致 → KL 温和(不再是"集中 vs 扁平")
  且同词表场景根本不走 chunk 路径
```

**结论**:几何均值本身是 NeMo 的设计(非 twinkle 引入),但 NeMo 因两侧对称而
影响小;twinkle 的不对称(学生恒等 vs 教师几何均值)导致 t_n>1 时偶发大 KL。
同词表场景 NeMo 直接绕开该机制。

---

### 2.5 KL 方向 —— 两项目完全一致

**机制**:KL(p‖q) 与 KL(q‖p) 数值和优化目标都不同,需要确认两侧实现用的是
哪个方向。

**NeMo**:默认(forward)= `teacher_probs·(ln teacher − ln student)` =
**KL(教师‖学生)**;`reverse_kl=True` = KL(学生‖教师)。

**twinkle**:`reverse_kl=False` 分支 =
`F.kl_div(input=log_projected_k, target=log_teacher_k, log_target=True)`,
按 `target.exp()·(target − input)` 展开 = 同样的 **KL(教师‖学生)**。

**示例**:

```
同一位置:学生 p(大)=0.65,教师 q(大)=0.9
  KL(教师‖学生) = Σ q·ln(q/p) ≈ 0.9×ln(0.9/0.65) ≈ 0.29
  KL(学生‖教师) = Σ p·ln(p/q) ≈ 0.65×ln(0.65/0.9) ≈ -0.21(加其余项后为正)

NeMo 默认与 twinkle reverse_kl=False 都算前者 → 完全一致
```

**结论**:twinkle 忠实复刻 NeMo 的约定(含 `reverse_kl` 命名),**无需改动**。
本会话早期提出的"方向疑问"撤回。

---

### 2.6 top-k 子集选取 —— 同构

**机制**:词表过大(15 万+),KL 只在"重要"的 k 个 token 子集上算;子集必须
batch 全局一致,chunk 平均后的 KL 才良定义。

**NeMo** 与 **twinkle** 实现逐行同构:

```
importance = 教师 logits 展平 (B×T) 后按词表取 max     # [V_t]
topk_idx   = topk(importance, k) 后升序排序             # [k]
两侧分布都切片到 [k] 列,再在子集内重归一化
```

**示例**:

```
batch 内教师 logits 中:token A 的最大 logit 0.9、B 0.7、其余 ≤ 0.1
  → importance(A)=0.9, importance(B)=0.7, ...
  → k=2 时 topk_idx = {A, B}
  → 学生投影分布与教师分布都只在 {A, B} 上重归一化后算 KL
```

**差异点仅工程面**:NeMo 在 CP>1 时对 importance 做 all-reduce(MAX) 保证各
rank 子集一致;twinkle 单 rank,无此需求。

---

### 2.7 top-k 外质量的处理 —— 一致(附注 NeMo 普通蒸馏的选项)

**机制**:重归一化会丢失"子集外质量"的信息(学生 90% 的质量在 top-64 外时,
子集内重归一化后分布被扭曲)。

- **NeMo 的 xtoken 路径** 与 **twinkle**:都在子集内重归一化,不补偿;
- 附注:NeMo **普通蒸馏** loss(`DistillationLossFn`)有一个 `zero_outside_topk`
  选项,用熵修正项 `H_rest − (−100)·P_rest` 把子集外质量补回来——但该选项
  **不在 xtoken 的 P-KL 路径中**,不参与本对比。

**结论**:xtoken 语义下两者一致。

---

### 2.8 掩码与归约

**NeMo**:有效位置 = `pair_valid`(数据级)∩ 两侧 chunk 非空 ∩ `sample_mask`;
归约按 **DP 全局有效数** 归一(跨 rank all-reduce 统计,每个有效位置等权)。

**twinkle**:有效 = `chunk_valid`(修复后含标签≠-100)∩ 学生位置;归约为
本地掩码均值。

**示例**(DP>1 时的归约差异):

```
DP=2:rank0 有 100 个有效位置,rank1 有 300 个
  NeMo:分母 = 400,每个位置等权
  twinkle 若按 rank 各自算均值再平均:(sum0/100 + sum1/300)/2 —— 位置不等权
  (twinkle 当前 loss 在单 rank 上计算,实际无差异;DP>1 时需注意)
```

**结论**:DP=1 时等价;NeMo 的全局归一在分布式下更规范。

---

### 2.9 温度与词表裁剪 —— 一致

**机制**:两侧都在 softmax 前除 T,损失最终乘 T²。

**示例**:

```
同一 logit 差 5:T=1.0 时概率比 e^5 ≈ 148 倍;T=0.8 时 e^(5/0.8) ≈ 518 倍(更尖锐)
```

另:HF lm_head 常把 out_features 填充到对齐宽度(如 Qwen3:tokenizer 151669,
lm_head 151936),两侧实现都先裁剪到 `len(tokenizer)` 再算——一致。

---

### 2.10 工程面差异(不影响算法语义)

| 项目 | NeMo | twinkle |
|---|---|---|
| 教师 logits 传输 | CUDA IPC 直传,常驻缓冲复用 | Ray 回传 driver,每步拷贝 |
| 对齐/投影执行位置 | DataLoader worker(常驻,缓存复用) | loss 内部,每步重算 chunk |
| 对齐器规模 | ~1300 行(乱码字节合并、unicode 修复、多 token 伪影、锚点加速) | ~60 行字符跨度启发式 |
| 多教师支持 | 原生(per-teacher 配置/权重/标志) | 支持(teacher_weights 聚合) |

### 2.11 位置掩码:只蒸 response(on-policy 适配,对应问题 C)

**机制**:on-policy 蒸馏中,序列 = 模板 prompt + 学生生成 response;哪些位置
参与 loss 对收敛行为影响巨大。

**NeMo xtoken 脚本**:off-policy 文本对,无 prompt/response 概念,整段文本
都是蒸馏对象;有效性 = `pair_valid`(对齐有效 + 非填充)∩ `token_mask` ∩
`sample_mask`。若用 chat 模板数据跑,模板位置同样会参与(无专门的模板掩码
机制)。

**twinkle 框架约定**:采样器 `vllm_sampler.py:250` 在标准路径就把 prompt 的
labels 置 -100(`feat['labels'] = [-100] * len(prompt_token_ids)`),response
部分由 `concat_input_feature` 补成 token id——即框架的"prompt + 生成"约定
就是**只对 response 算 loss**(GRPO/PPO 等 RL 算法依赖此约定)。

**twinkle on-policy 训练器(修复前)**:从 input_ids 重建"全有效" labels →
偏离框架约定 → 跨模型训练时模板结构位置(\n、<|im_start|>、user/assistant)
主导 KD 与 CE:

- top-20 异常位置 ~95% 是模板 token,对每个样本/每步恒定,占 KD 均值大头;
- **KD 与 CE 在结构位置互斥**:KD(KL(教师‖学生))把 p(\n) 从 1.0 压向教师
  的 ≈0,而 CE 要求 p(自身强制 token) 高 → KD 降多少 CE 补多少;
- 实测(50 步):kd 2.41→1.23、CE 6.32→3.75、total 8.73→4.98——在下降但
  平台高、速度慢,观感"不收敛"。

**修复(已实施)**:构造 `student_labels` 时把 prompt 部分置 -100(边界 =
`resp.prompt_token_ids` 长度);得益于 chunk_valid 已叠加标签掩码(2.3),
**CE 与 KD 同时只覆盖 response 部分**;`student_ids`(chunk 对齐用)保持完整
序列不变。

```
[<|im_start|>][user][\n][你是...][<|im_end|>][\n][<|im_start|>][assistant][\n][response...]
  └───────────── prompt(置 -100,不参与 CE/KD)──────────────┘└── response(参与)──┘
```

**与 NeMo 的一致性(三层)**:

| 层面 | 是否一致 |
|---|---|
| twinkle 框架约定(采样器 labels 语义) | ✅ 完全一致,是"回归约定" |
| NeMo xtoken 原则(只算有效/对齐良好的位置) | ✅ 原则一致,机制不同(pair_valid vs labels) |
| RL 社区惯例(只对 completion/response 算 loss) | ✅ 符合标准做法 |

---

## 3. 关键结论

| 本会话问题 | NeMo 是否有 | NeMo 的处理 | twinkle 现状 |
|---|---|---|---|
| A:strip 文本坍缩 | 无 | canonical 化保留空格前缀 + embedding 投影 | 已修复(exact-first);可进一步引入 canonical |
| 2:padding 泄漏 | 无 | 对齐前 drop padding + pair_valid 门控 | 已修复(标签掩码) |
| B:几何均值压扁 | 有(同函数) | 两侧对称平均,影响小;同词表不走 chunk | 不对称,偶发大 KL(未修,影响极小) |
| C:模板结构位置主导 loss | 无(off-policy 文本对无 prompt/response 结构) | 有效性靠 pair_valid 门控(只算有效位置) | 已修复(只蒸 response,labels 掩码) |
| D:同文本多 id 坍缩(字节 token) | 无(投影为 embedding 法,非文本匹配) | — | 已修复(同 id 优先) |
| KL 方向 | forward = KL(教师‖学生) | 默认 forward | 一致,无需改 |

## 4. 可落地的改进建议(按收益排序)

1. **加"同词表快速路径"**(收益最大,改动最小):trainer 检测学生/教师词表一致
   时,教师直接吃学生 input_ids、loss 走逐位置 top-k KL(不重编码、不投影、
   不对齐)。同模型验证可直接消除根因 B 与重编码错位;若正式训练也同词表,
   效果同样显著。实现参照 NeMo 的 `projection_matrix_path=None` 分支。
2. **引入 `canonical_token` 归一化**(~60 行)替换"exact-first + strip 兜底":
   跨分词器家族统一 ▁/Ġ/空格与换行变体,同时保留空格前缀不坍缩,比现有修复
   更完整。
3. **对齐算法升级为 pair-based DP**:从根上解决 t_n>1 与边界错位(padding
   泄漏也随之消失);若不想大改,退一步可让学生侧 chunk 也做对称平均。
4. **投影矩阵离线化**:可复刻 NeMo `tools/x_token/` CLI 生成 embedding 法矩阵
   文件,twinkle 的 loss 加载文件(与现有 sparse COO 结构兼容)。
5. **(已实施)只蒸 response**:构造 `student_labels` 时把 prompt 部分置 -100
   (边界取 `resp.prompt_token_ids` 长度),CE 与 KD 同时只算 response 部分;
   与 twinkle 框架约定(`vllm_sampler.py` 对 prompt 置 -100)和 RL 惯例一致,
   消除模板结构位置对 loss 的主导与 KD/CE 互斥。
6. **(已实施)同 id 优先**:投影构建时学生 id 在教师词表内且同 id 解码文本
   相同 → 恒等映射,消除"同文本多 id"坍缩(字节回退 token `�`);跨词表
   无回归。

---

## 5. 待办项与验证记录(截至 2026-08-04)

| 项 | 说明 | 状态 |
|---|---|---|
| VOCAB_TOPK 64→512 | 修复"batch 全局 top-k 子集排除局部主导 token"导致的 KL 伪影(双侧一致仍 KL=2~8)。实测:cov 0.51→0.94,伪影行消失,kd 在 33→100 步间 0.86→0.46 | ✅ 已验证生效(环境变量) |
| 同 id 优先(问题 D) | 字节回退 token `�` 的"同文本多 id"坍缩(238→151560,双侧一致仍 KL=23)→ 同 id 恒等映射 | ✅ 已实施,待重跑验证 |
| CE_LOSS_WEIGHT 下调 | CE 平台 ~1.5(KD 把学生分布拉离自身 token 的稳定张力);若想 total 更低可设 0.3~0.5 | ⏳ 待定 |
| 采样配置核对 | 响应出现 `�` 乱码与 `<think></think>`+中置 `<|im_end|>` 结构——疑似 vLLM stop tokens / `include_stop_str_in_output` / enable_thinking 与模型行为交互问题,属生成质量,独立于 loss | ⏳ 待查 |
| 同词表快速路径 | NeMo 直连方案:词表一致时教师直接吃学生 input_ids + 逐位置 KL(不重编码/不投影/不对齐),从根上消除 B 与所有文本映射边角 | ⏳ 远期 |
| canonical 归一化 | 用 NeMo `canonical_token` 风格归一化替换"exact-first + strip 兜底"(跨分词器家族统一 ▁/Ġ/空格) | ⏳ 远期 |
| pair-based DP 对齐 | 替换字符跨度启发式,解决 t_n>1 与边界错位(NeMo 同款) | ⏳ 远期 |
| 投影矩阵离线化 | 复刻 NeMo `tools/x_token/` embedding 法 CLI,loss 加载文件 | ⏳ 远期 |
