# 流式采样收益测试报告

> 配套方案：`docs/designs/streaming_bench_plan.md`
> 配套脚本：`cookbook/rl/reward_loop/bench_streaming_local.py`
> 数据：`results/bench_timeline.jsonl`、`results/bench_summary.csv`

## 0. 当前状态

- 脚本与文档已完成，脚本纯逻辑部分（item_id 回填、奖励重叠聚合、语义检查、
  扫描矩阵）已通过桩模块验证（`ALL LOGIC CHECKS PASSED`）。
- **本机（开发沙箱）无 GPU / twinkle 环境，尚未实跑**。请在训练机
  （≥2 卡，已装 twinkle + vLLM）上按第 1 节运行，然后将结果回填第 3、4、5 节。
- 已知问题（已在脚本修复）：
  1. **DataLoader actor 重名崩溃**：twinkle 的 Ray actor 名含调用点行号，循环内
     同处新建多个 `DataLoader` 会报 `ActorAlreadyExistsError` —— 已为每个
     DataLoader 传唯一 `instance_id`；
  2. **结果只在全部跑完后写盘**：中途崩溃会留下空 `results/` —— 已改为每完成
     一个 run 即落盘，异常时也会先保存已完成的 run 再抛错；
  3. **输出目录依赖 CWD** —— 已改为默认输出到仓库根 `results/`（绝对路径，
     启动日志会打印实际路径）。

## 1. 运行方法

```bash
# 冒烟验证（极小配置，验证链路与正确性检查可用）
cd cookbook/rl/reward_loop
BENCH_RUNS=smoke python bench_streaming_local.py

# 完整测试（基础配置 + 单变量扫描，约 1-2 小时，需 ≥2 卡）
python bench_streaming_local.py
```

可用环境变量：`TWINKLE_MODEL_ID`（默认 `ms://Qwen/Qwen3.5-4B`，可传本地目录）、
`TWINKLE_DATASET_ID`（默认 `ms://modelscope/gsm8k`）、`TWINKLE_MODEL_GPUS` /
`TWINKLE_SAMPLER_GPUS`（默认各 1）、`TWINKLE_BATCH_SIZE`、`TWINKLE_MAX_STEPS`、
`TWINKLE_REWARD_DELAY_MS`、`TWINKLE_REWARD_NUM_WORKERS`、`BENCH_OUT_DIR`、
`BENCH_RUNS`（`all` 或 `smoke`）。

## 2. 测试环境

| 项 | 值 |
|---|---|
| 机器 / GPU | （待填充，如：单机 2×A100 80G） |
| 部署模式 | 本地 ray（无 twinkle-server） |
| 模型 | （待填充） |
| 数据集 | GSM8K train |
| 采样参数 | 见各 run 配置 |

## 3. 正确性验证结果

### 3.1 Level 1（确定性：greedy + 固定 seed）

| 检查项 | 结果 |
|---|---|
| 样本总数一致（A=B=batch×gen） | （待填充：通过 / 不通过） |
| tokens 逐位一致 | （待填充：完全一致 / 最大偏差 xx tokens / 差异计数） |
| logprobs 逐位一致 | （待填充：完全一致 / 最大绝对差） |
| decoded 文本一致 | （待填充） |
| reward 值逐样本一致 | （待填充） |
| 优势（GRPOAdvantage）一致 | （待填充） |

### 3.2 Level 2（语义：随机采样）

| 检查项 | A（批量） | B（流式） |
|---|---|---|
| 无丢样本 / 无重复 item_id | （待填充） | （待填充） |
| handle 结果与 item_id 对齐 | （待填充） | （待填充） |
| reward 函数确定性（重打分一致） | （待填充） | （待填充） |

## 4. 收益测量结果

### 4.1 基础配置（batch=4, gen=4, max_tokens=1024, delay=0）

| 指标 | A（批量） | B（流式） | 收益（B 相对 A） |
|---|---|---|---|
| t_sample（s） | （待填充） | （待填充） | （待填充） |
| t_submit（s） | （待填充） | （待填充） | （待填充） |
| t_collect（s，奖励尾部等待） | （待填充） | （待填充） | （待填充） |
| t_train（s） | （待填充） | （待填充） | （待填充） |
| **t_total（s/步）** | （待填充） | （待填充） | （待填充） |
| 吞吐（seq/s） | （待填充） | （待填充） | （待填充） |
| reward_head_start（s） | （待填充） | （待填充） | — |
| reward_tail_after_sample（s） | （待填充） | （待填充） | （待填充） |

### 4.2 单变量扫描

| run | 变量 | A t_total | B t_total | 收益 | 说明 |
|---|---|---|---|---|---|
| gen2 | gen=2 | （待填充） | （待填充） | （待填充） | |
| gen8 | gen=8 | （待填充） | （待填充） | （待填充） | |
| tok512 | max_tokens=512 | （待填充） | （待填充） | （待填充） | |
| tok2048 | max_tokens=2048 | （待填充） | （待填充） | （待填充） | |
| d200 | delay=200ms | （待填充） | （待填充） | （待填充） | |
| d1000 | delay=1000ms | （待填充） | （待填充） | （待填充） | |

## 5. 结论

（待填充：正确性是否通过；流式收益在哪些场景显著、哪些场景可忽略；
收益随 max_tokens / num_generations / 奖励耗时 的变化趋势；建议）

---

*报告由 `bench_streaming_local.py` 运行产物汇总生成。*
