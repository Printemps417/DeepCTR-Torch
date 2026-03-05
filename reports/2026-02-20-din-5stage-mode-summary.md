# DIN 五阶段模式与“出图前”操作详解

日期：2026-02-20  
仓库：DeepCTR-Torch  
相关脚本：
- `examples/eval_din_emblayer_4way.sh`
- `examples/benchmark_dinemblayer_infer.py`
- `examples/export_din_frozen_graph.py`
- `examples/onnx_emblayerseq_converter.py`
- `examples/onnx_emblayer_joint_converter.py`

---

## 1. 文档目的

本文用于回答“各个 stage 的模式，在出图前（以及运行前）到底做了哪些操作”，并把 Stage0~Stage5 的输入构造、算子路径、ONNX 变换、运行时开关一次讲清楚。

这份说明覆盖两层含义：

1. **图层（ONNX）侧**：每个 stage 的图是如何从 baseline 图转换得到的。  
2. **运行时（benchmark）侧**：每个 stage 在实际推理前准备了哪些输入，走了哪些算子路径。

---

## 2. 先看全局：Stage0~Stage5 是怎么串起来的

`eval_din_emblayer_4way.sh` 的执行顺序：

1. **Stage0（导出 + 转图）**
   - 导出 `stage1 baseline` 和 `stage2 multislice` 两个基础 ONNX。
   - 基于 converter 继续生成 `stage3/4/5` ONNX。
2. **Stage1~Stage5（profile + benchmark）**
   - 分别调用 `benchmark_dinemblayer_infer.py --mode xxx`。
   - 每个 stage 都做 nsys profile，并导出统计 CSV。

可理解为：

- Stage0 负责“准备图”。
- Stage1~5 负责“按不同输入/算子模式跑图并采集性能”。

---

## 3. Stage0：出图前统一准备（最关键）

### 3.1 自动计算 vec/seq 列范围

若未显式传 `SEQ_RANGES` / `VEC_RANGES`，脚本会：

- 用 `export_din_frozen_graph.py` 里的特征配置 + `DIN.feature_index` 自动推导：
  - `VEC_RANGES`：所有非序列特征（`sparse_*` + `score`）列区间。
  - `SEQ_RANGES`：所有序列特征（`hist_sparse_*`）列区间。

这一步决定了后续 converter 替换哪些 Slice 节点。

### 3.2 先导两张“基础图”

- `din_stage1_baseline.onnx`：原始 baseline 图。
- `din_stage2_multislice.onnx`：将向量类 Slice 合并为 MultiSlice 后的图（由导出脚本完成）。

### 3.3 再生成 stage3/4/5 图

- Stage3 图：对 baseline 图做 `onnx_emblayerseq_converter.py`，把 seq 区域 Slice 替换为 `EmblayerSeq`。
- Stage4 图：对 stage2（multislice）图再做 seq 替换，得到 `MultiSlice + EmblayerSeq`。
- Stage5 图：对 baseline 图做 `onnx_emblayer_joint_converter.py`，同时处理 vec + seq，得到 `EmblayerVec + EmblayerSeq`。

---

## 4. benchmark 侧的核心输入张量（所有 stage 的“原材料”）

`benchmark_dinemblayer_infer.py` 的 `build_inputs()` 会一次性构造多套输入视图，供不同 stage 复用：

1. `full`：完整 padded 输入（baseline 形态）。
2. `non_seq_full`：和 `full` 同宽，但仅保留非 seq 列（seq 列全 0）。
3. `non_seq_compact`：把非 seq 列紧凑拼接后的张量。
4. `seq_values/seq_prefix/seq_lengths/seq_offsets`：seq 压缩表示（单 iobuffer）。
5. `vec_values/vec_indices/vec_prefix`：vec 压缩表示（给 EmblayerVec 用）。
6. `compact_layout`：记录“紧凑列 <-> 原始列”映射关系。
7. `seq_feature_info`：每个 seq 特征的列区间与 maxlen 信息。

也就是说，**Stage 差异不是数据源不同，而是“在运行前选哪套视图 + 走哪条重建路径”不同**。

---

## 5. 五个 stage 的“出图前/运行前”具体动作

## Stage1：baseline（直接 padded）

- benchmark mode：`baseline`
- 输入准备：
  - 只搬运 `full` 到设备侧。
- 计算路径：
  - `DINBaselinePerInputSlice`：按原始输入 span 切分后再 `cat` 回模型输入。
- 语义：
  - 不启用自定义压缩重建算子，作为对照基线。

---

## Stage2：multislice（只替换 vec 路径）

- benchmark mode：`multislice`
- 输入准备：
  - 同样搬运 `full`。
- 计算路径：
  - `DINMultiSliceInfer`：
    1. 对 `full` 执行 `multislice` 抽取非 seq 紧凑块。
    2. 按 `compact_layout` 回填到完整 `x`。
    3. 送入 DIN。
- 语义：
  - 主要验证多 Slice 合并成一个 MultiSlice 对向量特征路径的影响。

---

## Stage3：emblayerSeq only（只压缩 seq）

- benchmark mode：`seq_only`
- 输入准备（emblayer 真实模式）：
  - `non_seq_compact`
  - `seq_values, seq_prefix, seq_lengths, seq_offsets`
- 计算路径：
  - `DINSeqOnlyInfer`：
    1. 先把 `non_seq_compact` 回填到完整 `x` 的非 seq 区域。
    2. 调 `emblayer_seq_fast` 还原 seq padded 结构。
    3. 按 `seq_assign_layout` 将 seq 数据铺回 `x`。
    4. 送入 DIN。
- 语义：
  - vec 侧不做 MultiSlice/Vec 压缩，只验证 seq 压缩 + GPU 重建。

---

## Stage4：multislice + emblayerSeq（vec+seq 各优化一半）

- benchmark mode：`multislice_seq`
- 输入准备（emblayer 真实模式）：
  - `non_seq_full`
  - `seq_values, seq_prefix, seq_lengths, seq_offsets`
- 计算路径：
  - `DINMultiSliceSeqInfer`：
    1. 对 `non_seq_full` 做 MultiSlice 得到非 seq 紧凑块并回填。
    2. 对 seq 压缩输入调用 `emblayer_seq_fast` 重建并回填。
    3. 送入 DIN。
- 语义：
  - 同时评估 vec 路径（MultiSlice）和 seq 路径（EmblayerSeq）的组合效果。

---

## Stage5：joint（emblayerSeq + emblayerVec）

- benchmark mode：`joint`
- 输入准备（emblayer 真实模式）：
  - vec 压缩：`vec_values, vec_prefix, vec_indices`
  - seq 压缩：`seq_values, seq_prefix, seq_lengths, seq_offsets`
- 计算路径：
  - `DINJointInfer`：
    1. `emblayer_vec_fast` 直接重建完整 `x` 的非 seq 部分。
    2. `emblayer_seq_fast` 重建 seq，并回填到 `x`。
    3. 送入 DIN。
- 语义：
  - vec + seq 全链路压缩重建的“最激进”方案。

---

## 6. 一个非常关键的运行时开关：scheduler vs emblayer

脚本支持两种 runtime 语义：

1. `BENCH_RUNTIME_MODE=emblayer`（默认）
   - 等价传参：`--rebuild_from_emblayer`
   - Stage3/4/5 走真实自定义算子路径（`emblayer_seq_fast` / `emblayer_vec_fast` 等）。

2. `BENCH_RUNTIME_MODE=scheduler`
   - 等价传参：`--scheduler_side_concat`
   - 在 benchmark 里，Stage3/4/5 会退回到 baseline raw 输入路径（直接吃 `full`），用于模拟“调度侧已拼好输入”的场景，而非测 Emblayer 内核本体。

因此解读图表时要先确认：当次结果到底是 `scheduler` 还是 `emblayer`。

---

## 7. 五阶段对照表（便于和图直接对应）

| Stage | benchmark mode | ONNX 来源 | 运行时主要输入 | 关键算子路径 | 目标 |
|---|---|---|---|---|---|
| 1 baseline | `baseline` | baseline 导出图 | `full` | 原始切分/拼接 + DIN | 基线 |
| 2 multislice | `multislice` | baseline -> multislice | `full` | `MultiSlice` + 回填 + DIN | vec Slice 合并 |
| 3 emblayerSeq only | `seq_only` | baseline -> seq converter | `non_seq_compact` + seq 压缩 | `EmblayerSeq` 重建 seq + DIN | seq 压缩收益 |
| 4 multislice+seq | `multislice_seq` | multislice 图 -> seq converter | `non_seq_full` + seq 压缩 | `MultiSlice` + `EmblayerSeq` + DIN | vec+seq 组合 |
| 5 joint | `joint` | baseline -> joint converter | vec 压缩 + seq 压缩 | `EmblayerVec` + `EmblayerSeq` + DIN | 全链路压缩 |

---

## 8. 常见误读与排查建议

### 8.1 为什么 stage3/4/5 的输入字节有时看起来没变？

若启用 `scheduler_side_concat`，脚本会统一按 `full` 输入统计，导致 stage3/4/5 的输入字节显示接近 baseline。此时测的是“调度侧拼接后”场景，不是 Emblayer 压缩输入场景。

### 8.2 为什么 ONNX 节点替换数量和预期不一致？

converter 的替换依赖 `SEQ_RANGES/VEC_RANGES` 与图中 Slice 形态匹配。若导出图结构变化，命中数量可能变化，需要先核对 feature_index 映射与 ranges。

### 8.3 为什么 H2D 提升但端到端不一定提升？

压缩输入常能降低拷贝量，但若重建 kernel 的 launch/访存开销较高，端到端收益会被抵消。要结合 nsys 的 kernel/API 汇总定位瓶颈。

---

## 9. 复现建议（最小命令）

```bash
cd /root/DeepCTR-Torch/examples
BENCH_RUNTIME_MODE=emblayer ITERS=2000 WARMUP=300 BATCH_SIZE=1024 \
MAX_SEQ_LEN=128 AVG_SEQ_LEN=8 bash eval_din_emblayer_4way.sh
```

若想看 scheduler 语义：

```bash
cd /root/DeepCTR-Torch/examples
BENCH_RUNTIME_MODE=scheduler ITERS=2000 WARMUP=300 BATCH_SIZE=1024 \
MAX_SEQ_LEN=128 AVG_SEQ_LEN=8 bash eval_din_emblayer_4way.sh
```

---

## 10. 结论（一句话版）

“出图前”核心是 Stage0 做统一导图与 converter 替换；“运行前”核心是根据 stage 选择 `full / non_seq_full / non_seq_compact / vec压缩 / seq压缩` 等输入视图，并决定是否启用 `MultiSlice`、`EmblayerSeq`、`EmblayerVec` 进行 GPU 侧重建。
