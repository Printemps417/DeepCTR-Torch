# DeepCTR-Torch 会话记录（DIN/DeepFM + MultiSlice + EmblayerSeq）

日期：2026-02-20  
仓库：DeepCTR-Torch（master）

## 背景与目标

本次主要目标：

1. 新增 DeepFM 示例脚本并解决环境兼容问题。  
2. 实现自定义 CUDA `MultiSlice` 算子并完成一次推理验证。  
3. 构建通用 ONNX converter，将 `Slice` 子图合并为 `MultiSlice`。  
4. 新增 DIN 导出链路（仅 sparse+dense 走 MultiSlice，seq 不处理）。  
5. 增加前后性能评估脚本（含 nsys before/after）。
6. 新增 `emblayerSeq` 算子（压缩 seq 输入 + GPU 内 padding）与 converter/benchmark。

---

## 关键变更文件

### DeepFM 相关

- `examples/run_deepfm.py`
- `examples/export_deepfm_frozen_graph.py`
- `examples/export_deepfm.sh`

### MultiSlice 算子与推理

- `examples/multislice_op/multislice.cpp`
- `examples/multislice_op/multislice_cuda.cu`
- `examples/run_deepfm_multislice_infer.py`

### ONNX converter 与 DIN 导出链路

- `examples/onnx_multislice_converter.py`
- `examples/export_din_frozen_graph.py`
- `examples/run_din_multislice_pipeline.py`
- `examples/run_din_multislice_pipeline.sh`

### 评估与性能分析

- `examples/eval_din_convert.sh`
- `examples/benchmark_din_multislice_infer.py`
- `examples/eval_din_nsys_before_after.sh`

### EmblayerSeq 算子与链路（新增）

- `examples/emblayer_op/emblayer_seq.cpp`
- `examples/emblayer_op/emblayer_seq_cuda.cu`
- `examples/run_emblayerseq_demo.py`
- `examples/onnx_emblayerseq_converter.py`
- `examples/benchmark_dinemblayerSeq_infer.py`
- `examples/eval_din_emblayerseq_nsys.sh`

### 报告与日志（新增）

- `reports/2026-02-20-din-emblayerseq-before-after-report.md`

---

## 主要实现说明

### 1) MultiSlice CUDA 算子

- 新增了 PyTorch C++/CUDA 扩展：输入二维张量 `[B, C]`，按多个 `(start, length)` 进行切列并拼接输出。
- 算子在示例中用于减少多次 `Slice`/切分访问，目标是优化数据路径与 kernel 结构。

### 2) 通用 ONNX Converter

- `onnx_multislice_converter.py` 支持扫描 ONNX 图中符合模式的 `Slice(axis=1, step=1, constant starts/ends)`。
- 将同源输入的多个 `Slice` 合并成自定义节点：`com.deepctr::MultiSlice`。
- 支持列范围过滤：
  - `include_col_start`
  - `include_col_end`
- 用于“只转换某些列块”，避免误动 seq 路径。

### 3) DIN 链路（按需求）

- `export_din_frozen_graph.py`：
  1. 导出 DIN 初始 ONNX；
  2. 调 converter；
  3. 仅转换 sparse+dense 列块；
  4. seq 相关 `Slice` 保留不变。
- `run_din_multislice_pipeline.py`：三步一键（导出 -> 转换 -> forward）。

### 4) Benchmark 与 nsys 对比

- `benchmark_din_multislice_infer.py` 现已改为：
  - batch 内 seq 不等长；
  - 图外先做 padding，再入图；
  - 并输出 batch 中 `seq_length` 的 unique 值确认数据分布。
- `eval_din_nsys_before_after.sh`：
  - baseline 与 multislice 各跑一次 nsys；
  - 导出 `.nsys-rep` 与 `stats.csv`。

### 5) EmblayerSeq（新增）

- 新增 `emblayerSeq` CUDA 自定义算子：输入 `iobuffer + prefix + seq_lengths`，在 GPU 内重建并 padding 为下游可消费格式。
- 算子 demo：`run_emblayerseq_demo.py`，已验证 CPU/GPU 结果一致。
- `onnx_emblayerseq_converter.py`：可把指定 seq 列范围的 `Slice` 替换为自定义 `EmblayerSeq` 节点。
- `benchmark_dinemblayerSeq_infer.py`：
  - 使用 batch 内不等长 seq；
  - baseline 传 padded 全量输入；
  - emblayer 路径传 compact non-seq + 单 iobuffer + prefix + lengths；
  - 提供端到端、H2D-only、输入字节量和数值一致性指标。
- `eval_din_emblayerseq_nsys.sh`：按现有风格实现 baseline vs emblayer 的 nsys 前后对比。

---

## 关键运行命令（可复现）

### A. DeepFM 导出

```bash
cd /root/DeepCTR-Torch/examples
bash export_deepfm.sh
```

### B. DIN 三步链路

```bash
cd /root/DeepCTR-Torch/examples
bash run_din_multislice_pipeline.sh
```

### C. DIN convert 前后评估

```bash
cd /root/DeepCTR-Torch/examples
bash eval_din_convert.sh
```

### D. nsys 前后性能对比

```bash
cd /root/DeepCTR-Torch/examples
ITERS=2000 WARMUP=300 BATCH_SIZE=1024 bash eval_din_nsys_before_after.sh
```

### E. EmblayerSeq benchmark

```bash
cd /root/DeepCTR-Torch/examples
/root/autodl-tmp/myenv/bin/python benchmark_dinemblayerSeq_infer.py \
  --mode both \
  --iters 500 \
  --warmup 100 \
  --batch_size 1024 \
  --max_seq_len 128 \
  --avg_seq_len 8 \
  --include_h2d
```

### F. EmblayerSeq nsys 对比

```bash
cd /root/DeepCTR-Torch/examples
ITERS=2000 WARMUP=300 BATCH_SIZE=1024 MAX_SEQ_LEN=128 AVG_SEQ_LEN=8 bash eval_din_emblayerseq_nsys.sh
```

---

## 已验证结果摘要

- DeepFM ONNX/TorchScript frozen 文件导出成功。  
- `MultiSlice` CUDA 扩展编译成功，DeepFM 与 DIN 示例均完成前向推理。  
- DIN converter 转换后：
  - sparse+dense 路径的 `Slice` 被合并为 `MultiSlice`；
  - seq 路径保留（符合需求）。
- nsys before/after 脚本可生成完整 profiling 产物。
- `emblayerSeq` 算子已编译通过并 demo 验证一致。
- `emblayerSeq` benchmark 已验证：
  - `input_bytes_reduction_ratio ≈ 0.903`（输入字节显著下降）；
  - `max|baseline-emblayer| = 0.0`（数值一致）；
  - `h2d speedup` 在样例中可达约 `1.46x`（具体值随 batch/seq 分布波动）。

---

## 备注

- 若要运行“转换后 ONNX”推理，需要目标运行时支持自定义 `MultiSlice`（例如 ORT custom op 或 TensorRT plugin）。
- 当前仓库脚本侧重点是：模型图转换、可视化、以及对比分析链路。
- EmblayerSeq 在“传输压缩”维度收益明确，但端到端收益仍依赖后续 kernel 融合与图内算子接线优化。