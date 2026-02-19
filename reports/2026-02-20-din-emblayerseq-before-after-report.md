# DIN EmblayerSeq 前后对比报告（更新）

日期：2026-02-20  
仓库：DeepCTR-Torch  
测试目标：对比 DIN baseline（直接传 padded seq）与 EmblayerSeq（压缩 seq + GPU 侧 padding）

## 1) 当前实现范围

- 算子：`examples/emblayer_op/emblayer_seq.cpp` + `examples/emblayer_op/emblayer_seq_cuda.cu`。
- converter：`examples/onnx_emblayerseq_converter.py`。
- benchmark：`examples/benchmark_dinemblayerSeq_infer.py`。
- nsys 对比脚本：`examples/eval_din_emblayerseq_nsys.sh`。

当前 benchmark 使用**单一 iobuffer + 一套 prefix/lengths**承载多个 seq 特征（更贴近 `emblayerSeq` 设计目标），并保证与 baseline 输出一致。

## 2) Benchmark 配置与结果（最新）

测试命令：

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

环境与数据：

- device：`cuda:0`
- 序列分布：`avg=8.1924, min=1, max=18`

指标：

| 指标 | Baseline | EmblayerSeq | 结论 |
|---|---:|---:|---|
| 每 batch 输入字节数 | 1,073,152 | 103,980 | 下降 90.31% |
| 端到端平均延迟 (ms) | 1.592786 | 1.743220 | 当前实现慢 9.43% |
| H2D-only 平均耗时 (ms) | 0.063319 | 0.043368 | 提升 1.46x |
| 数值一致性 max\|diff\| | - | 0.0 | 完全一致 |

补充比值：

- `input_bytes_reduction_ratio = 0.903108`
- `speedup (baseline/emblayer) = 0.913703`
- `h2d speedup (baseline/emblayer) = 1.460018`

## 3) ONNX Converter 结果（EmblayerSeq）

命令：

```bash
cd /root/DeepCTR-Torch/examples
/root/autodl-tmp/myenv/bin/python onnx_emblayerseq_converter.py \
  --input ./din_report_before.onnx \
  --output ./din_report_emblayerseq.onnx \
  --seq_ranges 5:9,9:13
```

结果：

- `removed Slice nodes: 1`
- `inserted EmblayerSeq nodes: 1`

说明：当前 converter 是按指定 `seq_ranges` 做模式替换，命中数取决于导出图中 Slice 子图形态。

## 4) nsys 前后对比脚本（新增）

脚本：`examples/eval_din_emblayerseq_nsys.sh`

用途：

1. baseline 路径做一次 nsys profile；
2. emblayerSeq 路径做一次 nsys profile；
3. 导出两份 `cuda_gpu_kern_sum/cuda_api_sum` csv。

示例命令：

```bash
cd /root/DeepCTR-Torch/examples
ITERS=2000 WARMUP=300 BATCH_SIZE=1024 MAX_SEQ_LEN=128 AVG_SEQ_LEN=8 bash eval_din_emblayerseq_nsys.sh
```

短测（`ITERS=50 WARMUP=10 BATCH_SIZE=128`）已验证脚本可正常生成：

- `examples/nsys_din_emblayerseq_before_after/din_baseline.nsys-rep`
- `examples/nsys_din_emblayerseq_before_after/din_emblayerseq.nsys-rep`
- 对应两份 `*.stats.csv`

## 5) 结论

- **优势已明确**：在变长 seq 场景下，EmblayerSeq 显著降低输入字节量，并改善 H2D 传输时间。  
- **端到端仍需优化**：当前计算侧仍有额外开销（重建写回、kernel 组织、访存模式），可能抵消部分传输收益。  
- **正确性通过**：baseline 与 EmblayerSeq 输出一致（`max|diff|=0`）。

## 6) 后续优化建议

1. 将 `emblayerSeq` 输出直接接入 embedding lookup，减少中间重建张量写回。  
2. 尝试融合“解压 + embedding”单 kernel，降低 launch 与访存往返。  
3. 扩展 converter 的匹配规则，提升 EmblayerSeq 替换命中率。  
4. 基于 `eval_din_emblayerseq_nsys.sh` 增加自动对比汇总（top kernel / top api / 占比差异）。
