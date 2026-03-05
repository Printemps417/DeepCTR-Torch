# DIN 六种方式对比报告

本报告由 `examples/eval_din_emblayer_4way.sh` 自动生成。

## 总览

| Stage | Avg Latency (ms) | H2D-only (ms) | Pure H2D (ms) | ONNX Nodes | Avg Latency相较于baseline的提升 | H2D-only相较于baseline的提升 | Pure H2D相较于baseline的提升 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1) baseline（直接输入 padded） | 48.434916 | 0.056327 | 0.017678 | 952 | 0.00% | 0.00% | 0.00% |
| 2) 只用 multislice | 46.921165 | 0.037188 | 0.017024 | 752 | 3.13% | 33.98% | 3.70% |
| 3) 只用 emblayerSeq（vec 保持原始 slice） | 45.754245 | 0.115751 | 0.05744 | 853 | 5.53% | -105.50% | -224.92% |
| 4) multislice + emblayerSeq | 47.274562 | 0.123065 | 0.058671 | 653 | 2.40% | -118.48% | -231.89% |
| 5) emblayerSeq + emblayerVec | 45.825286 | 0.164693 | 0.077414 | 653 | 5.39% | -192.39% | -337.91% |
| 6) emblayerSeq + emblayerVecV2（user 侧共享压缩） | 45.695002 | 0.210513 | 0.097279 | 653 | 5.66% | -273.73% | -450.28% |

## 产物路径

- 1) baseline（直接输入 padded）
  - ONNX: `onnx/din_stage1_baseline.onnx`
  - NSYS: `nsys/stage1_baseline.nsys-rep`
  - NSYS Stats: `nsys/stage1_baseline.stats.csv`
  - Log: `logs/stage1_baseline.log`
- 2) 只用 multislice
  - ONNX: `onnx/din_stage2_multislice.onnx`
  - NSYS: `nsys/stage2_multislice.nsys-rep`
  - NSYS Stats: `nsys/stage2_multislice.stats.csv`
  - Log: `logs/stage2_multislice.log`
- 3) 只用 emblayerSeq（vec 保持原始 slice）
  - ONNX: `onnx/din_stage3_emblayerseq_only.onnx`
  - NSYS: `nsys/stage3_emblayerseq_only.nsys-rep`
  - NSYS Stats: `nsys/stage3_emblayerseq_only.stats.csv`
  - Log: `logs/stage3_emblayerseq_only.log`
- 4) multislice + emblayerSeq
  - ONNX: `onnx/din_stage4_multislice_emblayerseq.onnx`
  - NSYS: `nsys/stage4_multislice_emblayerseq.nsys-rep`
  - NSYS Stats: `nsys/stage4_multislice_emblayerseq.stats.csv`
  - Log: `logs/stage4_multislice_emblayerseq.log`
- 5) emblayerSeq + emblayerVec
  - ONNX: `onnx/din_stage5_emblayer_joint.onnx`
  - NSYS: `nsys/stage5_emblayer_joint.nsys-rep`
  - NSYS Stats: `nsys/stage5_emblayer_joint.stats.csv`
  - Log: `logs/stage5_emblayer_joint.log`
- 6) emblayerSeq + emblayerVecV2（user 侧共享压缩）
  - ONNX: `onnx/din_stage6_emblayer_joint_v2.onnx`
  - NSYS: `nsys/stage6_emblayer_joint_v2.nsys-rep`
  - NSYS Stats: `nsys/stage6_emblayer_joint_v2.stats.csv`
  - Log: `logs/stage6_emblayer_joint_v2.log`

## 说明

- Stage0 使用 `export_din_frozen_graph.py` 导出 baseline/multislice ONNX，再通过 converter 生成 stage3/4/5/6 的真实推理 ONNX。
- Stage2/Stage3/Stage4/Stage5/Stage6 的 latency 分别来自对应 benchmark 脚本。
- Stage3 使用 `benchmark_dinemblayer_infer.py --mode seq_only`（运行时模式由 `BENCH_RUNTIME_MODE` 控制）。
- Stage4 使用 `benchmark_dinemblayer_infer.py --mode multislice_seq`（MultiSlice + EmblayerSeq）。
- Stage5 使用 `benchmark_dinemblayer_infer.py --mode joint`（EmblayerSeq + EmblayerVec）。
- Stage6 使用 `benchmark_dinemblayer_infer.py --mode joint_v2`（EmblayerSeq + EmblayerVecV2，user 特征共享压缩）。
- Stage6 额外参数 `USER_SPARSE_COUNT` 控制前多少个 sparse 特征按 user 共享处理。
- `BENCH_RUNTIME_MODE=emblayer`（默认）会使用 `--rebuild_from_emblayer`，用于真实测 Emblayer runtime。
- `BENCH_RUNTIME_MODE=scheduler` 会使用 `--scheduler_side_concat`，用于仅测调度侧已拼接输入场景。
