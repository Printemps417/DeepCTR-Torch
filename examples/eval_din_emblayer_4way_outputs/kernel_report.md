# DIN 六种方式对比报告

本报告由 `examples/eval_din_emblayer_4way.sh` 自动生成。

## 总览

| Stage | Host Data Build (ms) | Feature Concat Prepare (ms) | Avg Latency (ms) | H2D-only (ms) | Pure H2D (ms) | H2D Total (B) | H2D Data (B) | H2D Meta (B) | Kernel Launches | ONNX Nodes | Build提升 | Concat提升 | Avg Latency提升 | H2D-only提升 | Pure H2D提升 | H2D Total提升 | H2D Data提升 | H2D Meta提升 | Kernel Launches提升 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1) baseline（直接输入 padded） | 699.087024 | 2.534736 | 91.731159 | 3.93846 | 1.403724 | 79056896 | 79056896 | 0 | 195482 | 1102 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | N/A | 0.00% |
| 2) 只用 multislice | 693.77124 | 2.520737 | 93.051152 | 3.92443 | 1.403693 | 79056896 | 79056896 | 0 | 195302 | 902 | 0.76% | 0.55% | -1.44% | 0.36% | 0.00% | 0.00% | 0.00% | N/A | 0.09% |
| 3) 只用 emblayerSeq（vec 保持原始 slice） | 725.506897 | 1.068611 | 93.077155 | 1.803256 | 0.734645 | 40908556 | 39984896 | 923660 | 195362 | 1102 | -3.78% | 57.84% | -1.47% | 54.21% | 47.66% | 48.25% | 49.42% | N/A | 0.06% |
| 4) multislice + emblayerSeq | 705.024483 | 1.563872 | 92.685494 | 3.692843 | 2.12897 | 119551756 | 118628096 | 923660 | 195422 | 902 | -0.85% | 38.30% | -1.04% | 6.24% | -51.67% | -51.22% | -50.05% | N/A | 0.03% |
| 5) emblayerSeq + emblayerVec | 717.861038 | 0.46514 | 92.524903 | 1.210106 | 0.744966 | 41324304 | 39984896 | 1339408 | 195302 | 902 | -2.69% | 81.65% | -0.87% | 69.27% | 46.93% | 47.73% | 49.42% | N/A | 0.09% |
| 6) emblayerSeq + emblayerVecV2（user 侧共享压缩） | 701.006071 | 0.518281 | 90.932211 | 1.265203 | 0.746922 | 40915504 | 39780496 | 1135008 | 195362 | 902 | -0.27% | 79.55% | 0.87% | 67.88% | 46.79% | 48.25% | 49.68% | N/A | 0.06% |

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
