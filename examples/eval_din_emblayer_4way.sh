#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/myenv/bin/python}"
NSYS_BIN="${NSYS_BIN:-nsys}"

OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/eval_din_emblayer_4way_outputs}"
ONNX_DIR="$OUT_DIR/onnx"
NSYS_DIR="$OUT_DIR/nsys"
LOG_DIR="$OUT_DIR/logs"
REPORT_PATH="$OUT_DIR/report.md"

ITERS="${ITERS:-2000}"
WARMUP="${WARMUP:-300}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-128}"
AVG_SEQ_LEN="${AVG_SEQ_LEN:-8}"
INCLUDE_H2D="${INCLUDE_H2D:-1}"

SEQ_RANGES="${SEQ_RANGES:-5:9,9:13}"
VEC_RANGES="${VEC_RANGES:-0:1,1:2,2:3,3:4,4:5}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python not found or not executable: $PYTHON_BIN"
  exit 1
fi
if ! command -v "$NSYS_BIN" >/dev/null 2>&1; then
  echo "[ERROR] nsys not found: $NSYS_BIN"
  exit 1
fi

mkdir -p "$ONNX_DIR" "$NSYS_DIR" "$LOG_DIR"

H2D_FLAG=""
if [[ "$INCLUDE_H2D" == "1" ]]; then
  H2D_FLAG="--include_h2d"
fi

echo "[1/6] Export ONNX variants (baseline/multislice/ms+seq/joint/seq-only)..."
BASE_ONNX="$ONNX_DIR/din_stage1_baseline.onnx"
MS_ONNX="$ONNX_DIR/din_stage2_multislice.onnx"
MS_SEQ_ONNX="$ONNX_DIR/din_stage3_multislice_emblayerseq.onnx"
JOINT_ONNX="$ONNX_DIR/din_stage4_emblayer_joint.onnx"
SEQ_ONLY_ONNX="$ONNX_DIR/din_stage5_emblayerseq_only.onnx"

"$PYTHON_BIN" export_din_frozen_graph.py \
  --cpu \
  --onnx_output "$BASE_ONNX" \
  --multislice_onnx_output "$MS_ONNX" \
  > "$LOG_DIR/export.log" 2>&1

"$PYTHON_BIN" onnx_emblayerseq_converter.py \
  --input "$MS_ONNX" \
  --output "$MS_SEQ_ONNX" \
  --seq_ranges "$SEQ_RANGES" \
  > "$LOG_DIR/convert_stage3.log" 2>&1

"$PYTHON_BIN" onnx_emblayer_joint_converter.py \
  --input "$BASE_ONNX" \
  --output "$JOINT_ONNX" \
  --vec_ranges "$VEC_RANGES" \
  --seq_ranges "$SEQ_RANGES" \
  > "$LOG_DIR/convert_stage4.log" 2>&1

"$PYTHON_BIN" onnx_emblayerseq_converter.py \
  --input "$BASE_ONNX" \
  --output "$SEQ_ONLY_ONNX" \
  --seq_ranges "$SEQ_RANGES" \
  > "$LOG_DIR/convert_stage5.log" 2>&1

echo "[2/6] Profile stage1 baseline (padded)..."
"$NSYS_BIN" profile \
  -t cuda,nvtx,osrt,cudnn,cublas \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  -o "$NSYS_DIR/stage1_baseline" \
  "$PYTHON_BIN" benchmark_dinemblayer_infer.py \
    --mode baseline \
    --iters "$ITERS" \
    --warmup "$WARMUP" \
    --batch_size "$BATCH_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --avg_seq_len "$AVG_SEQ_LEN" \
    $H2D_FLAG \
  > "$LOG_DIR/stage1_baseline.log" 2>&1

echo "[3/6] Profile stage2 multislice..."
"$NSYS_BIN" profile \
  -t cuda,nvtx,osrt,cudnn,cublas \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  -o "$NSYS_DIR/stage2_multislice" \
  "$PYTHON_BIN" benchmark_dinemblayer_infer.py \
    --mode multislice \
    --iters "$ITERS" \
    --warmup "$WARMUP" \
    --batch_size "$BATCH_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --avg_seq_len "$AVG_SEQ_LEN" \
    $H2D_FLAG \
  > "$LOG_DIR/stage2_multislice.log" 2>&1

echo "[4/6] Profile stage3 multislice+emblayerSeq..."
"$NSYS_BIN" profile \
  -t cuda,nvtx,osrt,cudnn,cublas \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  -o "$NSYS_DIR/stage3_multislice_emblayerseq" \
  "$PYTHON_BIN" benchmark_dinemblayer_infer.py \
    --mode multislice_seq \
    --iters "$ITERS" \
    --warmup "$WARMUP" \
    --batch_size "$BATCH_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --avg_seq_len "$AVG_SEQ_LEN" \
    $H2D_FLAG \
  > "$LOG_DIR/stage3_multislice_emblayerseq.log" 2>&1

echo "[5/6] Profile stage4 emblayerSeq+emblayerVec..."
"$NSYS_BIN" profile \
  -t cuda,nvtx,osrt,cudnn,cublas \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  -o "$NSYS_DIR/stage4_emblayer_joint" \
  "$PYTHON_BIN" benchmark_dinemblayer_infer.py \
    --mode joint \
    --iters "$ITERS" \
    --warmup "$WARMUP" \
    --batch_size "$BATCH_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --avg_seq_len "$AVG_SEQ_LEN" \
    $H2D_FLAG \
  > "$LOG_DIR/stage4_emblayer_joint.log" 2>&1

echo "[6/6] Profile stage5 emblayerSeq only..."
"$NSYS_BIN" profile \
  -t cuda,nvtx,osrt,cudnn,cublas \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  -o "$NSYS_DIR/stage5_emblayerseq_only" \
  "$PYTHON_BIN" benchmark_dinemblayer_infer.py \
    --mode seq_only \
    --iters "$ITERS" \
    --warmup "$WARMUP" \
    --batch_size "$BATCH_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --avg_seq_len "$AVG_SEQ_LEN" \
    $H2D_FLAG \
  > "$LOG_DIR/stage5_emblayerseq_only.log" 2>&1

for name in stage1_baseline stage2_multislice stage3_multislice_emblayerseq stage4_emblayer_joint stage5_emblayerseq_only; do
  "$NSYS_BIN" stats --force-export=true --report cuda_gpu_kern_sum,cuda_api_sum --format csv "$NSYS_DIR/$name.nsys-rep" > "$NSYS_DIR/$name.stats.csv"
done

"$PYTHON_BIN" - "$ONNX_DIR" "$LOG_DIR" "$REPORT_PATH" <<'PY'
import os
import re
import sys
import onnx

onnx_dir, log_dir, report_path = sys.argv[1], sys.argv[2], sys.argv[3]

stage_info = {
    "stage1_baseline": {
        "title": "1) baseline（直接输入 padded）",
        "onnx": os.path.join(onnx_dir, "din_stage1_baseline.onnx"),
        "log": os.path.join(log_dir, "stage1_baseline.log"),
        "nsys": "nsys/stage1_baseline.nsys-rep",
        "stats": "nsys/stage1_baseline.stats.csv",
    },
    "stage2_multislice": {
        "title": "2) 只用 multislice",
        "onnx": os.path.join(onnx_dir, "din_stage2_multislice.onnx"),
        "log": os.path.join(log_dir, "stage2_multislice.log"),
        "nsys": "nsys/stage2_multislice.nsys-rep",
        "stats": "nsys/stage2_multislice.stats.csv",
    },
    "stage3_multislice_emblayerseq": {
        "title": "3) multislice + emblayerSeq",
        "onnx": os.path.join(onnx_dir, "din_stage3_multislice_emblayerseq.onnx"),
        "log": os.path.join(log_dir, "stage3_multislice_emblayerseq.log"),
        "nsys": "nsys/stage3_multislice_emblayerseq.nsys-rep",
        "stats": "nsys/stage3_multislice_emblayerseq.stats.csv",
    },
    "stage4_emblayer_joint": {
        "title": "4) emblayerSeq + emblayerVec",
        "onnx": os.path.join(onnx_dir, "din_stage4_emblayer_joint.onnx"),
        "log": os.path.join(log_dir, "stage4_emblayer_joint.log"),
        "nsys": "nsys/stage4_emblayer_joint.nsys-rep",
        "stats": "nsys/stage4_emblayer_joint.stats.csv",
    },
    "stage5_emblayerseq_only": {
      "title": "5) 只用 emblayerSeq",
      "onnx": os.path.join(onnx_dir, "din_stage5_emblayerseq_only.onnx"),
      "log": os.path.join(log_dir, "stage5_emblayerseq_only.log"),
      "nsys": "nsys/stage5_emblayerseq_only.nsys-rep",
      "stats": "nsys/stage5_emblayerseq_only.stats.csv",
    },
}


def read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def first_match(text, patterns):
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1)
    return "N/A"


def onnx_summary(path):
    if not os.path.exists(path):
        return {"nodes": "N/A", "slice": "N/A", "multislice": "N/A", "emblayerseq": "N/A", "emblayervec": "N/A"}
    m = onnx.load(path)
    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    return {
        "nodes": str(len(m.graph.node)),
        "slice": str(ops.get("Slice", 0)),
        "multislice": str(ops.get("MultiSlice", 0)),
        "emblayerseq": str(ops.get("EmblayerSeq", 0)),
        "emblayervec": str(ops.get("EmblayerVec", 0)),
    }

lines = []
lines.append("# DIN 五种方式对比报告")
lines.append("")
lines.append("本报告由 `examples/eval_din_emblayer_4way.sh` 自动生成。")
lines.append("")
lines.append("## 总览")
lines.append("")
lines.append("| Stage | Avg Latency (ms) | H2D-only (ms) | ONNX Nodes | Slice | MultiSlice | EmblayerSeq | EmblayerVec |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

for key, info in stage_info.items():
    txt = read_text(info["log"])
    avg = first_match(txt, [
        r"baseline avg latency \(ms\):\s*([0-9.]+)",
        r"joint avg latency \(ms\):\s*([0-9.]+)",
        r"hybrid avg latency \(ms\):\s*([0-9.]+)",
        r"avg latency \(ms\):\s*([0-9.]+)",
    ])
    h2d = first_match(txt, [
        r"baseline h2d-only \(ms\):\s*([0-9.]+)",
        r"joint h2d-only \(ms\):\s*([0-9.]+)",
        r"hybrid h2d-only \(ms\):\s*([0-9.]+)",
        r"h2d-only \(ms\):\s*([0-9.]+)",
    ])
    s = onnx_summary(info["onnx"])
    lines.append(f"| {info['title']} | {avg} | {h2d} | {s['nodes']} | {s['slice']} | {s['multislice']} | {s['emblayerseq']} | {s['emblayervec']} |")

lines.append("")
lines.append("## 产物路径")
lines.append("")
for _, info in stage_info.items():
    lines.append(f"- {info['title']}")
    lines.append(f"  - ONNX: `{os.path.relpath(info['onnx'], os.path.dirname(report_path))}`")
    lines.append(f"  - NSYS: `{info['nsys']}`")
    lines.append(f"  - NSYS Stats: `{info['stats']}`")
    lines.append(f"  - Log: `logs/{os.path.basename(info['log'])}`")

lines.append("")
lines.append("## 说明")
lines.append("")
lines.append("- Stage2/Stage3/Stage4/Stage5 的 latency 分别来自对应 benchmark 脚本。")
lines.append("- Stage3 使用 `benchmark_dinemblayer_infer.py --mode multislice_seq`（MultiSlice + EmblayerSeq）。")
lines.append("- Stage4 使用 `benchmark_dinemblayer_infer.py --mode joint`（EmblayerSeq + EmblayerVec）。")
lines.append("- Stage5 使用 `benchmark_dinemblayer_infer.py --mode seq_only`（只用 EmblayerSeq）。")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"Report generated: {report_path}")
PY

echo "Done. Report: $REPORT_PATH"
echo "ONNX files: $ONNX_DIR"
echo "NSYS files: $NSYS_DIR"
