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
BENCH_RUNTIME_MODE="${BENCH_RUNTIME_MODE:-emblayer}"
NUM_SPARSE="${NUM_SPARSE:-200}"
NUM_SEQ="${NUM_SEQ:-100}"
EVEN_VOCAB_SIZE="${EVEN_VOCAB_SIZE:-4096}"
ODD_VOCAB_SIZE="${ODD_VOCAB_SIZE:-64}"

SEQ_RANGES="${SEQ_RANGES:-}"
VEC_RANGES="${VEC_RANGES:-}"

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

BENCH_RUNTIME_FLAG="--rebuild_from_emblayer"
if [[ "$BENCH_RUNTIME_MODE" == "scheduler" ]]; then
  BENCH_RUNTIME_FLAG="--scheduler_side_concat"
elif [[ "$BENCH_RUNTIME_MODE" != "emblayer" ]]; then
  echo "[ERROR] BENCH_RUNTIME_MODE must be 'emblayer' or 'scheduler', got: $BENCH_RUNTIME_MODE"
  exit 1
fi
echo "[INFO] BENCH_RUNTIME_MODE: $BENCH_RUNTIME_MODE ($BENCH_RUNTIME_FLAG)"

if [[ -z "$SEQ_RANGES" || -z "$VEC_RANGES" ]]; then
  mapfile -t _AUTO_RANGES < <("$PYTHON_BIN" - <<'PY'
from export_din_frozen_graph import get_din_feature_config
from deepctr_torch.models.din import DIN
import os

max_seq_len = int(os.environ.get('MAX_SEQ_LEN', '128'))
num_sparse = int(os.environ.get('NUM_SPARSE', '200'))
num_seq = int(os.environ.get('NUM_SEQ', '100'))
even_vocab = int(os.environ.get('EVEN_VOCAB_SIZE', '4096'))
odd_vocab = int(os.environ.get('ODD_VOCAB_SIZE', '64'))

feature_columns, behavior = get_din_feature_config(
    max_seq_len=max_seq_len,
    num_sparse=num_sparse,
    num_seq=num_seq,
    even_vocab_size=even_vocab,
    odd_vocab_size=odd_vocab,
)
model = DIN(feature_columns, behavior, device='cpu', att_weight_normalization=True)
fi = model.feature_index

vec_ranges = []
for idx in range(num_sparse):
    s, e = fi[f'sparse_{idx}']
    vec_ranges.append(f'{s}:{e}')
s, e = fi['score']
vec_ranges.append(f'{s}:{e}')

seq_ranges = []
for idx in range(num_seq):
    s, e = fi[f'hist_sparse_{idx}']
    seq_ranges.append(f'{s}:{e}')

print(','.join(vec_ranges))
print(','.join(seq_ranges))
PY
)

  if [[ -z "$VEC_RANGES" ]]; then
    VEC_RANGES="${_AUTO_RANGES[0]}"
  fi
  if [[ -z "$SEQ_RANGES" ]]; then
    SEQ_RANGES="${_AUTO_RANGES[1]}"
  fi
fi

echo "[INFO] VEC_RANGES count: $(awk -F',' '{print NF}' <<< "$VEC_RANGES")"
echo "[INFO] SEQ_RANGES count: $(awk -F',' '{print NF}' <<< "$SEQ_RANGES")"

run_nsys_profile() {
  local rep_base="$1"
  local log_file="$2"
  shift 2

  set +e
  "$NSYS_BIN" profile \
    -t cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    -o "$rep_base" \
    "$@" > "$log_file" 2>&1
  local status=$?
  set -e

  if [[ $status -eq 0 ]]; then
    return 0
  fi

  if [[ $status -eq 139 && -f "${rep_base}.nsys-rep" ]]; then
    echo "[WARN] nsys exited 139 but report exists, continue: ${rep_base}.nsys-rep"
  elif [[ $status -ne 0 ]]; then
    echo "[ERROR] nsys profile failed with code ${status}: ${rep_base}"
    echo "[ERROR] see log: ${log_file}"
    return $status
  fi

  if ! grep -Eq 'avg latency \(ms\):' "$log_file"; then
    echo "[WARN] latency metrics missing in ${log_file}; rerun benchmark without nsys to fill report metrics"
    set +e
    "$@" >> "$log_file" 2>&1
    local metric_status=$?
    set -e
    if [[ $metric_status -ne 0 ]]; then
      echo "[ERROR] benchmark rerun for metrics failed with code ${metric_status}: ${log_file}"
      return $metric_status
    fi
  fi

  return 0
}

echo "[0/5] Stage0 Export ONNX variants for converter + graph stats..."
BASE_ONNX="$ONNX_DIR/din_stage1_baseline.onnx"
MS_ONNX="$ONNX_DIR/din_stage2_multislice.onnx"
SEQ_ONLY_ONNX="$ONNX_DIR/din_stage3_emblayerseq_only.onnx"
MS_SEQ_ONNX="$ONNX_DIR/din_stage4_multislice_emblayerseq.onnx"
JOINT_ONNX="$ONNX_DIR/din_stage5_emblayer_joint.onnx"

"$PYTHON_BIN" export_din_frozen_graph.py \
  --cpu \
  --skip_torchscript \
  --max_seq_len "$MAX_SEQ_LEN" \
  --avg_seq_len "$AVG_SEQ_LEN" \
  --num_sparse "$NUM_SPARSE" \
  --num_seq "$NUM_SEQ" \
  --even_vocab_size "$EVEN_VOCAB_SIZE" \
  --odd_vocab_size "$ODD_VOCAB_SIZE" \
  --onnx_output "$BASE_ONNX" \
  --multislice_onnx_output "$MS_ONNX" \
  > "$LOG_DIR/export_stage1.log" 2>&1

"$PYTHON_BIN" onnx_emblayerseq_converter.py \
  --input "$BASE_ONNX" \
  --output "$SEQ_ONLY_ONNX" \
  --seq_ranges "$SEQ_RANGES" \
  > "$LOG_DIR/export_stage3.log" 2>&1

"$PYTHON_BIN" onnx_emblayerseq_converter.py \
  --input "$MS_ONNX" \
  --output "$MS_SEQ_ONNX" \
  --seq_ranges "$SEQ_RANGES" \
  > "$LOG_DIR/export_stage4.log" 2>&1

"$PYTHON_BIN" onnx_emblayer_joint_converter.py \
  --input "$BASE_ONNX" \
  --output "$JOINT_ONNX" \
  --vec_ranges "$VEC_RANGES" \
  --seq_ranges "$SEQ_RANGES" \
  > "$LOG_DIR/export_stage5.log" 2>&1

echo "[1/5] Stage1 Profile baseline (padded)..."
run_nsys_profile "$NSYS_DIR/stage1_baseline" "$LOG_DIR/stage1_baseline.log" \
  "$PYTHON_BIN" benchmark_dinemblayer_infer.py \
    --mode baseline \
    --iters "$ITERS" \
    --warmup "$WARMUP" \
    --batch_size "$BATCH_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --avg_seq_len "$AVG_SEQ_LEN" \
    --num_sparse "$NUM_SPARSE" \
    --num_seq "$NUM_SEQ" \
    --even_vocab_size "$EVEN_VOCAB_SIZE" \
    --odd_vocab_size "$ODD_VOCAB_SIZE" \
    $BENCH_RUNTIME_FLAG \
    $H2D_FLAG

echo "[2/5] Stage2 Profile multislice..."
run_nsys_profile "$NSYS_DIR/stage2_multislice" "$LOG_DIR/stage2_multislice.log" \
  "$PYTHON_BIN" benchmark_dinemblayer_infer.py \
    --mode multislice \
    --iters "$ITERS" \
    --warmup "$WARMUP" \
    --batch_size "$BATCH_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --avg_seq_len "$AVG_SEQ_LEN" \
    --num_sparse "$NUM_SPARSE" \
    --num_seq "$NUM_SEQ" \
    --even_vocab_size "$EVEN_VOCAB_SIZE" \
    --odd_vocab_size "$ODD_VOCAB_SIZE" \
    $BENCH_RUNTIME_FLAG \
    $H2D_FLAG

echo "[3/5] Stage3 Profile emblayerSeq only..."
run_nsys_profile "$NSYS_DIR/stage3_emblayerseq_only" "$LOG_DIR/stage3_emblayerseq_only.log" \
  "$PYTHON_BIN" benchmark_dinemblayer_infer.py \
    --mode seq_only \
    --iters "$ITERS" \
    --warmup "$WARMUP" \
    --batch_size "$BATCH_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --avg_seq_len "$AVG_SEQ_LEN" \
    --num_sparse "$NUM_SPARSE" \
    --num_seq "$NUM_SEQ" \
    --even_vocab_size "$EVEN_VOCAB_SIZE" \
    --odd_vocab_size "$ODD_VOCAB_SIZE" \
    $BENCH_RUNTIME_FLAG \
    $H2D_FLAG

echo "[4/5] Stage4 Profile multislice+emblayerSeq..."
run_nsys_profile "$NSYS_DIR/stage4_multislice_emblayerseq" "$LOG_DIR/stage4_multislice_emblayerseq.log" \
  "$PYTHON_BIN" benchmark_dinemblayer_infer.py \
    --mode multislice_seq \
    --iters "$ITERS" \
    --warmup "$WARMUP" \
    --batch_size "$BATCH_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --avg_seq_len "$AVG_SEQ_LEN" \
    --num_sparse "$NUM_SPARSE" \
    --num_seq "$NUM_SEQ" \
    --even_vocab_size "$EVEN_VOCAB_SIZE" \
    --odd_vocab_size "$ODD_VOCAB_SIZE" \
    $BENCH_RUNTIME_FLAG \
    $H2D_FLAG

echo "[5/5] Stage5 Profile emblayerSeq+emblayerVec..."
run_nsys_profile "$NSYS_DIR/stage5_emblayer_joint" "$LOG_DIR/stage5_emblayer_joint.log" \
  "$PYTHON_BIN" benchmark_dinemblayer_infer.py \
    --mode joint \
    --iters "$ITERS" \
    --warmup "$WARMUP" \
    --batch_size "$BATCH_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --avg_seq_len "$AVG_SEQ_LEN" \
    --num_sparse "$NUM_SPARSE" \
    --num_seq "$NUM_SEQ" \
    --even_vocab_size "$EVEN_VOCAB_SIZE" \
    --odd_vocab_size "$ODD_VOCAB_SIZE" \
    $BENCH_RUNTIME_FLAG \
    $H2D_FLAG

for name in stage1_baseline stage2_multislice stage3_emblayerseq_only stage4_multislice_emblayerseq stage5_emblayer_joint; do
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
    "stage3_emblayerseq_only": {
      "title": "3) 只用 emblayerSeq（vec 保持原始 slice）",
      "onnx": os.path.join(onnx_dir, "din_stage3_emblayerseq_only.onnx"),
      "log": os.path.join(log_dir, "stage3_emblayerseq_only.log"),
      "nsys": "nsys/stage3_emblayerseq_only.nsys-rep",
      "stats": "nsys/stage3_emblayerseq_only.stats.csv",
    },
    "stage4_multislice_emblayerseq": {
        "title": "4) multislice + emblayerSeq",
      "onnx": os.path.join(onnx_dir, "din_stage4_multislice_emblayerseq.onnx"),
      "log": os.path.join(log_dir, "stage4_multislice_emblayerseq.log"),
      "nsys": "nsys/stage4_multislice_emblayerseq.nsys-rep",
      "stats": "nsys/stage4_multislice_emblayerseq.stats.csv",
    },
    "stage5_emblayer_joint": {
        "title": "5) emblayerSeq + emblayerVec",
      "onnx": os.path.join(onnx_dir, "din_stage5_emblayer_joint.onnx"),
      "log": os.path.join(log_dir, "stage5_emblayer_joint.log"),
      "nsys": "nsys/stage5_emblayer_joint.nsys-rep",
      "stats": "nsys/stage5_emblayer_joint.stats.csv",
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
lines.append("| 0) 导出与图转换 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")

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
lines.append("- Stage0 使用 `export_din_frozen_graph.py` 导出 baseline/multislice ONNX，再通过 converter 生成 stage3/4/5 的真实推理 ONNX。")
lines.append("- Stage2/Stage3/Stage4/Stage5 的 latency 分别来自对应 benchmark 脚本。")
lines.append("- Stage3 使用 `benchmark_dinemblayer_infer.py --mode seq_only`（运行时模式由 `BENCH_RUNTIME_MODE` 控制）。")
lines.append("- Stage4 使用 `benchmark_dinemblayer_infer.py --mode multislice_seq`（MultiSlice + EmblayerSeq）。")
lines.append("- Stage5 使用 `benchmark_dinemblayer_infer.py --mode joint`（EmblayerSeq + EmblayerVec）。")
lines.append("- `BENCH_RUNTIME_MODE=emblayer`（默认）会使用 `--rebuild_from_emblayer`，用于真实测 Emblayer runtime。")
lines.append("- `BENCH_RUNTIME_MODE=scheduler` 会使用 `--scheduler_side_concat`，用于仅测调度侧已拼接输入场景。")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"Report generated: {report_path}")
PY

echo "Done. Report: $REPORT_PATH"
echo "ONNX files: $ONNX_DIR"
echo "NSYS files: $NSYS_DIR"
