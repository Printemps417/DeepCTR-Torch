#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/myenv/bin/python}"
NSYS_BIN="${NSYS_BIN:-nsys}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/nsys_din_emblayerseq_before_after}"
ITERS="${ITERS:-2000}"
WARMUP="${WARMUP:-300}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-128}"
AVG_SEQ_LEN="${AVG_SEQ_LEN:-8}"
INCLUDE_H2D="${INCLUDE_H2D:-1}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python not found or not executable: $PYTHON_BIN"
  exit 1
fi

if ! command -v "$NSYS_BIN" >/dev/null 2>&1; then
  echo "[ERROR] nsys not found: $NSYS_BIN"
  exit 1
fi

mkdir -p "$OUT_DIR"

BASE_PREFIX="$OUT_DIR/din_baseline"
EMB_PREFIX="$OUT_DIR/din_emblayerseq"

H2D_FLAG=""
if [[ "$INCLUDE_H2D" == "1" ]]; then
  H2D_FLAG="--include_h2d"
fi

echo "[1/4] Run baseline DIN under nsys..."
"$NSYS_BIN" profile \
  -t cuda,nvtx,osrt,cudnn,cublas \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  -o "$BASE_PREFIX" \
  "$PYTHON_BIN" benchmark_dinemblayerSeq_infer.py \
    --mode baseline \
    --iters "$ITERS" \
    --warmup "$WARMUP" \
    --batch_size "$BATCH_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --avg_seq_len "$AVG_SEQ_LEN" \
    $H2D_FLAG

echo "[2/4] Run DIN EmblayerSeq under nsys..."
"$NSYS_BIN" profile \
  -t cuda,nvtx,osrt,cudnn,cublas \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  -o "$EMB_PREFIX" \
  "$PYTHON_BIN" benchmark_dinemblayerSeq_infer.py \
    --mode emblayer \
    --iters "$ITERS" \
    --warmup "$WARMUP" \
    --batch_size "$BATCH_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --avg_seq_len "$AVG_SEQ_LEN" \
    $H2D_FLAG

echo "[3/4] Export nsys stats (kernel/api summaries)..."
"$NSYS_BIN" stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv "$BASE_PREFIX.nsys-rep" > "$BASE_PREFIX.stats.csv"
"$NSYS_BIN" stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv "$EMB_PREFIX.nsys-rep" > "$EMB_PREFIX.stats.csv"

echo "[4/4] Done"
echo "baseline report:   $BASE_PREFIX.nsys-rep"
echo "emblayer report:   $EMB_PREFIX.nsys-rep"
echo "baseline stats:    $BASE_PREFIX.stats.csv"
echo "emblayer stats:    $EMB_PREFIX.stats.csv"
