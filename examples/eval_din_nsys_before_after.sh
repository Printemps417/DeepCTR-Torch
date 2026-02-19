#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/myenv/bin/python}"
NSYS_BIN="${NSYS_BIN:-nsys}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/nsys_din_before_after}"
ITERS="${ITERS:-2000}"
WARMUP="${WARMUP:-300}"
BATCH_SIZE="${BATCH_SIZE:-1024}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python not found or not executable: $PYTHON_BIN"
  exit 1
fi

if ! command -v "$NSYS_BIN" >/dev/null 2>&1; then
  echo "[ERROR] nsys not found: $NSYS_BIN"
  exit 1
fi

mkdir -p "$OUT_DIR"

BASE_PREFIX="$OUT_DIR/din_before"
OPT_PREFIX="$OUT_DIR/din_after_multislice"

echo "[1/4] Run baseline DIN under nsys..."
"$NSYS_BIN" profile \
  -t cuda,nvtx,osrt,cudnn,cublas \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  -o "$BASE_PREFIX" \
  "$PYTHON_BIN" benchmark_din_multislice_infer.py \
    --mode baseline \
    --iters "$ITERS" \
    --warmup "$WARMUP" \
    --batch_size "$BATCH_SIZE"

echo "[2/4] Run MultiSlice DIN under nsys..."
"$NSYS_BIN" profile \
  -t cuda,nvtx,osrt,cudnn,cublas \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  -o "$OPT_PREFIX" \
  "$PYTHON_BIN" benchmark_din_multislice_infer.py \
    --mode multislice \
    --iters "$ITERS" \
    --warmup "$WARMUP" \
    --batch_size "$BATCH_SIZE"

echo "[3/4] Export nsys stats (kernel/api summaries)..."
"$NSYS_BIN" stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv "$BASE_PREFIX.nsys-rep" > "$BASE_PREFIX.stats.csv"
"$NSYS_BIN" stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv "$OPT_PREFIX.nsys-rep" > "$OPT_PREFIX.stats.csv"

echo "[4/4] Done"
echo "before report: $BASE_PREFIX.nsys-rep"
echo "after report:  $OPT_PREFIX.nsys-rep"
echo "before stats:  $BASE_PREFIX.stats.csv"
echo "after stats:   $OPT_PREFIX.stats.csv"
