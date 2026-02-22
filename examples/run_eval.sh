#!/usr/bin/env bash
set -euo pipefail

# 与在线压测参数保持一致（rate=2600 的单点探测下 avg_batch≈126，离线评测取 BATCH_SIZE=128）
RATE=2800
PYTHON_BIN=/root/autodl-tmp/myenv/bin/python \
OUT_DIR=./eval_din_emblayer_4way_outputs_rate${RATE} \
ITERS=300 \
WARMUP=50 \
BATCH_SIZE=128 \
MAX_SEQ_LEN=1024 \
AVG_SEQ_LEN=4 \
INCLUDE_H2D=1 \
NUM_SPARSE=300 \
NUM_SEQ=150 \
USER_SPARSE_COUNT=260 \
EVEN_VOCAB_SIZE=4096 \
ODD_VOCAB_SIZE=64 \
BENCH_RUNTIME_MODE=emblayer \
bash eval_din_emblayer_4way.sh