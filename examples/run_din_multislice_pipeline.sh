#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/myenv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python not found or not executable: $PYTHON_BIN"
  echo "Set PYTHON_BIN to your interpreter path, e.g.:"
  echo "  PYTHON_BIN=/path/to/python bash run_din_multislice_pipeline.sh"
  exit 1
fi

"$PYTHON_BIN" run_din_multislice_pipeline.py "$@"
