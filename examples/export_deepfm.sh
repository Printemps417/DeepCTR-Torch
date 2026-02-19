#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/myenv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python not found or not executable: $PYTHON_BIN"
  echo "Set PYTHON_BIN to your interpreter path, e.g.:"
  echo "  PYTHON_BIN=/path/to/python bash export_deepfm.sh"
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import importlib.util
import sys

missing = [pkg for pkg in ("onnx", "onnxscript") if importlib.util.find_spec(pkg) is None]
if missing:
    print("[ERROR] Missing packages:", ", ".join(missing))
    print("Install with:")
    print("  pip install onnx onnxscript")
    sys.exit(1)
PY

"$PYTHON_BIN" export_deepfm_frozen_graph.py \
  --cpu \
  --opset 18 \
  --onnx_output ./deepfm_frozen.onnx \
  --torchscript_output ./deepfm_frozen.ts \
  "$@"

echo "Done. Open with Netron:"
echo "  $SCRIPT_DIR/deepfm_frozen.onnx"
