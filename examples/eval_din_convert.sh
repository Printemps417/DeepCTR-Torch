#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/myenv/bin/python}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/eval_din_convert_outputs}"
USE_CPU="${USE_CPU:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python not found or not executable: $PYTHON_BIN"
  echo "Set PYTHON_BIN to your interpreter path, e.g.:"
  echo "  PYTHON_BIN=/path/to/python bash eval_din_convert.sh"
  exit 1
fi

mkdir -p "$OUT_DIR"

BEFORE_ONNX="$OUT_DIR/din_before.onnx"
AFTER_ONNX="$OUT_DIR/din_after_multislice.onnx"

CPU_FLAG=""
if [[ "$USE_CPU" == "1" ]]; then
  CPU_FLAG="--cpu"
fi

echo "[1/2] Export DIN ONNX and convert with MultiSlice..."
"$PYTHON_BIN" run_din_multislice_pipeline.py \
  --onnx_output "$BEFORE_ONNX" \
  --multislice_onnx_output "$AFTER_ONNX" \
  $CPU_FLAG \
  "$@"

echo "[2/2] Evaluate before/after ONNX models..."
"$PYTHON_BIN" - "$BEFORE_ONNX" "$AFTER_ONNX" <<'PY'
import sys
import time
import numpy as np
import onnx

before_path = sys.argv[1]
after_path = sys.argv[2]


def summarize(path, title):
    model = onnx.load(path)
    ops = {}
    for n in model.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(f"\n[{title}]")
    print("path:", path)
    print("total nodes:", len(model.graph.node))
    print("Slice:", ops.get("Slice", 0))
    print("MultiSlice:", ops.get("MultiSlice", 0))
    print("Top ops:", sorted(ops.items(), key=lambda x: (-x[1], x[0]))[:10])


def infer_input_shape(path):
    model = onnx.load(path)
    dims = model.graph.input[0].type.tensor_type.shape.dim
    shape = []
    for i, d in enumerate(dims):
        if d.dim_value > 0:
            shape.append(int(d.dim_value))
        else:
            shape.append(1 if i == 0 else 14)
    return tuple(shape)


def benchmark_onnxruntime(path, iters=300, warmup=50):
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(path, providers=providers)
    inp = sess.get_inputs()[0]
    shape = infer_input_shape(path)
    x = np.random.random(shape).astype(np.float32)

    for _ in range(warmup):
        sess.run(None, {inp.name: x})

    t0 = time.perf_counter()
    for _ in range(iters):
        sess.run(None, {inp.name: x})
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters


summarize(before_path, "Before Convert")
summarize(after_path, "After Convert")

try:
    import onnxruntime as _
except Exception:
    print("\n[ONNXRuntime Benchmark]")
    print("onnxruntime not installed, skip latency test.")
    print("Install with: pip install onnxruntime-gpu  (or onnxruntime)")
    sys.exit(0)

print("\n[ONNXRuntime Benchmark]")
try:
    ms_before = benchmark_onnxruntime(before_path)
    print(f"before convert latency: {ms_before:.4f} ms")
except Exception as e:
    print("before convert run failed:", repr(e))

try:
    ms_after = benchmark_onnxruntime(after_path)
    print(f"after convert latency:  {ms_after:.4f} ms")
except Exception as e:
    print("after convert run failed:", repr(e))
    print("note: converted model contains custom MultiSlice op and needs runtime plugin/custom op registration.")
PY

echo "\nDone. Artifacts:"
echo "  before: $BEFORE_ONNX"
echo "  after:  $AFTER_ONNX"
