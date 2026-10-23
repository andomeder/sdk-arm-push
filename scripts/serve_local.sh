#!/usr/bin/env bash
# Start llama.cpp's OpenAI-compatible server for the model under test.
# Order: Intel GPU via SYCL (oneAPI + Level Zero), then any other GPU build
# (Vulkan/CUDA, offload all layers), then CPU (-ngl 0). Same flags on every backend.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/_env.sh

LLAMA_SERVER=${LLAMA_SERVER:-llama-server}
GGUF=${GGUF:?set GGUF to a .gguf path (see README)}
PORT=${LLAMA_PORT:-8091}
CTX=${LLAMA_CTX:-8192}
command -v "$LLAMA_SERVER" >/dev/null || { echo "llama-server not found: set LLAMA_SERVER" >&2; exit 1; }
[ -f "$GGUF" ] || { echo "model not found: $GGUF" >&2; exit 1; }

# oneAPI runtime + Level Zero device selection for SYCL builds (harmless otherwise).
if [ -f "${ONEAPI_SETVARS:-/opt/intel/oneapi/setvars.sh}" ]; then
  set +u; . "${ONEAPI_SETVARS:-/opt/intel/oneapi/setvars.sh}" >/dev/null 2>&1 || true; set -u
fi
export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-level_zero:0}
export ZES_ENABLE_SYSMAN=${ZES_ENABLE_SYSMAN:-1}

# llama.cpp lists the devices its build can use, e.g. "SYCL0: Intel(R) Arc(TM) B580 ...".
devices=$("$LLAMA_SERVER" --list-devices 2>&1 | sed -n '/Available devices/,$p' | tail -n +2)
if grep -qi '^ *SYCL' <<<"$devices"; then backend=sycl
elif grep -qiE '^ *(Vulkan|CUDA|ROCm|OpenVINO)' <<<"$devices"; then backend=gpu
else backend=cpu; echo "no GPU device in this llama.cpp build; running on CPU" >&2
fi
echo "$devices" | sed 's/^/  /' >&2

ngl=99; [ "$backend" = cpu ] && ngl=0
echo "backend=$backend  model=$GGUF  http://127.0.0.1:$PORT/v1" >&2
# --jinja: use the model's chat template, which carries its tool-call format.
# -np 1: this llama.cpp defaults to several slots that share one KV pool (-c).
# A concurrent eval then fails with "Context size has been exceeded".
exec "$LLAMA_SERVER" -m "$GGUF" --jinja -ngl "$ngl" -c "$CTX" -np 1 \
  --host 127.0.0.1 --port "$PORT" --alias "${MODEL_NAME:-local}"
