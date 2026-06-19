#!/usr/bin/env bash
# Launch a vLLM OpenAI-compatible server from a config file.
# Usage:  ./serving/launch_vllm.sh serving/configs/tp2.env
#
# vLLM exposes Prometheus metrics at  http://HOST:PORT/metrics  by default,
# and the chat/completions API at     http://HOST:PORT/v1/...
set -euo pipefail

CONFIG="${1:?usage: launch_vllm.sh <config.env>}"
# shellcheck disable=SC1090
source "$CONFIG"

: "${MODEL:?set MODEL in the config}"
TP="${TP:-1}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-float16}"            # T4 has no bf16 — keep float16
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
EXTRA_ARGS="${EXTRA_ARGS:-}"        # per-experiment flags (expert parallel, prefix caching, ...)

echo "==> model=$MODEL  TP=$TP  dtype=$DTYPE  port=$PORT"
echo "==> extra: $EXTRA_ARGS"

# shellcheck disable=SC2086
exec vllm serve "$MODEL" \
  --tensor-parallel-size "$TP" \
  --dtype "$DTYPE" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --port "$PORT" \
  --host 0.0.0.0 \
  $EXTRA_ARGS
