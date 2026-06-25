#!/usr/bin/env bash
# Launch a local vLLM OpenAI-compatible server for Qwen3-32B on the two RTX 4090s.
#
# Qwen3 "thinking" is disabled per-request by the clients (src/llm_infer_s5.py
# sends chat_template_kwargs.enable_thinking=false; the HF attention path uses
# enable_thinking=False), so the server itself stays vanilla.
#
# The two 4090s (GPU 2,3) may already host another vLLM. Set KILL_EXISTING=1 to
# stop any running vLLM first (the user authorized this for these cards).
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/home/amax/models/Qwen3-32B}
SERVED_NAME=${SERVED_NAME:-Qwen3-32B}
GPUS=${GPUS:-2,3}
TP=${TP:-2}
PORT=${PORT:-8100}
MAX_LEN=${MAX_LEN:-16384}
GPU_UTIL=${GPU_UTIL:-0.90}
# Dedicated vLLM env (isolated from .venv so vLLM's pinned torch/transformers
# don't clash with the HF-attention env that monkeypatches transformers internals).
VENV=${VENV:-/home/amax/al/RePairTQA/.venv-vllm}

# Never route through the port-10088 proxy; the weights are local so go offline.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

if [[ "${KILL_EXISTING:-0}" == "1" ]]; then
  echo "[serve] stopping existing vLLM EngineCore processes ..."
  pkill -f 'VLLM::EngineCore' 2>/dev/null || true
  pkill -f 'vllm.entrypoints' 2>/dev/null || true
  sleep 5
fi

echo "[serve] starting vLLM: $MODEL_PATH on GPUs $GPUS (TP=$TP) port $PORT"
CUDA_VISIBLE_DEVICES="$GPUS" exec "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TP" \
  --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --dtype bfloat16
