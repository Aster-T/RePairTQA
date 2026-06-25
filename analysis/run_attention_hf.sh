#!/usr/bin/env bash
# Phase D + E: tokenization metrics (CPU), attention capture (HF eager on the two
# 4090s), then correlation. vLLM MUST be stopped first (HF needs the cards empty);
# this script refuses to start the heavy model if the 4090s still look occupied.
#
# Usage: bash analysis/run_attention_hf.sh [VERBALIZED_JSON]
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
IN=${1:-pipeline_results/run_example/example_verbalized.json}
OUT=${OUT:-analysis_outputs}
ACC=${ACC:-baseline_outputs/example_qwen3}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
export QWEN3_PATH=${QWEN3_PATH:-/home/amax/models/Qwen3-32B}
MAX_L=${MAX_L:-4096}
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy || true
mkdir -p "$OUT"

echo "[D0] tokenization metrics (CPU)"
$PY -m analysis.tokenization_metrics --input "$IN" --output "$OUT/tok.jsonl" --max_length "$MAX_L"

if [[ "${SKIP_ATTENTION:-0}" != "1" ]]; then
  echo "[D1] memory gate (micro-test before full run)"
  $PY -m analysis.attention_capture --micro_test --micro_l 512

  echo "[D2] attention capture (HF eager, L<=$MAX_L)"
  $PY -m analysis.attention_capture \
    --input "$IN" --output "$OUT/attn.jsonl" \
    --max_l "$MAX_L" ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES}
  ATTN_ARG=(--attn "$OUT/attn.jsonl")
else
  ATTN_ARG=()
fi

echo "[E] correlate + report"
$PY -m analysis.correlate \
  --tok "$OUT/tok.jsonl" "${ATTN_ARG[@]}" \
  --acc structured="$ACC/structured.jsonl" \
        semi-structured="$ACC/semi-structured.jsonl" \
        unstructured="$ACC/unstructured.jsonl" \
  --out_dir "$OUT/report"

echo "[D/E] done -> $OUT/report"
