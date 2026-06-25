#!/usr/bin/env bash
# Phase B: QA accuracy (EM/PM) for the three representations via Qwen3-32B on vLLM.
# Run 1 (data_type=semi-structured + --structured_output_file) yields BOTH the
# semi-structured result and the structured baseline (built from raw_tables).
# Run 2 yields unstructured. This makes 'structured' == original table (raw_tables),
# matching analysis/config.prompt_for.
#
# Usage: bash analysis/run_accuracy_vllm.sh [VERBALIZED_JSON]
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
MODEL=${MODEL:-Qwen3-32B}
IN=${1:-pipeline_results/run_example/example_verbalized.json}
OUT=${OUT:-baseline_outputs/example_qwen3}
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://localhost:8100/v1}
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy || true
mkdir -p "$OUT"

echo "[B1] semi-structured (main) + structured baseline (raw_tables)"
$PY src/llm_infer_s5.py \
  --input_file "$IN" \
  --output_file "$OUT/semi-structured.jsonl" \
  --structured_output_file "$OUT/structured.jsonl" \
  --data_type semi-structured \
  --model "$MODEL" --provider custom --api_key "$OPENAI_API_KEY" ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES}

echo "[B2] unstructured"
$PY src/llm_infer_s5.py \
  --input_file "$IN" \
  --output_file "$OUT/unstructured.jsonl" \
  --data_type unstructured \
  --model "$MODEL" --provider custom --api_key "$OPENAI_API_KEY" ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES}

echo "[B] done -> $OUT/{structured,semi-structured,unstructured}.jsonl"
