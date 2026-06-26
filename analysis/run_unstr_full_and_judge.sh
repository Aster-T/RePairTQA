#!/usr/bin/env bash
# vLLM phase for the FULL unstructured representation. Requires a running vLLM server.
#   (0) merge full-row text into the canonical 0.5 file as `verbalized_data_full`
#       (semi-structured config left untouched; sample set paired with the others)
#   (1) score unstructured-FULL by reading that field (--verbal_field)
#   (2) LLM-as-judge (s7) over every representation's predictions
set -uo pipefail
cd "$(dirname "$0")/.."

export OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://localhost:8100/v1}
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy || true
PY=.venv/bin/python
MODEL=${MODEL:-Qwen3-32B}
W=${WORKERS:-16}
SPLITS="bird_S1 bird_S3 bird_S4 bird_S5 tableeval_S2 mmqa_M1 mmqa_M2"

echo "##### Phase 0: merge full-row text -> verbalized_data_full (semi untouched) #####"
$PY -m analysis.merge_unstructured_full

echo "##### Phase B: FULL unstructured accuracy (reads verbalized_data_full) #####"
for s in $SPLITS; do
  vb="pipeline_results/$s/example_verbalized_full.json"
  [ -f "$vb" ] || { echo "[SKIP] $s (no full file)"; continue; }
  echo "=== B unstructured_full: $s @ $(date '+%T') ==="
  $PY src/llm_infer_s5.py --input_file "$vb" \
    --output_file "baseline_outputs/$s/unstructured_full.jsonl" \
    --data_type unstructured --verbal_field verbalized_data_full \
    --model "$MODEL" --provider custom --api_key EMPTY --workers "$W" \
    || echo "[WARN] B failed: $s"
done

echo "##### s7 LLM-as-judge over all representations #####"
for s in $SPLITS; do
  mkdir -p "evaluation_outputs/$s"
  for rep in structured semi-structured unstructured unstructured_full; do
    pred="baseline_outputs/$s/$rep.jsonl"
    [ -f "$pred" ] || continue
    echo "=== judge: $s/$rep @ $(date '+%T') ==="
    $PY src/gpt_evaluator_s7.py --input_file "$pred" \
      --output_file "evaluation_outputs/$s/${rep}_judged.jsonl" \
      --model "$MODEL" --workers "$W" 2>&1 | grep -E 'Accuracy|complete' || echo "[WARN] judge failed: $s/$rep"
  done
done
echo "##### vLLM phase DONE @ $(date '+%T') #####"
