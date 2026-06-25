#!/usr/bin/env bash
# Phase A (verbalization) + Phase B (accuracy) over the real diagnostic splits,
# via the running vLLM Qwen3-32B server.
#
# Default split list = BIRD S1/S3/S4/S5 (these carry SQL, so s4_1 works as-is).
# To also include the no-SQL datasets (TableEval S2, MMQA M1/M2), s4_1 needs a
# --no_sql path; set INCLUDE_NOSQL=1 AFTER that flag is added (build_paired_data
# passes NO_SQL=1 through for those).
set -uo pipefail
cd "$(dirname "$0")/.."

export OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://localhost:8100/v1}
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy || true
export WORKERS="${WORKERS:-1}"   # concurrent LLM requests for s2/s3/s5 (child scripts read it)
PY=.venv/bin/python

# name|input_json|needs_no_sql
SPLITS=(
  "bird_S1|data_selection/BIRD/splits/bird_S1_short_lookup_with_answer.json|0"
  "bird_S3|data_selection/BIRD/splits/bird_S3_short_compositional_with_answer.json|0"
  "bird_S4|data_selection/BIRD/splits/bird_S4_long_lookup_with_answer.json|0"
  "bird_S5|data_selection/BIRD/splits/bird_S5_long_compositional_with_answer.json|0"
)
if [[ "${INCLUDE_NOSQL:-0}" == "1" ]]; then
  SPLITS+=(
    "tableeval_S2|data_selection/TableEval/TableEval_S2_simple_short_flat.json|1"
    "mmqa_M1|data_selection/MMQA/splits/MMQA_M1_short_multi.json|1"
    "mmqa_M2|data_selection/MMQA/splits/MMQA_M2_long_multi.json|1"
  )
fi

for entry in "${SPLITS[@]}"; do
  IFS='|' read -r name in needs_nosql <<< "$entry"
  [ -f "$in" ] || { echo "[SKIP] $name: missing $in"; continue; }
  n=$($PY -c "import json;print(len(json.load(open('$in'))))" 2>/dev/null)
  echo "######### Phase A: $name ($n samples) @ $(date '+%T') #########"
  OUT="pipeline_results/$name" NO_SQL="$needs_nosql" \
    bash analysis/build_paired_data.sh "$in" || echo "[WARN] Phase A failed: $name"
  vb="pipeline_results/$name/example_verbalized.json"
  [ -f "$vb" ] || { echo "[SKIP B] $name: no verbalized output"; continue; }
  echo "######### Phase B: $name @ $(date '+%T') #########"
  OUT="baseline_outputs/$name" \
    bash analysis/run_accuracy_vllm.sh "$vb" || echo "[WARN] Phase B failed: $name"
done
echo "===== ALL Phase A+B DONE @ $(date '+%T') ====="
