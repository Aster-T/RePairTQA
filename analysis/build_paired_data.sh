#!/usr/bin/env bash
# Phase A: manufacture paired (structured/semi/unstructured) data with Qwen3-32B.
# Runs s1 -> s2 -> s3 --all_columns -> s4_1, with the LLM pointed at local vLLM.
# Output JSON gains raw_tables (structured) + verbalized_data (semi/unstructured).
#
# Usage: bash analysis/build_paired_data.sh [INPUT_JSON]
# Requires a running vLLM server (scripts/serve_vllm_qwen3.sh).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
MODEL=${MODEL:-Qwen3-32B}
DATA=${1:-sample_data/example.json}
OUT=${OUT:-pipeline_results/run_example}
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://localhost:8100/v1}
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy || true
mkdir -p "$OUT"

echo "[A1] table match (s1)"
$PY src/matched_columns_analysis_s1.py \
  --input "$DATA" --output "$OUT/table_match.csv" --mode table_match

echo "[A2] column selection (s2, LLM via vLLM)"
$PY src/column_selection_s2.py \
  --csv_in "$OUT/table_match.csv" \
  --csv_cols_out "$OUT/col_labels.csv" \
  --csv_tmpl_out "$OUT/col_combinations.csv" \
  --prompt_template_path prompts/column_selection.txt \
  --model "$MODEL" --api_key "$OPENAI_API_KEY" --workers "${WORKERS:-1}"

echo "[A3] template generation --all_columns (s3, LLM via vLLM)"
$PY src/template_generation_s3.py \
  --csv_templates "$OUT/col_combinations.csv" \
  --csv_matched "$OUT/table_match.csv" \
  --prompt_template_path prompts/template_generation.txt \
  --model "$MODEL" --api_key "$OPENAI_API_KEY" --all_columns --workers "${WORKERS:-1}"

echo "[A4] row-based verbalization (s4_1) -> verbalized_data + raw_tables"
NOSQL_FLAG=""
[ "${NO_SQL:-0}" = "1" ] && NOSQL_FLAG="--no_sql"   # for datasets without SQL (TableEval/MMQA)
$PY src/verbalization_row_s4_1.py \
  --data_path "$DATA" \
  --template_path "$OUT/col_combinations_all_columns.csv" \
  --output_path "$OUT/example_verbalized.json" \
  --failed_path "$OUT/example_failed.json" \
  --ratio "${RATIO:-0.5}" $NOSQL_FLAG

echo "[A] done -> $OUT/example_verbalized.json"
$PY - "$OUT/example_verbalized.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"samples: {len(d)}")
if d:
    print("has verbalized_data:", "verbalized_data" in d[0],
          "| has raw_tables:", "raw_tables" in d[0])
PY
