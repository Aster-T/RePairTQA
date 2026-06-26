#!/usr/bin/env bash
# HF phase: tokenization + attention for the FULL unstructured representation only
# (ANALYSIS_REPS=unstructured), over each split's example_verbalized_full.json.
# vLLM MUST be stopped first.
set -uo pipefail
cd "$(dirname "$0")/.."

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
export QWEN3_PATH=${QWEN3_PATH:-/home/amax/models/Qwen3-32B}
export ANALYSIS_REPS=unstructured_full   # full text-only view (reads verbalized_data_full)
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy || true
PY=.venv/bin/python
MAX_L=${MAX_L:-4096}
SPLITS="bird_S1 bird_S3 bird_S4 bird_S5 tableeval_S2 mmqa_M1 mmqa_M2"

for s in $SPLITS; do
  vb="pipeline_results/$s/example_verbalized_full.json"
  [ -f "$vb" ] || { echo "[SKIP] $s"; continue; }
  OUT="analysis_outputs/$s"; mkdir -p "$OUT"
  echo "##### unstr-full tokenization: $s @ $(date '+%T') #####"
  $PY -m analysis.tokenization_metrics --input "$vb" --output "$OUT/tok_unstr_full.jsonl" \
      --split "$s" --max_length "$MAX_L" 2>&1 | grep -vE "PyTorch was not found" | tail -2
  echo "##### unstr-full attention: $s @ $(date '+%T') #####"
  $PY -m analysis.attention_capture --input "$vb" --output "$OUT/attn_unstr_full.jsonl" \
      --max_l "$MAX_L" 2>&1 | grep -vE "PyTorch was not found|torch_dtype|it/s\]$" | tail -2
done
echo "##### unstr-full HF phase DONE @ $(date '+%T') #####"
