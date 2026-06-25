#!/usr/bin/env bash
# Phase D (attention capture, HF eager) + Phase E (correlate) over the splits that
# already have Phase A/B outputs. vLLM MUST be stopped first (HF needs the 4090s).
#
# ATTN_MAX (optional): cap attention samples per split to bound wall-clock; the
# tokenization + accuracy still cover all samples.
set -uo pipefail
cd "$(dirname "$0")/.."

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
export QWEN3_PATH=${QWEN3_PATH:-/home/amax/models/Qwen3-32B}
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy || true
PY=.venv/bin/python
MAX_L=${MAX_L:-4096}

SPLITS=(bird_S1 bird_S3 bird_S4 bird_S5)
if [[ "${INCLUDE_NOSQL:-0}" == "1" ]]; then
  SPLITS+=(tableeval_S2 mmqa_M1 mmqa_M2)
fi

echo "===== memory gate (once) @ $(date '+%T') ====="
$PY -m analysis.attention_capture --micro_test --micro_l 512 2>&1 | grep -E 'n_layers|peak' || true

for name in "${SPLITS[@]}"; do
  vb="pipeline_results/$name/example_verbalized.json"
  [ -f "$vb" ] || { echo "[SKIP] $name: no verbalized data"; continue; }
  OUT="analysis_outputs/$name"; mkdir -p "$OUT"
  echo "######### D0 tokenization: $name @ $(date '+%T') #########"
  $PY -m analysis.tokenization_metrics --input "$vb" --output "$OUT/tok.jsonl" \
      --split "$name" --max_length "$MAX_L" 2>&1 | grep -vE "PyTorch was not found" | tail -3
  echo "######### D2 attention: $name @ $(date '+%T') #########"
  $PY -m analysis.attention_capture --input "$vb" --output "$OUT/attn.jsonl" \
      --max_l "$MAX_L" ${ATTN_MAX:+--max_samples $ATTN_MAX} 2>&1 \
      | grep -vE "PyTorch was not found|torch_dtype|it/s\]$" | tail -3
  echo "######### E correlate: $name #########"
  acc="baseline_outputs/$name"
  $PY -m analysis.correlate --tok "$OUT/tok.jsonl" --attn "$OUT/attn.jsonl" \
      --acc structured="$acc/structured.jsonl" \
            semi-structured="$acc/semi-structured.jsonl" \
            unstructured="$acc/unstructured.jsonl" \
      --out_dir "$OUT/report" 2>&1 | grep -vE "PyTorch was not found" | tail -3
done
echo "===== ALL Phase D+E DONE @ $(date '+%T') ====="
