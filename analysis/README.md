# Analysis: Tokenization × Attention × Table-Sequence Semantic Invariance (Qwen3-32B)

Mechanistic add-on to RePairTQA. For the **same** table content rendered in the
paper's three representations, it measures how the **Qwen3 tokenizer** segments
the content and how **Qwen3-32B's attention** is distributed while answering,
then correlates both with QA accuracy — explaining *why* representation shift
moves accuracy even though semantics are invariant.

Representations (reuse `src/llm_infer_s5.py` prompt builders, so prompts are
byte-identical to the accuracy run):
- `structured`     — original full table as markdown (from `raw_tables`)
- `semi-structured`— verbalized table + verbalized text
- `unstructured`   — verbalized text only

## Hardware / env
- Model: `/home/amax/models/Qwen3-32B` (BF16). Serve / run on the two RTX 4090s
  (`CUDA_VISIBLE_DEVICES=2,3`). The A800s (0,1) are off-limits.
- Python env: `.venv` (uv, py3.13). Domestic mirrors only; **never** the
  port-10088 proxy.
- vLLM (accuracy) and HF-eager (attention) each need ~65GB → they **time-share**
  the 4090s. Stop one before starting the other.

## Phase ordering
```
# Phase A — paired data (needs vLLM up)
KILL_EXISTING=1 bash scripts/serve_vllm_qwen3.sh        # in one shell
bash analysis/build_paired_data.sh sample_data/example.json

# Phase B — accuracy (same vLLM)
bash analysis/run_accuracy_vllm.sh pipeline_results/run_example/example_verbalized.json

# Phase C — stop vLLM (free the 4090s), confirm with nvidia-smi

# Phase D+E — tokenization (CPU) + attention (HF eager) + report
bash analysis/run_attention_hf.sh pipeline_results/run_example/example_verbalized.json
```

`tokenization_metrics.py` is CPU-only and can run any time once the tokenizer
files are present. `run_attention_hf.sh` runs a memory micro-test gate before
the full attention pass (`MAX_L=4096`; falls back if OOM).

## Modules
| file | phase | what |
|------|-------|------|
| `config.py` | all | paths, model, `prompt_for(sample, rep)` mapping, s5 bridge |
| `evidence_span.py` | D,E | locate gold-answer token span (exact/fuzzy/missing) |
| `tokenization_metrics.py` | D0 | length, fertility, number fragmentation, structural share, evidence pos |
| `attention_capture.py` | D | HF eager, reduce-and-free attention hooks; evidence mass, entropy, struct ratio; `--micro_test` memory gate |
| `correlate.py` | E | join tok+attn+EM/PM; logistic regression; semantic-invariance contrast; plots |

## Key implementation notes
- **Attention memory:** never use `output_attentions=True` (OOMs ~L>1200 for 64
  layers). `attention_capture.install_patch()` wraps Qwen3's
  `eager_attention_forward` to reduce each layer to per-head scalars and free the
  matrix → peak = one layer.
- **Determinism:** greedy everywhere; Qwen3 *thinking* disabled (s5 sends
  `chat_template_kwargs.enable_thinking=false` to vLLM; the HF path passes
  `enable_thinking=False` to `apply_chat_template`).
- **Env-gated repoints:** `src/column_selection_s2.py`, `template_generation_s3.py`,
  `llm_infer_s5.py` read `OPENAI_BASE_URL`; unset = real OpenAI (no behavior
  change), set = local vLLM.
