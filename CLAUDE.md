# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this fork is

Upstream (megagonlabs) is the data-construction + evaluation pipeline for the paper *"Same Content, Different Representations: A Controlled Study for Table QA"* (arXiv 2509.22983). **This fork adds a mechanistic study** under `analysis/`: for the *same* table content rendered in different serializations, it measures how the **Qwen3 tokenizer** segments it and how **Qwen3-32B's attention** is distributed while answering, then correlates both with QA accuracy. North-star goal: understand format bias well enough to eventually **debias LLMs against table serialization format**.

There is **no closed-source API access** here. The paper's GPT-4o/Gemini roles are all served by a **local Qwen3-32B via vLLM**. Read `README.md` ("Gotchas & Fixes") and `analysis/README.md` before touching the pipeline — they document non-obvious upstream breakage and the analysis phase ordering.

## Environments — never use system Python

Two **uv**-managed venvs at the repo root (py3.13), deliberately separated to avoid dependency coupling:

- **`.venv`** — the data pipeline (`src/`), the `analysis/` package, and the HF-eager attention capture. `torch==2.7.0` (a bare `pip install torch` pulls a huge CUDA-13 build — pin it).
- **`.venv-vllm`** — vLLM serving only. Pin `transformers==4.53.3` (vLLM 0.9.2's `aimv2` registration clashes with transformers 5.x).

Invoke explicitly, e.g. `.venv/bin/python -m analysis.synthesize`. Analysis modules are a package — run with `-m analysis.<mod>` from the repo root, not as loose scripts.

Downloads (models/datasets) must use **domestic mirrors (ModelScope / hf-mirror)**; **never** the port-10088 proxy (the user's VPN can't carry that traffic).

## Hardware constraints (hard rules)

- Use **only the two RTX 4090 48GB cards = physical GPUs 2,3**. The A800s (GPU 0,1) are **off-limits**.
- Always export `CUDA_DEVICE_ORDER=PCI_BUS_ID` (mixed A800+4090 box defaults to fastest-first ordering, so `CUDA_VISIBLE_DEVICES=2,3` otherwise grabs the wrong cards). The serve/attention scripts already do this.
- The 4090s are **shared with the sibling project `/home/amax/al/strable`** — coordinate, do not kill its live jobs.
- vLLM (~65GB) and HF-eager attention (~65GB) **cannot co-reside** → they **time-share** the 4090s. Stop one before starting the other; confirm with `nvidia-smi`.

## Architecture: data pipeline → accuracy → attention

The pipeline is a chain of numbered `src/` steps; outputs of one are inputs to the next. A sample is a dict with `id_`, `Question`, `answer`, optional `SQL`, `table_names`, `tables[]` (see README "Data Format").

```
s1 matched_columns_analysis_s1.py   which columns appear in the SQL (naive substring match) → table_match.csv
s2 column_selection_s2.py           LLM picks important columns / column combinations
s3 template_generation_s3.py        LLM writes NL templates per column combo (--all_columns variant)
s4 verbalization_col_s4.py          column-based verbalization (--no_sql supported)
s4_1 verbalization_row_s4_1.py      ROW-based verbalization (--ratio): the one that emits verbalized_data
s5 llm_infer_s5.py                  QA inference → EM (exact) / PM (substring) per sample
s6 nl2sql_baseline_s6.py            SELECT-only NL2SQL baseline over in-memory SQLite
s7 gpt_evaluator_s7.py              LLM-as-judge: CORRECT/INCORRECT (more lenient than EM)
```

**The central data-flow fact** (it took the original authors and this fork the most pain): `verbalized_data` + `raw_tables` exist **only after Step 4/4_1**. The three representations all derive from one post-s4_1 file:

- `structured` ← `raw_tables` (original full table as markdown)
- `semi-structured` ← `tables` (rows left after verbalization) **+** `verbalized_data` (text)
- `unstructured` ← `verbalized_data` only

`verbalization_row_s4_1.run(ratio=r)` verbalizes a fraction `r` of rows into text and leaves the rest as `tables`. At the paper's default `ratio=0.5`, `unstructured` only sees *half* the rows — a content-loss confound, not pure format. To study a *fair* full-text view this fork adds **`verbalized_data_full`**: every row verbalized, stored in a **separate field** so the semi/structured configs on `verbalized_data`/`tables` stay byte-identical (a single `verbalized_data` cannot be both half-text for semi and full-text for unstructured). `analysis/merge_unstructured_full.py` produces the clean `example_verbalized_full.json`; `s5 --verbal_field verbalized_data_full` and `analysis/config.prompt_for(sample, "unstructured_full")` read it.

### Analysis subsystem (`analysis/`, Phases A–E)

`config.py` bridges to `src/llm_infer_s5.py` so analysis prompts are **byte-identical** to the accuracy run. Phase ordering (each over all 7 diagnostic splits: `bird_S1/S3/S4/S5`, `tableeval_S2`, `mmqa_M1/M2`):

- **A** build paired data (s1→s4_1) — vLLM up → `pipeline_results/<split>/example_verbalized.json`
- **B** accuracy (s5 ×reps) — same vLLM → `baseline_outputs/<split>/<rep>.jsonl`
- **C** stop vLLM
- **D** tokenization (CPU) + attention (HF eager) → `analysis_outputs/<split>/{tok.jsonl, attn.jsonl}`
- **E** `correlate.py` joins tok+attn+EM/PM → `analysis_outputs/<split>/report/merged.csv`
- cross-split: `synthesize.py` → `analysis_outputs/synthesis/` (pooled metrics + `format_divergence.csv`, the L2 distance between structured vs re-serialized attention profiles — the quantity a debiasing method should minimize). `finalize_full.py` rebuilds `merged.csv` to add `unstructured_full` as a 4th rep and a per-rep LLM-judge column.

## Non-negotiable implementation invariants

- **Attention capture must never use `output_attentions=True`** — on 64 layers it materializes all layer matrices at once and OOMs (~L>1200). `attention_capture.install_patch()` monkeypatches Qwen3's `eager_attention_forward` to reduce each layer to per-head scalars and free the matrix immediately (peak = one layer). Requires `attn_implementation="eager"`. Keep `MAX_L=4096`; `run_attention_hf.sh` runs a `--micro_test` memory gate first.
- **Greedy + Qwen3 "thinking" OFF everywhere** (else JSON answer parsing breaks). vLLM clients send `chat_template_kwargs={"enable_thinking": false}`; the HF path passes `enable_thinking=False` to `apply_chat_template`. Greedy makes EM correspond to the captured attention.
- **`OPENAI_BASE_URL` gates the local-model repoints** in `column_selection_s2.py`, `template_generation_s3.py`, `llm_infer_s5.py`, `gpt_evaluator_s7.py`: unset = real OpenAI (no behavior change); set = local vLLM at `http://localhost:8100/v1`. Stay additive — don't make these depend on vLLM unconditionally.
- **`MAX_TABLE_ROWS` (default 1000)** caps rows when building prompts/counting cells so monster tables (MMQA M2 ~190k rows) don't hang CPU tokenization. Honor it in any new prompt/table code.
- **`--workers N`** (s2/s3/s5/s7) batches concurrent requests through vLLM. Short outputs (s2/s5/s7) are bit-identical across worker counts; long outputs (s3 templates) vary run-to-run from vLLM's greedy non-determinism — that's inherent, not a concurrency bug.
- **BIRD row-cache staleness:** `bird_pipeline.py` reuses `.table_rows.pkl` without checking coverage. Building splits before all DBs are extracted yields `row_count=0` → tables wrongly pass the "short" filter and get fetched uncapped → 30GB+ split + OOM. Fix is operational: delete the pkl and rebuild once all train+dev DBs are present (not a code change).

## Common commands

```bash
# Serve Qwen3-32B on the 4090s (TP=2, port 8100). KILL_EXISTING=1 to replace a running server.
bash scripts/serve_vllm_qwen3.sh

# Phase A+B over all splits (vLLM up). INCLUDE_NOSQL=1 adds TableEval/MMQA; WORKERS=16 for concurrency.
INCLUDE_NOSQL=1 WORKERS=16 bash analysis/run_all_AB.sh

# Full-unstructured accuracy + LLM-judge over all reps (vLLM up)
bash analysis/run_unstr_full_and_judge.sh

# --- stop vLLM, confirm nvidia-smi shows the 4090s free, then: ---

# Phase D+E attention/correlate (HF eager, no vLLM)
INCLUDE_NOSQL=1 bash analysis/run_all_DE.sh
# unstructured_full attention only
bash analysis/run_unstr_full_attention.sh

# Cross-split synthesis (CPU)
.venv/bin/python -m analysis.finalize_full
.venv/bin/python -m analysis.synthesize

# Memory gate / single-module sanity (CPU tokenization needs only the tokenizer files)
.venv/bin/python -m analysis.attention_capture --micro_test --micro_l 512
.venv/bin/python -m analysis.tokenization_metrics --input <verbalized.json> --output /tmp/tok.jsonl --split dbg
```

There is **no unit-test suite**. Verification is empirical: the toy `sample_data/example.json` exercises the whole chain (it has *no* `raw_tables`/`verbalized_data`, so it must go through s1→s4_1 first — it cannot be fed directly to s5/s6), and the attention `--micro_test` gates GPU memory before a full run.

## Git / data hygiene

- Large datasets and the 65GB model weights are **git-ignored and must not be committed** (dataset redistribution is not permitted per the upstream Disclosure).
- The fork remote is `Aster-T/RePairTQA`; commit as user `Aster-T` (a stray `pass-lin` identity has needed fixing before — verify `git config user.name` before committing).
