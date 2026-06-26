---
name: results-dashboard
description: Build or refresh a static HTML+ECharts dashboard that visualizes the RePairTQA × Qwen3-32B analysis results — accuracy (EM / PM / LLM-judge), tokenizer metrics, attention evidence-mass & entropy, and format-divergence — across the four representations (structured, semi-structured, unstructured, unstructured_full) and seven diagnostic splits. Use whenever the user asks to visualize, chart, plot, or build a dashboard/report UI from the experiment outputs under analysis_outputs/ and evaluation_outputs/.
---

# Results dashboard (static HTML + ECharts)

A **dependency-free, offline** dashboard for this fork's analysis outputs. No node
build, no runtime CDN: data is baked into the page and the charting lib is
vendored locally. Open the produced `dashboard/index.html` directly (file://) or
serve it.

## When to use
The user wants to *see* the experiment results — accuracy across representations,
attention behavior, tokenizer inflation, or the format-divergence (debiasing)
signal. Also use to refresh the dashboard after re-running the analysis pipeline.

## Stack & invariants (do not drift)
- **Pure static**: one `dashboard/index.html` + a vendored `dashboard/vendor/echarts.min.js`. No framework, no bundler, no network at view time.
- **Data is baked in**: `build_data.py` reads the analysis outputs and (a) writes `dashboard/data.json` and (b) inlines the same JSON into `index.html` (a `<script type="application/json">` block), so `index.html` works on `file://` with no server. A `fetch('./data.json')` fallback covers the `python -m http.server` path.
- **Charting lib must be vendored, via a domestic mirror** (the box has no general internet and the port-10088 proxy is off-limits). Fetch once:
  ```bash
  mkdir -p dashboard/vendor
  curl -L -o dashboard/vendor/echarts.min.js \
    https://registry.npmmirror.com/echarts/5.5.1/files/dist/echarts.min.js
  ```
  If that mirror path changes, any ECharts 5.x `echarts.min.js` works; do **not** point `index.html` at a live CDN.

## How to build / refresh
Run from the repo root with the project venv:
```bash
.venv/bin/python .claude/skills/results-dashboard/build_data.py
# -> dashboard/data.json and dashboard/index.html  (vendor echarts.min.js once, see above)
# view:  open dashboard/index.html   OR   .venv/bin/python -m http.server -d dashboard 8200
```
`build_data.py` is the contract: edit it (not ad-hoc parsing) when the upstream
schema changes, then regenerate. `template.html` is the chart layout; edit it to
add/restyle charts, then rebuild to re-inline the data.

## Data sources & schema (what build_data.py reads)
All produced by the `analysis/` pipeline. The dashboard is **read-only** over them.

- `analysis_outputs/<split>/report/merged.csv` — per `(id, representation)` row.
  Key columns: `EM`, `PM`, `judge` (0/1; present after `analysis.finalize_full`),
  `representation`, `split`, `total_tokens`, `length_inflation_ratio`,
  `structural_token_share`, `number_frag_mean`, `evidence_rel_pos`,
  `prompt_evidence_mass_overall`, `prompt_entropy_overall`,
  `gold_evidence_mass_overall`. The four representations appear as rows;
  `unstructured_full` rows exist only after `finalize_full`.
- `analysis_outputs/synthesis/format_divergence.csv` — per-sample, columns
  `split, id, contrast (structured~semi-structured | structured~unstructured_full),
  ev_div, ent_div, EM_structured, EM_other, flip_struct_to_other`. `ev_div` is the
  L2 distance between the structured and the other rep's per-layer evidence-mass
  attention profile — **the debiasing target; lower = more format-invariant.**
- `analysis_outputs/synthesis/pooled_by_representation.csv` — optional cross-check
  of pooled means (build_data recomputes pooled means from merged.csv itself, so
  this is not required).

If `merged.csv` is missing `judge` or `unstructured_full` rows, run the finalize
step first: `.venv/bin/python -m analysis.finalize_full` (needs the HF attention
outputs `tok_unstr_full.jsonl` / `attn_unstr_full.jsonl`; without them the full
view still shows accuracy but no attention bars).

## Canonical chart set (keep these; add more freely)
1. **Accuracy by representation** — grouped bars of pooled `EM` vs `judge` (judge
   is the headline, EM the strict floor). Rep order: structured → semi-structured
   → unstructured → unstructured_full.
2. **Accuracy heatmap** — `judge` over (split × representation); shows where format
   shift hurts most (long/compositional/multi-table splits).
3. **Attention** — pooled `prompt_evidence_mass_overall` (↑ better) and
   `prompt_entropy_overall` (↑ = more diffuse) per representation.
4. **Tokenizer cost** — pooled `length_inflation_ratio` and
   `structural_token_share` per representation.
5. **Format-divergence** — mean `ev_div` per contrast, plus mean `ev_div` for
   flips (struct✓→other✗) vs kept; this is the "format bias" headline.

## Representation semantics (label these in the UI)
- `structured` — full table as markdown (from `raw_tables`).
- `semi-structured` — leftover table rows + verbalized text (ratio 0.5).
- `unstructured` — verbalized text only, **but only ~half the rows** (ratio 0.5;
  a content-loss confound — annotate it).
- `unstructured_full` — **every** row verbalized (the fair text-only view; reads
  `verbalized_data_full`). Comparing it against `unstructured` separates pure
  format effect from data loss.

## Conventions
- Output goes under `dashboard/` at the repo root (git-ignore it unless the user
  wants it committed; it's regenerable).
- Keep the page self-contained and legible offline; no external fonts/CDNs.
- Use a fixed color per representation across all charts for cross-chart reading.
