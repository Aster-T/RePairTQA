# PLAN — Where does the LLM "wash out" table format? (layer-wise canonicalization study)

> **Status: PROPOSAL only. No code written, no GPU run.** Awaiting approval of the
> code changes in §"Code changes" before implementing.

## Question

For the SAME table content rendered two ways (structured ↔ semi-structured), at
which **decoder layer** does Qwen3-32B's internal representation stop depending on
the serialization format — i.e. where is surface form discarded and only content
left? And **causally**, which layer/component carries the format effect?

Why it matters: (a) a mechanistic result none of the three reference papers have
(they stop at behavior / SQL attribution); (b) it directly tells FIFT (see
[FIFT PLAN](../format_invariance/PLAN.md)) **which layer to put the consistency loss at**.

## Precise definitions

- **Layer** = one of the **64 Qwen3 decoder blocks** (`num_hidden_layers=64`). The
  object measured = the **residual-stream hidden state**, a **5120-d** vector
  (`hidden_size=5120`) at each block's output → HF `output_hidden_states` returns
  65 tensors (embedding + 64 layers). NOT the tokenizer / a single head / MLP neuron.
- **Anchor position** = the **last prompt token** (the "decision token"): same
  semantic role in both formats, so no cross-tokenization alignment needed.
  Robustness anchors: mean-pool over prompt tokens; mean-pool over the evidence span.
- **Pair** = same content `id`, structured vs semi prompts (controlled contrast),
  restricted to the ≤2048-token subset (526 pairs; same filter as FIFT).
- **Distance** (per the metric discussion): **per-dimension standardized L2**
  (z-score each of the 5120 dims across the pooled set, then L2) — *not* raw L2
  (residual norms grow with depth + outlier dims dominate). Cross-check with
  **mean-centered cosine**. **Always reported as a RATIO to a baseline** so it is
  unitless and comparable across layers (see Phase 1).

## Method (cheap → causal)

**Phase 1 — Representational convergence curve (correlational).**
Per layer L, at the anchor:
- `d_within(L)` = mean standardized distance between the two *formats of the same
  content* (paired).
- `d_across(L)` = mean distance between *different contents, same format* (the
  natural spread / baseline).
- Plot **`r(L) = d_within / d_across`** vs L.
  - `r(L) → 0` at some L\* ⇒ formats converge ⇒ **surface form removed at L\***.
  - `r(L) ≈ 1` throughout ⇒ a format change moves the state as much as a content
    change ⇒ **never canonicalizes** (consistent with our attention-proxy result).

**Phase 2 — Per-layer format probe.** Linear classifier "structured vs semi" on the
anchor hidden state at each layer, **GroupKFold by content id** (no leakage). AUC vs
L; the layer where **AUC → 0.5** = format info linearly washed out. (This is the
proper, residual-stream version of the attention-proxy probe we already ran,
which gave AUC 0.77 — i.e. format stayed decodable.)

**Phase 3 — Logit lens (content readout).** Apply the unembedding to `h_L` and record
at which layer each format's top token matches the gold answer. The **depth gap**
"structured locks the answer at layer a, semi at layer b>a (or never)" localizes the
format cost in depth. Cheap once hidden states are saved; intuitive to visualize.

**Phase 4 — Causal activation patching (gold standard).** For pairs where structured
is right and semi wrong: run the semi forward pass, **patch in structured's `h_L`** at
the anchor, continue, and check whether the output flips to correct/structured
behavior. Sweep L → the layer(s) where patching **recovers** = the carrier layer.
Refine by patching residual-stream vs attention-output vs MLP-output to localize the
component. Report recovery-rate vs L (control layers should show ~0).

## Code changes (THIS is what you approve)

1. **`analysis/format_invariance/capture_hidden.py`** (new): forward pass with
   `output_hidden_states=True`, save the **anchor-position** hidden state for all 64
   layers per `(id, representation)` → compact `.npz` (65×5120 floats/sample/format ≈
   ~0.67 MB/sample/format; ≤526 pairs → a few GB total, fine). **Not** an OOM risk
   (hidden states are `[batch, seq, 5120]`, unlike `output_attentions`).
2. **`analysis/format_invariance/canonicalization.py`** (new, CPU): Phases 1–3
   (convergence curve, probes, logit lens) + plots/CSV from the saved hidden states.
3. **`analysis/format_invariance/patch.py`** (new, GPU): Phase 4 interchange
   intervention (the heavier new code).
4. *(optional, later)* a dashboard "Canonicalization" tab: `r(L)` + probe-AUC +
   logit-lens trajectory curves.

No existing files are modified except adding these new modules; `attention_capture.py`
is left as-is (hidden-state capture is a separate, additive module).

## Where it runs

- **GPU (await your go + free 4090s):** hidden-state capture (Phase 0 forward
  passes), logit lens unembedding, activation patching (Phase 4).
- **CPU (after capture):** convergence curve, probes, analysis, plots (Phases 1–3).

## Pre-registered outcomes (either is a result)

- **Deliverable:** the `r(L)` curve + probe-AUC curve → a clear yes/no + location for
  "does format wash out, and where".
  - *Clean collapse at L\** → canonicalization exists → FIFT loss goes at L\*.
  - *Persistent separation* → it doesn't → explains why steering failed and why FIFT
    must enforce **output-level** consistency (as currently planned).
- **Causal success:** patching at the identified layer recovers ≥ (say) 30% of
  disagreement cases vs ~0% at control layers.
- **Consistency check:** the wash-out layer (if any) should line up with where our
  existing per-layer attention divergence drops.

## Scope / risks / honesty

- ≤2048-tok subset only (monster tables excluded) — state in any claim.
- Anchor choice matters → report last-token **and** mean-pool to show robustness.
- Linear probe = **lower bound** (format could be non-linearly encoded).
- Patching across different tokenizations is only clean at the **shared last-token
  anchor**; window/multi-position patching needs careful position handling.
- Report standardized-L2 **and** cosine so conclusions aren't a metric artifact.

## Execution checklist

1. (CPU) write `capture_hidden.py` + `canonicalization.py`; dry-run on the toy sample.
2. (GPU, on go) capture hidden states for the 526 pairs.
3. (CPU) convergence curve + probes + logit lens → candidate layer(s).
4. (GPU, on go) patch at candidate + control layers.
5. (CPU) synthesize; feed L\* into [FIFT PLAN](../format_invariance/PLAN.md) (FIFT); optional dashboard tab.
