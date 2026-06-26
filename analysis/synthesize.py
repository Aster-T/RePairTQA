#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-split synthesis for the tokenization × attention × semantic-invariance study.

Pulls the per-split outputs (analysis_outputs/<split>/{report/merged.csv, attn.jsonl})
and produces:
  1. master_summary.csv  — per (split, representation) means of tokenizer + attention
     + EM metrics.
  2. pooled_by_representation.csv — pooled (all splits) structured/semi/unstructured
     comparison: the headline mechanism (evidence-mass down, entropy up, length/number
     fragmentation up → EM down).
  3. format_divergence.csv — THE debiasing-oriented metric: per sample, the distance
     between the *structured* and *semi-structured* attention profiles for the SAME
     content (same answer). A large divergence = the model processes the two formats
     differently = format bias. This is the quantity a future debiasing method would
     minimize. Reports mean divergence per split and its correlation with EM-flips
     (structured-correct but semi-wrong).
  4. SYNTHESIS.md — a readable write-up of the findings.

CPU-only; reads existing outputs, runs no model.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config

SPLITS = ["bird_S1", "bird_S3", "bird_S4", "bird_S5", "tableeval_S2", "mmqa_M1", "mmqa_M2"]
REPS = ["structured", "semi-structured", "unstructured", "unstructured_full"]
OUT = config.OUT_DIR / "synthesis"


def _read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# --------------------------------------------------------------------------- #
def load_merged() -> pd.DataFrame:
    frames = []
    for s in SPLITS:
        p = f"analysis_outputs/{s}/report/merged.csv"
        if os.path.exists(p):
            df = pd.read_csv(p)
            df["split"] = s
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


METRICS = [
    "EM", "PM", "judge", "total_tokens", "length_inflation_ratio", "fertility_cell",
    "structural_token_share", "number_frag_mean", "evidence_token_count",
    "evidence_rel_pos", "prompt_evidence_mass_overall", "prompt_entropy_overall",
    "gold_evidence_mass_overall",
]


def master_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [m for m in METRICS if m in df.columns]
    g = df.groupby(["split", "representation"])[cols].mean(numeric_only=True)
    order = [(s, r) for s in SPLITS for r in REPS]
    return g.reindex([o for o in order if o in g.index])


def pooled(df: pd.DataFrame) -> pd.DataFrame:
    cols = [m for m in METRICS if m in df.columns]
    g = df.groupby("representation")[cols].mean(numeric_only=True)
    return g.reindex([r for r in REPS if r in g.index])


# --------------------------------------------------------------------------- #
def _layer_div(a: dict, b: dict, key: str) -> float:
    va = np.array((a or {}).get(key) or [])
    vb = np.array((b or {}).get(key) or [])
    if va.shape != vb.shape or va.size == 0:
        return np.nan
    return float(np.linalg.norm(va - vb))


def format_divergence() -> pd.DataFrame:
    """Per-sample distance between the structured attention profile and each
    re-serialized profile (semi-structured, and the FULL unstructured text view)
    for the SAME content/answer.

    Uses the per-layer profiles captured in attn.jsonl / attn_unstr_full.jsonl:
      - prompt_evidence_mass_layer  : [n_layers]  (attention mass on the answer span)
      - prompt_entropy_layer        : [n_layers]  (dispersion of attention)
    Divergence = L2 distance between the structured and the other rep's layer
    vectors. A large divergence = the model processes the two formats differently
    = format bias; the quantity a debiasing method should drive to 0. Joined with
    EM to test: do format-divergent samples flip structured✓ -> other✗ more often?

    structured↔unstructured_full is the cleanest contrast: identical content, one
    as a table, one as full prose (no rows dropped), so divergence isolates the
    serialization format rather than any content loss.
    """
    rows = []
    for s in SPLITS:
        recs = _read_jsonl(f"analysis_outputs/{s}/attn.jsonl")
        recs += _read_jsonl(f"analysis_outputs/{s}/attn_unstr_full.jsonl")
        by_id: Dict[str, Dict[str, dict]] = {}
        for r in recs:
            by_id.setdefault(r["id"], {})[r["representation"]] = r
        # EM per (id, rep) from merged
        mp = f"analysis_outputs/{s}/report/merged.csv"
        em = {}
        if os.path.exists(mp):
            md = pd.read_csv(mp)
            for _, x in md.iterrows():
                em[(x["id"], x["representation"])] = x.get("EM")
        for _id, reps in by_id.items():
            st = reps.get("structured")
            if not st:
                continue
            em_st = em.get((_id, "structured"))
            for other in ("semi-structured", "unstructured_full"):
                ot = reps.get(other)
                if not ot:
                    continue
                ev_div = _layer_div(st, ot, "prompt_evidence_mass_layer")
                if np.isnan(ev_div):
                    continue
                en_div = _layer_div(st, ot, "prompt_entropy_layer")
                em_ot = em.get((_id, other))
                flip = (int(em_st == 1 and em_ot == 0)
                        if em_st is not None and em_ot is not None else None)
                rows.append({"split": s, "id": _id, "contrast": f"structured~{other}",
                             "ev_div": ev_div, "ent_div": en_div,
                             "EM_structured": em_st, "EM_other": em_ot,
                             "flip_struct_to_other": flip})
    return pd.DataFrame(rows)


def write_synthesis_md(master: pd.DataFrame, pool: pd.DataFrame, div: pd.DataFrame) -> None:
    lines = ["# Cross-split synthesis — tokenization × attention × format invariance\n"]
    lines.append("## Pooled representation comparison (all splits)\n")
    lines.append(pool.to_markdown(floatfmt=".4f"))
    lines.append("\n\n## Format-divergence (structured vs re-serialized attention)\n")
    if not div.empty:
        agg = div.groupby("contrast").agg(
            n=("id", "count"), ev_div=("ev_div", "mean"), ent_div=("ent_div", "mean"),
            flips=("flip_struct_to_other", "sum"))
        lines.append(agg.to_markdown(floatfmt=".4f"))
        # divergence vs flip correlation, per contrast
        for contrast, d0 in div.groupby("contrast"):
            d = d0.dropna(subset=["flip_struct_to_other", "ev_div"])
            if len(d) > 10 and d["flip_struct_to_other"].nunique() > 1:
                r = d[["ev_div", "flip_struct_to_other"]].corr().iloc[0, 1]
                mean_flip = d[d.flip_struct_to_other == 1]["ev_div"].mean()
                mean_keep = d[d.flip_struct_to_other == 0]["ev_div"].mean()
                lines.append(f"\n\n**{contrast} — divergence ↔ format-induced failure:** "
                             f"corr(ev_div, flip) = {r:+.3f}; "
                             f"mean ev_div for flips (struct✓→other✗) = {mean_flip:.4f} vs kept = {mean_keep:.4f}. "
                             f"A debiasing method should drive ev_div → 0 (format-invariant attention).")
    lines.append("\n")
    (OUT / "SYNTHESIS.md").write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_merged()
    if df.empty:
        print("No merged.csv found under analysis_outputs/*/report/.")
        return
    master = master_summary(df)
    master.to_csv(OUT / "master_summary.csv")
    pool = pooled(df)
    pool.to_csv(OUT / "pooled_by_representation.csv")
    div = format_divergence()
    div.to_csv(OUT / "format_divergence.csv", index=False)
    write_synthesis_md(master, pool, div)
    print("=== Pooled by representation ===")
    print(pool.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nWrote synthesis -> {OUT}")
    if not div.empty:
        for contrast, d0 in div.groupby("contrast"):
            print(f"format_divergence [{contrast}]: {len(d0)} paired samples; "
                  f"mean ev_div={d0['ev_div'].mean():.4f}")


if __name__ == "__main__":
    run()
