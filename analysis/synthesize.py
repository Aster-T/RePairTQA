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
REPS = ["structured", "semi-structured", "unstructured"]
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
    "EM", "PM", "total_tokens", "length_inflation_ratio", "fertility_cell",
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
def format_divergence() -> pd.DataFrame:
    """Per-sample distance between structured vs semi-structured attention profiles.

    Uses the per-layer profiles captured in attn.jsonl:
      - prompt_evidence_mass_layer  : [n_layers]  (attention mass on the answer span)
      - prompt_entropy_layer        : [n_layers]  (dispersion of attention)
    Divergence = L2 distance between the structured and semi 64-d layer vectors.
    Joined with EM to test: do format-divergent samples fail more under shift?
    """
    rows = []
    for s in SPLITS:
        recs = _read_jsonl(f"analysis_outputs/{s}/attn.jsonl")
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
            st, se = reps.get("structured"), reps.get("semi-structured")
            if not st or not se:
                continue
            ev_st = np.array(st.get("prompt_evidence_mass_layer") or [])
            ev_se = np.array(se.get("prompt_evidence_mass_layer") or [])
            en_st = np.array(st.get("prompt_entropy_layer") or [])
            en_se = np.array(se.get("prompt_entropy_layer") or [])
            if ev_st.shape != ev_se.shape or ev_st.size == 0:
                continue
            ev_div = float(np.linalg.norm(ev_st - ev_se))
            en_div = float(np.linalg.norm(en_st - en_se)) if en_st.shape == en_se.shape and en_st.size else np.nan
            em_st, em_se = em.get((_id, "structured")), em.get((_id, "semi-structured"))
            flip = None
            if em_st is not None and em_se is not None:
                flip = int(em_st == 1 and em_se == 0)  # structured-correct, semi-wrong
            rows.append({"split": s, "id": _id, "ev_div": ev_div, "ent_div": en_div,
                         "EM_structured": em_st, "EM_semi": em_se, "flip_struct_to_semi": flip})
    return pd.DataFrame(rows)


def write_synthesis_md(master: pd.DataFrame, pool: pd.DataFrame, div: pd.DataFrame) -> None:
    lines = ["# Cross-split synthesis — tokenization × attention × format invariance\n"]
    lines.append("## Pooled representation comparison (all splits)\n")
    lines.append(pool.to_markdown(floatfmt=".4f"))
    lines.append("\n\n## Format-divergence (structured vs semi attention, per split)\n")
    if not div.empty:
        agg = div.groupby("split").agg(
            n=("id", "count"), ev_div=("ev_div", "mean"), ent_div=("ent_div", "mean"),
            flips=("flip_struct_to_semi", "sum"))
        lines.append(agg.to_markdown(floatfmt=".4f"))
        # divergence vs flip correlation
        d = div.dropna(subset=["flip_struct_to_semi", "ev_div"])
        if len(d) > 10 and d["flip_struct_to_semi"].nunique() > 1:
            r = d[["ev_div", "flip_struct_to_semi"]].corr().iloc[0, 1]
            mean_flip = d[d.flip_struct_to_semi == 1]["ev_div"].mean()
            mean_keep = d[d.flip_struct_to_semi == 0]["ev_div"].mean()
            lines.append(f"\n\n**Divergence ↔ format-induced failure:** corr(ev_div, flip) = {r:+.3f}; "
                         f"mean ev_div for flips (struct✓→semi✗) = {mean_flip:.4f} vs kept = {mean_keep:.4f}. "
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
        print(f"format_divergence: {len(div)} paired samples; mean ev_div="
              f"{div['ev_div'].mean():.4f}")


if __name__ == "__main__":
    run()
