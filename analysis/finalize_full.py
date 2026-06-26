#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebuild each split's report/merged.csv to include a FOURTH representation,
``unstructured_full`` (every row verbalized, semi-structured config untouched),
and attach an LLM-as-judge accuracy column for every representation.

Inputs per split (all already produced upstream):
  analysis_outputs/<s>/tok.jsonl                 structured/semi/unstructured tokenizer metrics
  analysis_outputs/<s>/attn.jsonl                ditto attention
  analysis_outputs/<s>/tok_unstr_full.jsonl      unstructured_full tokenizer metrics (HF phase)
  analysis_outputs/<s>/attn_unstr_full.jsonl     unstructured_full attention (HF phase)
  baseline_outputs/<s>/<rep>.jsonl               s5 EM/PM per rep (incl. unstructured_full.jsonl)
  evaluation_outputs/<s>/<rep>_judged.jsonl      s7 gpt_eval CORRECT/INCORRECT per rep

Output: analysis_outputs/<s>/report/merged.csv  (4 reps; adds a `judge` 0/1 column)

This keeps the original 3-rep artifacts reproducible (it re-derives them the same
way correlate.py did) and simply adds the full view + judge signal on top.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import pandas as pd

from . import correlate

SPLITS = ["bird_S1", "bird_S3", "bird_S4", "bird_S5",
          "tableeval_S2", "mmqa_M1", "mmqa_M2"]
REPS3 = ["structured", "semi-structured", "unstructured"]


def _judge_map(path: str) -> Dict[str, int]:
    """id -> 1 if judged CORRECT else 0, from an s7 *_judged.jsonl file."""
    out: Dict[str, int] = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            lab = str(r.get("gpt_eval", "")).strip().upper()
            out[r.get("id")] = 1 if lab == "CORRECT" else 0
    return out


def _merged_for_split(s: str) -> Optional[pd.DataFrame]:
    tok3 = f"analysis_outputs/{s}/tok.jsonl"
    attn3 = f"analysis_outputs/{s}/attn.jsonl"
    if not os.path.exists(tok3):
        print(f"[SKIP] {s}: no tok.jsonl")
        return None

    acc3 = {r: f"baseline_outputs/{s}/{r}.jsonl" for r in REPS3
            if os.path.exists(f"baseline_outputs/{s}/{r}.jsonl")}
    m3 = correlate.build_merged(tok3, attn3, acc3)

    frames = [m3]
    tokf = f"analysis_outputs/{s}/tok_unstr_full.jsonl"
    attnf = f"analysis_outputs/{s}/attn_unstr_full.jsonl"
    accf = f"baseline_outputs/{s}/unstructured_full.jsonl"
    if os.path.exists(tokf):
        mf = correlate.build_merged(
            tokf, attnf if os.path.exists(attnf) else None,
            {"unstructured_full": accf} if os.path.exists(accf) else None)
        frames.append(mf)
    else:
        print(f"[WARN] {s}: no tok_unstr_full.jsonl (run the HF attention phase first)")

    # align columns across frames so concat doesn't warn on all-NA dtype inference
    all_cols = list(dict.fromkeys(c for f in frames for c in f.columns))
    frames = [f.reindex(columns=all_cols) for f in frames if not f.empty]
    merged = pd.concat(frames, ignore_index=True)

    # Recompute length_inflation_ratio uniformly against the structured token count
    # per id. The unstructured_full tokenization ran rep-only (no structured in its
    # file), so its inflation column is otherwise just total_tokens; ids are unique
    # within a split, so a structured-token lookup fixes all reps consistently.
    if "total_tokens" in merged.columns and (merged["representation"] == "structured").any():
        base = (merged[merged["representation"] == "structured"]
                .set_index("id")["total_tokens"].to_dict())
        merged["length_inflation_ratio"] = [
            (tt / base[i]) if (i in base and base[i]) else None
            for i, tt in zip(merged["id"], merged["total_tokens"])
        ]

    # attach LLM-judge accuracy per (id, representation)
    judges: Dict[str, Dict[str, int]] = {}
    for rep in REPS3 + ["unstructured_full"]:
        judges[rep] = _judge_map(f"evaluation_outputs/{s}/{rep}_judged.jsonl")
    merged["judge"] = [
        judges.get(rep, {}).get(_id)
        for _id, rep in zip(merged["id"], merged["representation"])
    ]
    return merged


def main() -> None:
    for s in SPLITS:
        merged = _merged_for_split(s)
        if merged is None:
            continue
        out_dir = f"analysis_outputs/{s}/report"
        os.makedirs(out_dir, exist_ok=True)
        merged.to_csv(os.path.join(out_dir, "merged.csv"), index=False)
        # quick per-rep EM / judge tally
        g = merged.groupby("representation").agg(
            n=("id", "count"), EM=("EM", "mean"),
            judge=("judge", "mean")).reindex(
            [r for r in REPS3 + ["unstructured_full"] if r in merged["representation"].unique()])
        print(f"\n=== {s} ===")
        print(g.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
