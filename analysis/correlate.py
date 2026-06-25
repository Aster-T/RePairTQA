#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase E: join tokenization + attention + accuracy, then quantify the mechanism.

The claim under test: under table-sequence semantic invariance (answer identical
across the three representations), shifting structured -> semi/unstructured
changes tokenization (length inflation, number fragmentation) and attention
(evidence mass down, entropy up), and these mediate the accuracy drop.

Inputs
  --tok   JSONL from tokenization_metrics.py  (per id x representation)
  --attn  JSONL from attention_capture.py     (per id x representation; optional)
  --acc   one or more rep=path pairs to s5 EM/PM JSONL, e.g.
          --acc structured=out/struct.jsonl semi-structured=out/semi.jsonl ...
Outputs (in --out_dir): merged.csv, summary_by_representation.csv,
  regression.txt, and bar/scatter plots.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import pandas as pd

from . import config


def _read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_accuracy(acc_map: Dict[str, str]) -> pd.DataFrame:
    frames = []
    for rep, path in acc_map.items():
        df = pd.DataFrame(_read_jsonl(path))
        if df.empty:
            continue
        df = df.rename(columns={"id": "id"})
        df["representation"] = rep
        frames.append(df[["id", "representation", "EM", "PM"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["id", "representation", "EM", "PM"])


# attention scalar columns kept for the merged frame
_ATTN_COLS = [
    "prompt_evidence_mass_overall", "prompt_entropy_overall",
    "gold_evidence_mass_overall", "gold_entropy_overall",
]


def build_merged(tok_path: str, attn_path: Optional[str],
                 acc_map: Optional[Dict[str, str]]) -> pd.DataFrame:
    tok = pd.DataFrame(_read_jsonl(tok_path))
    merged = tok
    if attn_path and os.path.exists(attn_path):
        attn = pd.DataFrame(_read_jsonl(attn_path))
        keep = ["id", "representation"] + [c for c in _ATTN_COLS if c in attn.columns]
        merged = merged.merge(attn[keep], on=["id", "representation"], how="left")
    if acc_map:
        acc = load_accuracy(acc_map)
        merged = merged.merge(acc, on=["id", "representation"], how="left")
    return merged


def summary_by_representation(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in [
        "EM", "PM", "total_tokens", "length_inflation_ratio", "fertility_cell",
        "structural_token_share", "number_frag_mean", "evidence_token_count",
        "evidence_rel_pos", "prompt_evidence_mass_overall", "prompt_entropy_overall",
        "gold_evidence_mass_overall",
    ] if c in df.columns]
    g = df.groupby("representation")[metrics].mean(numeric_only=True)
    # order rows as in REPRESENTATIONS
    return g.reindex([r for r in config.REPRESENTATIONS if r in g.index])


def run_regression(df: pd.DataFrame, out_txt: str) -> None:
    if "EM" not in df.columns or df["EM"].notna().sum() < 8:
        with open(out_txt, "w") as f:
            f.write("Not enough EM labels to fit a regression.\n")
        return
    import statsmodels.formula.api as smf

    preds = [c for c in ["prompt_evidence_mass_overall", "prompt_entropy_overall",
                         "structural_token_share", "number_frag_mean",
                         "length_inflation_ratio"] if c in df.columns]
    d = df.dropna(subset=["EM"] + preds).copy()
    d["rep"] = d["representation"].astype("category")
    lines = []
    if len(d) >= 8 and preds:
        formula = "EM ~ " + " + ".join(preds) + " + C(rep)"
        try:
            model = smf.logit(formula, data=d).fit(disp=False)
            lines.append("=== Logistic regression: EM ~ metrics + representation ===")
            lines.append(str(model.summary()))
        except Exception as exc:  # separation / singular etc.
            lines.append(f"Logit failed ({exc}); falling back to correlations.")
    # point-biserial correlations EM vs each metric
    lines.append("\n=== Correlation of EM with each metric (Pearson) ===")
    for c in preds:
        try:
            r = d[["EM", c]].corr().iloc[0, 1]
            lines.append(f"  {c:<34} r = {r:+.3f}")
        except Exception:
            pass
    with open(out_txt, "w") as f:
        f.write("\n".join(lines) + "\n")


def invariance_contrast(df: pd.DataFrame, out_csv: str) -> None:
    """For ids where EM differs across representations, report the mean metric
    shift structured -> {semi, unstructured} (the mechanism, holding answer fixed)."""
    if "EM" not in df.columns:
        return
    piv = df.pivot_table(index="id", columns="representation", values="EM")
    flips = piv[piv.nunique(axis=1) > 1].index
    cols = [c for c in ["length_inflation_ratio", "number_frag_mean",
                        "evidence_token_count", "prompt_evidence_mass_overall",
                        "prompt_entropy_overall"] if c in df.columns]
    sub = df[df["id"].isin(flips)]
    if sub.empty:
        return
    sub.groupby("representation")[cols].mean(numeric_only=True).reindex(
        [r for r in config.REPRESENTATIONS if r in sub["representation"].unique()]
    ).to_csv(out_csv)


def make_plots(summary: pd.DataFrame, df: pd.DataFrame, out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    for col in [c for c in ["EM", "length_inflation_ratio",
                            "prompt_evidence_mass_overall", "prompt_entropy_overall",
                            "number_frag_mean"] if c in summary.columns]:
        ax = summary[col].plot(kind="bar", title=col, rot=20, figsize=(5, 3.2))
        ax.set_xlabel("")
        ax.figure.tight_layout()
        ax.figure.savefig(os.path.join(out_dir, f"bar_{col}.png"), dpi=130)
        ax.figure.clf()


def parse_acc(pairs: Optional[List[str]]) -> Dict[str, str]:
    out = {}
    for p in (pairs or []):
        if "=" in p:
            rep, path = p.split("=", 1)
            out[rep] = path
    return out


def run(tok: str, attn: Optional[str], acc: Optional[List[str]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = build_merged(tok, attn, parse_acc(acc))
    df.to_csv(os.path.join(out_dir, "merged.csv"), index=False)
    summ = summary_by_representation(df)
    summ.to_csv(os.path.join(out_dir, "summary_by_representation.csv"))
    print("\n=== Summary by representation ===")
    print(summ.to_string(float_format=lambda x: f"{x:.4f}"))
    run_regression(df, os.path.join(out_dir, "regression.txt"))
    invariance_contrast(df, os.path.join(out_dir, "invariance_contrast.csv"))
    make_plots(summ, df, out_dir)
    print(f"\nWrote merged.csv, summary, regression.txt, plots -> {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Correlate tokenization/attention with accuracy.")
    p.add_argument("--tok", required=True, help="tokenization_metrics JSONL")
    p.add_argument("--attn", default=None, help="attention_capture JSONL (optional)")
    p.add_argument("--acc", nargs="*", default=None,
                   help="rep=path pairs to s5 EM/PM JSONL")
    p.add_argument("--out_dir", default=str(config.OUT_DIR / "report"))
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(a.tok, a.attn, a.acc, a.out_dir)
