#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bake the RePairTQA × Qwen3 analysis outputs into a single dashboard payload.

Reads (read-only):
  analysis_outputs/<split>/report/merged.csv            per (id, representation)
  analysis_outputs/synthesis/format_divergence.csv      per-sample structured~other divergence

Writes:
  <out>/data.json     the payload (pooled / by_split / divergence / meta)
  <out>/index.html    template.html with the payload inlined (works on file://)

Run from the repo root:  .venv/bin/python .claude/skills/results-dashboard/build_data.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import pandas as pd

REPS = ["structured", "semi-structured", "unstructured", "unstructured_full"]
SPLITS = ["bird_S1", "bird_S3", "bird_S4", "bird_S5",
          "tableeval_S2", "mmqa_M1", "mmqa_M2"]
# metrics surfaced to the UI (kept if present in merged.csv)
METRICS = ["EM", "PM", "judge", "prompt_evidence_mass_overall",
           "prompt_entropy_overall", "gold_evidence_mass_overall",
           "length_inflation_ratio", "structural_token_share",
           "number_frag_mean", "total_tokens", "evidence_rel_pos"]

HERE = Path(__file__).resolve().parent


def _rep_order(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["__r"] = df["representation"].map({r: i for i, r in enumerate(REPS)})
    df["__s"] = df["split"].map({s: i for i, s in enumerate(SPLITS)}) if "split" in df else 0
    return df.sort_values(["__s", "__r"]).drop(columns=[c for c in ["__r", "__s"] if c in df])


def load_merged(root: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(glob.glob(str(root / "analysis_outputs" / "*" / "report" / "merged.csv"))):
        split = Path(p).parent.parent.name
        df = pd.read_csv(p)
        if "split" not in df.columns:
            df["split"] = split
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def agg(df: pd.DataFrame, keys: list) -> list:
    cols = [m for m in METRICS if m in df.columns]
    g = df.groupby(keys)[cols].mean(numeric_only=True).reset_index()
    # sample counts per group
    n = df.groupby(keys).size().reset_index(name="n")
    g = g.merge(n, on=keys)
    g = _rep_order(g)
    recs = []
    for _, r in g.iterrows():
        rec = {k: r[k] for k in keys}
        rec["n"] = int(r["n"])
        for m in cols:
            v = r[m]
            rec[m] = (None if pd.isna(v) else round(float(v), 6))
        recs.append(rec)
    return recs


def load_divergence(root: Path) -> list:
    p = root / "analysis_outputs" / "synthesis" / "format_divergence.csv"
    if not p.exists():
        return []
    d = pd.read_csv(p)
    if d.empty or "contrast" not in d.columns:
        return []
    out = []
    for contrast, g in d.groupby("contrast"):
        flips = g.dropna(subset=["flip_struct_to_other", "ev_div"]) if "flip_struct_to_other" in g else g.iloc[0:0]
        mean_flip = flips[flips["flip_struct_to_other"] == 1]["ev_div"].mean() if len(flips) else float("nan")
        mean_keep = flips[flips["flip_struct_to_other"] == 0]["ev_div"].mean() if len(flips) else float("nan")
        out.append({
            "contrast": contrast,
            "n": int(len(g)),
            "ev_div": round(float(g["ev_div"].mean()), 6),
            "ent_div": round(float(g["ent_div"].mean()), 6) if "ent_div" in g and g["ent_div"].notna().any() else None,
            "flips": int(g["flip_struct_to_other"].sum()) if "flip_struct_to_other" in g else None,
            "ev_div_flip": None if pd.isna(mean_flip) else round(float(mean_flip), 6),
            "ev_div_keep": None if pd.isna(mean_keep) else round(float(mean_keep), 6),
        })
    return out


def build(root: Path, out: Path) -> dict:
    merged = load_merged(root)
    if merged.empty:
        raise SystemExit(f"No merged.csv found under {root}/analysis_outputs/*/report/. "
                         "Run the analysis pipeline (and analysis.finalize_full) first.")
    reps_present = [r for r in REPS if r in merged["representation"].unique()]
    splits_present = [s for s in SPLITS if s in merged["split"].unique()]
    payload = {
        "meta": {
            "reps": reps_present,
            "splits": splits_present,
            "has_judge": bool("judge" in merged.columns and merged["judge"].notna().any()),
            "has_full": bool("unstructured_full" in reps_present),
            "n_rows": int(len(merged)),
            "metrics_help": {
                "EM": "Exact match after normalization (strict).",
                "PM": "Partial / substring match (either direction).",
                "judge": "LLM-as-judge semantic correctness (lenient; the headline).",
                "prompt_evidence_mass_overall": "Attention mass on answer-evidence tokens (higher = on-target).",
                "prompt_entropy_overall": "Attention dispersion (higher = more diffuse).",
                "length_inflation_ratio": "Tokens vs the structured baseline.",
                "structural_token_share": "Fraction of tokens that are markdown structure (| --- ).",
                "ev_div": "L2 distance of structured-vs-other per-layer evidence attention; the debiasing target (lower = format-invariant).",
            },
        },
        "pooled": agg(merged, ["representation"]),
        "by_split": agg(merged, ["split", "representation"]),
        "divergence": load_divergence(root),
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # inline into index.html (escape </ to keep the <script> block valid)
    tmpl = (HERE / "template.html").read_text(encoding="utf-8")
    blob = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    html = tmpl.replace("/*__DASHBOARD_DATA__*/", blob, 1)
    (out / "index.html").write_text(html, encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Bake analysis outputs into the dashboard.")
    ap.add_argument("--root", default=".", help="repo root (default: cwd)")
    ap.add_argument("--out", default="dashboard", help="output dir (default: ./dashboard)")
    args = ap.parse_args()
    payload = build(Path(args.root).resolve(), Path(args.out).resolve())
    m = payload["meta"]
    print(f"reps={m['reps']}")
    print(f"splits={len(m['splits'])}  has_judge={m['has_judge']}  has_full={m['has_full']}")
    print(f"pooled rows={len(payload['pooled'])}  by_split rows={len(payload['by_split'])}  "
          f"divergence contrasts={len(payload['divergence'])}")
    print(f"wrote {args.out}/data.json and {args.out}/index.html")
    if not (Path(args.out) / "vendor" / "echarts.min.js").exists():
        print("[!] vendor/echarts.min.js missing — fetch it once (see SKILL.md):")
        print("    curl -L -o dashboard/vendor/echarts.min.js "
              "https://registry.npmmirror.com/echarts/5.5.1/files/dist/echarts.min.js")


if __name__ == "__main__":
    main()
