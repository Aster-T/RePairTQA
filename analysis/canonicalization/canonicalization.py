#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phases 1-3 (CPU): from the saved hidden states, quantify WHERE table format is
washed out across the 65 layers (embedding + 64 blocks).

  Phase 1  convergence curve  r(L) = d_within / d_across
           d_within = standardized distance between the two FORMATS of the SAME
                      content (paired); d_across = distance between DIFFERENT
                      content, same format (baseline). r->0 => format removed.
           Reported with standardized-L2 AND mean-centered cosine (no metric artifact).
  Phase 2  per-layer linear probe "structured vs semi" (GroupKFold by id) -> AUC.
           AUC->0.5 layer = format linearly undecodable.
  Phase 3  logit-lens "answer lock depth": per layer, fraction of samples whose
           top-1 token already matches the gold first token, for each format.

    .venv/bin/python -m analysis.canonicalization.canonicalization --anchor last
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from analysis import config

ROOT = config.RUNS_ROOT / config.MODEL_NAME / "canonicalization"


def load(anchor: str, split: str | None):
    files = ([ROOT / f"hidden_{split}.npz"] if split
             else sorted(Path(ROOT).glob("hidden_*.npz")))
    S, M, ids, hs, hm = [], [], [], [], []
    for f in files:
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        S.append(d[f"{anchor}_struct"].astype(np.float32))   # [n, L+1, H]
        M.append(d[f"{anchor}_semi"].astype(np.float32))
        ids.append(d["ids"]); hs.append(d["hit_struct"]); hm.append(d["hit_semi"])
    if not S:
        raise SystemExit(f"no hidden_*.npz under {ROOT} — run capture_hidden first")
    return (np.concatenate(S), np.concatenate(M), np.concatenate(ids),
            np.concatenate(hs), np.concatenate(hm))


def _pairdist(a, b, kind):                       # a,b: [n, H] already standardized/centered
    if kind == "l2":
        return np.linalg.norm(a - b, axis=1)
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return 1 - (an * bn).sum(1)                  # cosine distance


def convergence(struct, semi, kind, rng):
    n, Lp1, H = struct.shape
    r, dw, da = [], [], []
    for L in range(Lp1):
        xs, xm = struct[:, L, :], semi[:, L, :]
        if kind == "l2":
            sc = StandardScaler().fit(np.vstack([xs, xm]))
            zs, zm = sc.transform(xs), sc.transform(xm)
        else:                                    # cosine: mean-center on pooled cloud
            mu = np.vstack([xs, xm]).mean(0)
            zs, zm = xs - mu, xm - mu
        d_within = _pairdist(zs, zm, kind).mean()
        # baseline: different content, same format (mix both reps)
        pool = np.vstack([zs, zm]); idx = rng.permutation(len(pool))
        a = pool; b = pool[idx]
        same = idx == np.arange(len(pool))       # avoid accidental self-pairs
        d_across = _pairdist(a[~same], b[~same], kind).mean()
        dw.append(d_within); da.append(d_across); r.append(d_within / (d_across + 1e-9))
    return np.array(dw), np.array(da), np.array(r)


def probe(struct, semi, ids):
    n, Lp1, H = struct.shape
    y = np.r_[np.zeros(n), np.ones(n)]
    groups = np.r_[ids, ids]
    aucs = []
    gkf = GroupKFold(n_splits=5)
    for L in range(Lp1):
        X = np.vstack([struct[:, L, :], semi[:, L, :]])
        fold = []
        for tr, te in gkf.split(X, y, groups):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
            fold.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
        aucs.append(float(np.mean(fold)))
    return np.array(aucs)


def run(anchor: str, split: str | None):
    struct, semi, ids, hit_s, hit_m = load(anchor, split)
    rng = np.random.default_rng(0)
    n, Lp1, H = struct.shape
    print(f"loaded {n} paired samples, {Lp1} layers (incl. embedding), H={H}, anchor={anchor}")

    dw_l2, da_l2, r_l2 = convergence(struct, semi, "l2", rng)
    _, _, r_cos = convergence(struct, semi, "cos", rng)
    auc = probe(struct, semi, ids)
    lens_s = hit_s.mean(0); lens_m = hit_m.mean(0)      # fraction matching gold, per layer

    out = ROOT / "report"; out.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out / f"canonicalization_{anchor}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "d_within_l2", "d_across_l2", "r_l2", "r_cos",
                    "probe_auc", "lens_hit_struct", "lens_hit_semi"])
        for L in range(Lp1):
            w.writerow([L, f"{dw_l2[L]:.4f}", f"{da_l2[L]:.4f}", f"{r_l2[L]:.4f}",
                        f"{r_cos[L]:.4f}", f"{auc[L]:.4f}", f"{lens_s[L]:.4f}", f"{lens_m[L]:.4f}"])

    # candidate canonicalization layer = min r_l2 (most converged)
    Lstar = int(np.argmin(r_l2))
    auc_floor = int(np.argmin(np.abs(auc - 0.5)))
    print(f"\nr_l2: min={r_l2.min():.3f} @ layer {Lstar}  (r near 0 => format washed out)")
    print(f"     r_l2 never < 0.8?  -> {'YES (persistent separation)' if r_l2.min()>=0.8 else 'no'}")
    print(f"probe AUC: min={auc.min():.3f} @ layer {auc_floor}  (0.5 => format undecodable)")
    print(f"logit-lens gold-hit: struct peaks {lens_s.max():.2f}@L{int(lens_s.argmax())}, "
          f"semi peaks {lens_m.max():.2f}@L{int(lens_m.argmax())}")
    print(f"\nwrote {out}/canonicalization_{anchor}.csv")
    _plots(r_l2, r_cos, auc, lens_s, lens_m, out, anchor)


def _plots(r_l2, r_cos, auc, lens_s, lens_m, out, anchor):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        return
    L = np.arange(len(r_l2))
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    ax[0].plot(L, r_l2, label="r (std-L2)"); ax[0].plot(L, r_cos, label="r (cosine)")
    ax[0].axhline(1, ls=":", c="gray"); ax[0].set_title("convergence r(L)=within/across"); ax[0].set_xlabel("layer"); ax[0].legend()
    ax[1].plot(L, auc); ax[1].axhline(0.5, ls=":", c="gray"); ax[1].set_title("format probe AUC"); ax[1].set_xlabel("layer")
    ax[2].plot(L, lens_s, label="structured"); ax[2].plot(L, lens_m, label="semi")
    ax[2].set_title("logit-lens gold-hit rate"); ax[2].set_xlabel("layer"); ax[2].legend()
    fig.tight_layout(); fig.savefig(out / f"canonicalization_{anchor}.png", dpi=130)
    print(f"wrote {out}/canonicalization_{anchor}.png")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--anchor", choices=["last", "mean"], default="last")
    p.add_argument("--split", default=None)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(a.anchor, a.split)
