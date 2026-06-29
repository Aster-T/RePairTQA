#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4 (GPU): causal activation patching (interchange intervention).

For pairs where STRUCTURED is correct but SEMI is wrong, we cache structured's
residual-stream vector at its decision-token anchor for each layer, then run the
SEMI forward pass while OVERWRITING the layer-L residual at semi's anchor with the
cached structured vector, and check whether semi's next-token decoding flips to the
gold answer. Sweeping L localizes the layer(s) that CARRY the format effect:
the layer where patching recovers the answer is the causal site, vs ~0 at controls.

    MODEL_PATH=/home/amax/models/Qwen3-32B \
      .venv/bin/python -m analysis.canonicalization.patch --split bird_S1 --max_pairs 40
"""
from __future__ import annotations
import argparse, glob, json, os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis import config
from analysis.evidence_span import atomic_answer_values

OUT = config.RUNS_ROOT / config.MODEL_NAME / "canonicalization"


def load_model():
    tok = AutoTokenizer.from_pretrained(config.MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    return tok, model


def _baseline_em(model_name, split):
    """id -> (em_struct, em_semi) from baseline_outputs, to pick struct✓ semi✗ pairs."""
    out = {}
    for rep in ("structured", "semi-structured"):
        p = config.RUNS_ROOT / model_name / "baseline_outputs" / split / f"{rep}.jsonl"
        if not p.exists():
            return {}
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.setdefault(str(r["id"]), {})[rep] = float(r.get("EM") or 0)
    return out


@torch.no_grad()
def _hidden_at_last(model, ids):
    out = model(input_ids=ids, use_cache=False, output_hidden_states=True)
    return [h[0, -1, :].detach().clone() for h in out.hidden_states]  # list L+1 of [H]


@torch.no_grad()
def _next_token(model, ids):
    return int(model(input_ids=ids, use_cache=False).logits[0, -1, :].argmax().item())


def _patch_layer_decode(model, ids, layer_idx, vec):
    """Run forward overwriting decoder-block `layer_idx` output at the last position
    with `vec`; return greedy next-token id."""
    block = model.model.layers[layer_idx]

    def hook(_mod, _inp, output):
        hs = output[0] if isinstance(output, tuple) else output
        hs[:, -1, :] = vec.to(hs.dtype).to(hs.device)
        return (hs, *output[1:]) if isinstance(output, tuple) else hs

    h = block.register_forward_hook(hook)
    try:
        with torch.no_grad():
            tid = int(model(input_ids=ids, use_cache=False).logits[0, -1, :].argmax().item())
    finally:
        h.remove()
    return tid


def run(split: str, max_pairs: int, max_l: int):
    tok, model = load_model()
    em = _baseline_em(config.MODEL_NAME, split)
    data = {str(s.get("id_")): s for s in json.load(open(f"pipeline_results/{split}/example_verbalized.json"))}
    n_layers = model.config.num_hidden_layers
    # eligible: structured correct, semi wrong
    elig = [i for i, e in em.items() if e.get("structured", 0) == 1 and e.get("semi-structured", 0) == 0 and i in data]
    elig = elig[:max_pairs]
    print(f"[{split}] {len(elig)} struct✓/semi✗ pairs to patch")

    recover = np.zeros(n_layers + 1)
    counted = 0
    for sid in elig:
        s = data[sid]
        try:
            ps, pm = config.prompt_for(s, "structured"), config.prompt_for(s, "semi-structured")
        except Exception:
            continue
        ids_s = tok(ps, return_tensors="pt", truncation=True, max_length=max_l, add_special_tokens=False)["input_ids"].to(model.device)
        ids_m = tok(pm, return_tensors="pt", truncation=True, max_length=max_l, add_special_tokens=False)["input_ids"].to(model.device)
        gold = atomic_answer_values(s.get("answer"))
        gid = tok(" ".join(gold), add_special_tokens=False)["input_ids"]
        if not gid:
            continue
        gid0 = gid[0]
        cached = _hidden_at_last(model, ids_s)        # struct hidden per layer
        counted += 1
        for L in range(n_layers):                     # patch block L (hidden_states[L+1])
            tid = _patch_layer_decode(model, ids_m, L, cached[L + 1])
            recover[L + 1] += int(tid == gid0)
    if counted:
        recover /= counted
    OUT.mkdir(parents=True, exist_ok=True)
    import csv
    with open(OUT / f"patch_{split}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["layer", "recovery_rate"])
        for L in range(n_layers + 1):
            w.writerow([L, f"{recover[L]:.4f}"])
    best = int(np.argmax(recover))
    print(f"[{split}] n={counted}  best recovery {recover[best]:.3f} @ layer {best}  -> {OUT}/patch_{split}.csv")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    p.add_argument("--max_pairs", type=int, default=40)
    p.add_argument("--max_len", type=int, default=2048)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(a.split, a.max_pairs, a.max_len)
