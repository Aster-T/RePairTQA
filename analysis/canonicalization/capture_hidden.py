#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 0 (GPU): capture residual-stream hidden states for paired structured/semi
prompts of the SAME content, to study WHERE the model washes out table format.

For each (id, representation) we save, at the **decision-token anchor** (last
prompt token) AND a mean-pool-over-prompt anchor, the hidden state of all 65
"layers" (embedding + 64 decoder blocks; Qwen3-32B hidden_size=5120). We also run
a **logit lens** (final norm + lm_head applied to each layer's hidden) and record
whether the layer's top-1 token already matches the gold answer's first token.

NOT an OOM risk: hidden states are [batch, seq, 5120] per layer, unlike
output_attentions. Saved as fp16 under runs/<MODEL>/canonicalization/ (git-ignored).

    MODEL_PATH=/home/amax/models/Qwen3-32B \
      .venv/bin/python -m analysis.canonicalization.capture_hidden --split bird_S1
"""
from __future__ import annotations
import argparse, glob, json, os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis import config
from analysis.evidence_span import atomic_answer_values

OUT_ROOT = config.RUNS_ROOT / config.MODEL_NAME / "canonicalization"
REPS = ["structured", "semi-structured"]


def load_model():
    tok = AutoTokenizer.from_pretrained(config.MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, output_hidden_states=True)
    model.eval()
    return tok, model


def _gold_first_token_id(tok, sample) -> int | None:
    vals = atomic_answer_values(sample.get("answer"))
    g = " ".join(vals).strip() if vals else ""
    if not g:
        return None
    ids = tok(g, add_special_tokens=False)["input_ids"]
    return ids[0] if ids else None


@torch.no_grad()
def capture_one(tok, model, prompt: str, gold_tok_id: int | None, max_l: int):
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=max_l,
              add_special_tokens=False)["input_ids"].to(model.device)
    out = model(input_ids=ids, use_cache=False)
    hs = out.hidden_states                      # tuple length n_layers+1, each [1, seq, H]
    last = torch.stack([h[0, -1, :].float().cpu() for h in hs])      # [L+1, H]
    mean = torch.stack([h[0].float().mean(0).cpu() for h in hs])     # [L+1, H]
    # logit lens: lm_head(norm(h_L)) at the last token -> top1 token + gold hit
    norm = model.model.norm
    lm = model.lm_head
    hits, top1 = [], []
    for h in hs:
        v = h[0, -1, :]
        logit = lm(norm(v.unsqueeze(0)))[0]      # [vocab]
        t = int(logit.argmax().item())
        top1.append(t)
        hits.append(int(gold_tok_id is not None and t == gold_tok_id))
    return last.numpy().astype(np.float16), mean.numpy().astype(np.float16), \
        np.array(top1, np.int32), np.array(hits, np.int8)


def run(split: str, max_l: int):
    tok, model = load_model()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    char_cap = max_l * 7
    f = f"pipeline_results/{split}/example_verbalized.json"
    data = json.load(open(f))
    ids, last_s, last_m, mean_s, mean_m, lens_s, lens_m, hit_s, hit_m = ([] for _ in range(9))
    for s in data:
        try:
            ps = config.prompt_for(s, "structured")
            pm = config.prompt_for(s, "semi-structured")
        except Exception:
            continue
        if len(ps) > char_cap or len(pm) > char_cap:
            continue
        if len(tok(ps, add_special_tokens=False)["input_ids"]) > max_l: continue
        if len(tok(pm, add_special_tokens=False)["input_ids"]) > max_l: continue
        g = _gold_first_token_id(tok, s)
        ls, ms, t1s, h1s = capture_one(tok, model, ps, g, max_l)
        lm_, mm_, t1m, h1m = capture_one(tok, model, pm, g, max_l)
        ids.append(str(s.get("id_")))
        last_s.append(ls); last_m.append(lm_); mean_s.append(ms); mean_m.append(mm_)
        lens_s.append(t1s); lens_m.append(t1m); hit_s.append(h1s); hit_m.append(h1m)
    out = OUT_ROOT / f"hidden_{split}.npz"
    np.savez_compressed(
        out, ids=np.array(ids), split=split,
        last_struct=np.array(last_s), last_semi=np.array(last_m),
        mean_struct=np.array(mean_s), mean_semi=np.array(mean_m),
        lens_struct=np.array(lens_s), lens_semi=np.array(lens_m),
        hit_struct=np.array(hit_s), hit_semi=np.array(hit_m))
    print(f"[{split}] saved {len(ids)} pairs -> {out}  ({out.stat().st_size/1e6:.0f} MB)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default=None, help="one split; default = all under pipeline_results/")
    p.add_argument("--max_len", type=int, default=2048)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    splits = [a.split] if a.split else [
        os.path.basename(os.path.dirname(f))
        for f in sorted(glob.glob("pipeline_results/*/example_verbalized.json"))
        if os.path.basename(os.path.dirname(f)) != "run_example"]
    for s in splits:
        run(s, a.max_len)
