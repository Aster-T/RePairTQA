#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase D: capture Qwen3-32B attention distributions across representations, on the
two RTX 4090s only, WITHOUT ever holding all 64 layers' attention matrices at once.

Why monkeypatching instead of output_attentions=True:
  output_attentions=True returns ALL 64 layers' [heads, L, L] matrices at once
  (~275GB fp32 at L=4096) -> instant OOM. Instead we patch the module-level
  `eager_attention_forward` so that, for each layer as it runs, we reduce the
  (already-computed) attention probabilities to a handful of per-head scalars and
  immediately drop the full matrix. Peak attention memory = ONE layer.

We read attention at two kinds of query positions:
  pass "prompt": input = prompt; query = last prompt token (what the model
                  looked at when deciding the first answer token).
  pass "gold":   input = prompt + GOLD answer; query = each gold-answer token
                  (clean evidence-localization / retrieval-head signal).

All metrics key off the evidence token span (analysis/evidence_span.py).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from . import config
from .evidence_span import evidence_token_indices

_STRUCTURAL_RE = re.compile(r"^[\s|\-]+$")


# --------------------------------------------------------------------------- #
# Capture context: set before each forward, read inside the patched attention.
# --------------------------------------------------------------------------- #
@dataclass
class CaptureCtx:
    query_positions: List[int]
    evidence_keys: torch.Tensor      # long tensor of key indices (on device)
    structural_keys: torch.Tensor    # long tensor of structural key indices
    n_layers: int
    # accumulators, shape [n_layers, n_heads]; filled lazily once n_heads known
    ev_mass: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None
    struct_mass: Optional[torch.Tensor] = None
    content_mass: Optional[torch.Tensor] = None
    seen_layers: set = field(default_factory=set)


_CTX: Optional[CaptureCtx] = None


def _reduce(layer_idx: int, attn_weights: torch.Tensor) -> None:
    """attn_weights: [batch=1, n_heads, q_len, kv_len]. Reduce to per-head scalars
    at the query positions of interest, accumulate, then let the tensor be freed."""
    ctx = _CTX
    if ctx is None:
        return
    n_heads = attn_weights.shape[1]
    if ctx.ev_mass is None:
        z = lambda: torch.zeros(ctx.n_layers, n_heads, dtype=torch.float32)
        ctx.ev_mass, ctx.entropy, ctx.struct_mass, ctx.content_mass = z(), z(), z(), z()

    a_all = attn_weights[0].float()  # [n_heads, q_len, kv_len]
    qpos = [p for p in ctx.query_positions if p < a_all.shape[1]]
    if not qpos:
        return
    a = a_all[:, qpos, :]            # [n_heads, n_q, kv_len]
    # device_map="auto" can place layers on different GPUs; align index tensors
    # to whatever device this layer's attention landed on.
    ev = ctx.evidence_keys.to(a.device)
    st = ctx.structural_keys.to(a.device)

    ev_mass = a.index_select(-1, ev).sum(-1) if ev.numel() else torch.zeros_like(a[..., 0])
    st_mass = a.index_select(-1, st).sum(-1) if st.numel() else torch.zeros_like(a[..., 0])
    ent = -(a.clamp_min(1e-12) * a.clamp_min(1e-12).log()).sum(-1)   # [n_heads, n_q]
    total = a.sum(-1)                                                # ~1.0

    ctx.ev_mass[layer_idx] += ev_mass.mean(-1).cpu()
    ctx.struct_mass[layer_idx] += st_mass.mean(-1).cpu()
    ctx.content_mass[layer_idx] += (total - st_mass).mean(-1).cpu()
    ctx.entropy[layer_idx] += ent.mean(-1).cpu()
    ctx.seen_layers.add(layer_idx)


# --------------------------------------------------------------------------- #
# Monkeypatch
# --------------------------------------------------------------------------- #
def install_patch():
    """Wrap Qwen3's module-level eager_attention_forward with a reduce-and-free
    shim. Returns the modeling module so the caller can restore if needed."""
    from transformers.models.qwen3 import modeling_qwen3 as mq

    if getattr(mq, "_repairtqa_patched", False):
        return mq
    orig = mq.eager_attention_forward

    def wrapped(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        attn_output, attn_weights = orig(
            module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs
        )
        try:
            _reduce(getattr(module, "layer_idx", 0), attn_weights)
        finally:
            pass
        # Drop the full matrix; we don't need it propagated/accumulated.
        return attn_output, None

    mq.eager_attention_forward = wrapped
    mq._repairtqa_patched = True
    mq._repairtqa_orig = orig
    return mq


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model(model_path: Optional[str] = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = model_path or config.MODEL_PATH
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",   # required for attention capture
        device_map="auto",             # split across the visible 4090s
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    install_patch()
    return model, tok, n_layers


_SYSTEM_PROMPT = "You are a helpful assistant for table question answering."


def build_input_text(tok, sample: dict, representation: str) -> str:
    """Chat-templated text the model actually sees at inference: same system+user
    messages as src/llm_infer_s5.py, Qwen3 thinking disabled (direct answers)."""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": config.prompt_for(sample, representation)},
    ]
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        # Older template without enable_thinking kwarg.
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _structural_key_indices(tokenizer, prompt: str, max_length: int) -> List[int]:
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False,
                    truncation=True, max_length=max_length)
    out = []
    for i, (ts, te) in enumerate(enc["offset_mapping"]):
        if ts != te and _STRUCTURAL_RE.match(prompt[ts:te] or ""):
            out.append(i)
    return out


@torch.no_grad()
def capture_sample(model, tok, n_layers, sample: dict, representation: str,
                   max_l: int) -> Dict[str, Any]:
    global _CTX
    prompt = build_input_text(tok, sample, representation)
    dev = next(model.parameters()).device

    ev = evidence_token_indices(tok, prompt, sample.get("answer"), max_l)
    struct_idx = _structural_key_indices(tok, prompt, max_l)

    prompt_ids = tok(prompt, return_tensors="pt", truncation=True,
                     max_length=max_l, add_special_tokens=False)["input_ids"]
    prompt_len = prompt_ids.shape[1]

    results: Dict[str, Any] = {
        "id": sample.get("id_"), "representation": representation,
        "prompt_len": prompt_len, "evidence_status": ev["status"],
        "n_evidence_tokens": len(ev["token_indices"]),
    }

    def run_pass(input_ids, query_positions, tag):
        global _CTX
        ev_keys = torch.tensor([k for k in ev["token_indices"] if k < input_ids.shape[1]],
                               dtype=torch.long, device=dev)
        st_keys = torch.tensor([k for k in struct_idx if k < input_ids.shape[1]],
                               dtype=torch.long, device=dev)
        _CTX = CaptureCtx(query_positions=query_positions, evidence_keys=ev_keys,
                          structural_keys=st_keys, n_layers=n_layers)
        model(input_ids=input_ids.to(dev), use_cache=False)
        ctx = _CTX
        n_q = max(1, len([p for p in query_positions if p < input_ids.shape[1]]))
        ev_mass = (ctx.ev_mass / n_q) if ctx.ev_mass is not None else torch.zeros(n_layers, 1)
        entropy = (ctx.entropy / n_q)
        struct = (ctx.struct_mass / n_q)
        content = (ctx.content_mass / n_q)
        ratio = struct / content.clamp_min(1e-9)
        _CTX = None
        return {
            f"{tag}_evidence_mass_heads": ev_mass.tolist(),     # [n_layers, n_heads]
            f"{tag}_evidence_mass_layer": ev_mass.mean(-1).tolist(),
            f"{tag}_entropy_layer": entropy.mean(-1).tolist(),
            f"{tag}_struct_ratio_layer": ratio.mean(-1).tolist(),
            f"{tag}_evidence_mass_overall": float(ev_mass.mean()),
            f"{tag}_entropy_overall": float(entropy.mean()),
        }

    # pass "prompt": query = last prompt token
    results.update(run_pass(prompt_ids, [prompt_len - 1], "prompt"))

    # pass "gold": teacher-force the GOLD answer, query = gold token positions
    from .evidence_span import atomic_answer_values
    gold_str = " ".join(atomic_answer_values(sample.get("answer"))) or ""
    if gold_str and ev["status"] != "missing":
        gold_ids = tok(gold_str, return_tensors="pt", add_special_tokens=False)["input_ids"]
        full = torch.cat([prompt_ids, gold_ids], dim=1)[:, :max_l]
        gpos = list(range(prompt_len, full.shape[1]))
        if gpos:
            results.update(run_pass(full, gpos, "gold"))
    return results


def run(input_json: str, output_jsonl: str, model_path: Optional[str] = None,
        max_l: Optional[int] = None, max_samples: Optional[int] = None) -> None:
    max_l = max_l or config.MAX_L
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if max_samples:
        data = data[:max_samples]

    model, tok, n_layers = load_model(model_path)
    skipped = 0
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for i, sample in enumerate(data):
            for rep in config.REPRESENTATIONS:
                try:
                    rec = capture_sample(model, tok, n_layers, sample, rep, max_l)
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                except Exception as e:  # OOM / per-sample failure: skip, keep the split alive
                    global _CTX
                    _CTX = None
                    torch.cuda.empty_cache()
                    skipped += 1
                    print(f"  [SKIP] {sample.get('id_')}/{rep}: {type(e).__name__}: {str(e)[:100]}")
            torch.cuda.empty_cache()
            print(f"[{i+1}/{len(data)}] {sample.get('id_')} done | "
                  f"peak={torch.cuda.max_memory_allocated()/1e9:.1f}GB")
    print(f"[attention] done: {len(data)} samples, {skipped} (sample,rep) skipped")


def micro_test(model_path: Optional[str] = None, l: int = 512) -> None:
    """Memory gate: 1 tiny forward, assert single-layer-resident attention."""
    torch.cuda.reset_peak_memory_stats()
    model, tok, n_layers = load_model(model_path)
    after_load = torch.cuda.max_memory_allocated() / 1e9
    text = "| a | b |\n| --- | --- |\n| 1 | 2 |\n" * (l // 16)
    ids = tok(text, return_tensors="pt", truncation=True, max_length=l,
              add_special_tokens=False)["input_ids"]
    dev = next(model.parameters()).device
    global _CTX
    _CTX = CaptureCtx([ids.shape[1] - 1],
                      torch.tensor([0], device=dev), torch.tensor([1], device=dev), n_layers)
    with torch.no_grad():
        model(input_ids=ids.to(dev), use_cache=False)
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"n_layers={n_layers} L={ids.shape[1]} | after_load={after_load:.1f}GB "
          f"peak={peak:.1f}GB delta={peak-after_load:.2f}GB (single-layer if small)")
    _CTX = None


def parse_args():
    p = argparse.ArgumentParser(description="Capture Qwen3-32B attention metrics.")
    p.add_argument("--input", help="Post-s4_1 JSON (raw_tables/verbalized_data).")
    p.add_argument("--output", help="Output JSONL of per-(sample,rep) attention metrics.")
    p.add_argument("--model_path", default=None)
    p.add_argument("--max_l", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--micro_test", action="store_true", help="Run the memory gate only.")
    p.add_argument("--micro_l", type=int, default=512)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.micro_test:
        micro_test(args.model_path, args.micro_l)
    else:
        run(args.input, args.output, args.model_path, args.max_l, args.max_samples)
