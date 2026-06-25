#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tokenization metrics across the paper's three representations, using the Qwen3
tokenizer (faithful to what the model sees, unlike the gpt-4o tiktoken default
hardcoded in src/llm_infer_s5.py).

For each sample x representation we measure how the SAME content is segmented:
length, table/text split, fertility, number fragmentation, structural-token
share, and where the answer "evidence" tokens land. These are the tokenizer-side
half of the "same content, different representation" mechanism.

CPU-only: needs just the tokenizer files, not the 65GB weights.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer

from . import config
from .evidence_span import evidence_token_indices

_STRUCTURAL_RE = re.compile(r"^[\s|\-]+$")   # markdown pipes, --- rules, whitespace
_DATE_RE = re.compile(r"\d{4}-\d{1,2}(-\d{1,2})?")


def is_number(s: str) -> bool:
    t = s.strip().lstrip("$").rstrip("%").replace(",", "")
    if t == "":
        return False
    try:
        float(t)
        return True
    except ValueError:
        return False


def is_date(s: str) -> bool:
    return bool(_DATE_RE.fullmatch(s.strip()))


_TOKENIZER_CACHE: Dict[str, Any] = {}


def get_tokenizer(path: Optional[str] = None):
    path = path or config.MODEL_PATH
    if path not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[path] = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    return _TOKENIZER_CACHE[path]


def _tables_for(sample: dict, representation: str) -> List[dict]:
    """Which table entries back this representation (for cell counting)."""
    if representation == "structured":
        raw = sample.get("raw_tables") or {}
        return raw.get("tables") or sample.get("tables", [])
    if representation == "semi-structured":
        return sample.get("tables", [])
    return []  # unstructured: no table block


def _section_char_bounds(prompt: str) -> Dict[str, Optional[tuple]]:
    """Char (start, end) of the table block and the verbal-text block."""
    def block(start_marker: str, end_markers: List[str]):
        i = prompt.find(start_marker)
        if i < 0:
            return None
        s = i + len(start_marker)
        ends = [prompt.find(m, s) for m in end_markers]
        ends = [e for e in ends if e >= 0]
        e = min(ends) if ends else len(prompt)
        return (s, e)

    return {
        "table": block("[Tables]\n", ["[Additional Information]", "[Question]"]),
        "text": (block("[Additional Information]\n", ["[Question]"])
                 or block("[Textual Information]\n", ["[Question]"])),
    }


def _count_tokens_in_span(offsets, span) -> int:
    if span is None:
        return 0
    cs, ce = span
    n = 0
    for (ts, te) in offsets:
        if ts == te:
            continue
        mid = (ts + te) / 2.0
        if cs <= mid < ce:
            n += 1
    return n


def number_fragmentation(tokenizer, tables: List[dict]) -> Dict[str, Any]:
    """How many subtokens each numeric/date cell value costs (bare, no context)."""
    frags: List[int] = []
    n_numeric = 0
    n_split = 0
    for t in tables:
        for row in t.get("table_content", []):
            for cell in (row if isinstance(row, list) else [row]):
                s = str(cell).strip()
                if not s:
                    continue
                if is_number(s) or is_date(s):
                    n_numeric += 1
                    k = len(tokenizer.encode(s, add_special_tokens=False))
                    frags.append(k)
                    if k >= 2:
                        n_split += 1
    return {
        "n_numeric_cells": n_numeric,
        "number_frag_mean": (sum(frags) / len(frags)) if frags else None,
        "number_frag_pct_split": (n_split / n_numeric) if n_numeric else None,
    }


def compute_one(tokenizer, sample: dict, representation: str,
                max_length: Optional[int]) -> Dict[str, Any]:
    prompt = config.prompt_for(sample, representation)
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    total = len(ids)

    structural = sum(
        1 for (ts, te) in offsets
        if ts != te and _STRUCTURAL_RE.match(prompt[ts:te] or "")
    )

    bounds = _section_char_bounds(prompt)
    table_tok = _count_tokens_in_span(offsets, bounds["table"])
    text_tok = _count_tokens_in_span(offsets, bounds["text"])

    tables = _tables_for(sample, representation)
    n_cells = sum(len(t.get("table_content", [])) * max(1, len(t.get("table_columns", [])))
                  for t in tables)
    numinfo = number_fragmentation(tokenizer, tables)

    ev = evidence_token_indices(tokenizer, prompt, sample.get("answer"), max_length)
    ev_idx = ev["token_indices"]
    ev_rel = (sum(ev_idx) / len(ev_idx) / total) if (ev_idx and total) else None

    return {
        "id": sample.get("id_"),
        "representation": representation,
        "total_tokens": total,
        "table_tokens": table_tok,
        "text_tokens": text_tok,
        "n_cells": n_cells,
        "fertility_cell": (table_tok / n_cells) if n_cells else None,
        "structural_token_share": (structural / total) if total else None,
        **numinfo,
        "evidence_status": ev["status"],
        "evidence_token_count": len(ev_idx),
        "evidence_rel_pos": ev_rel,
    }


def run(input_json: str, output_jsonl: str, tokenizer_path: Optional[str] = None,
        max_length: Optional[int] = None, split: Optional[str] = None) -> List[dict]:
    tok = get_tokenizer(tokenizer_path)
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    records: List[dict] = []
    for sample in data:
        per_rep = {}
        for rep in config.REPRESENTATIONS:
            rec = compute_one(tok, sample, rep, max_length)
            rec["split"] = split
            per_rep[rep] = rec
        # length inflation relative to structured
        base = per_rep["structured"]["total_tokens"] or 1
        for rep, rec in per_rep.items():
            rec["length_inflation_ratio"] = rec["total_tokens"] / base
            records.append(rec)

    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for r in records:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    _print_summary(records)
    return records


def _print_summary(records: List[dict]) -> None:
    agg = defaultdict(lambda: defaultdict(list))
    for r in records:
        for k in ("total_tokens", "fertility_cell", "structural_token_share",
                  "number_frag_mean", "length_inflation_ratio", "evidence_token_count"):
            if r.get(k) is not None:
                agg[r["representation"]][k].append(r[k])
    print("\n=== Tokenization summary (mean over samples) ===")
    hdr = ["representation", "total_tokens", "fertility_cell", "struct_share",
           "num_frag", "len_inflation", "ev_tok"]
    print("{:<16} {:>12} {:>14} {:>12} {:>9} {:>13} {:>7}".format(*hdr))
    for rep in config.REPRESENTATIONS:
        a = agg[rep]
        def m(k):
            v = a.get(k, [])
            return (sum(v) / len(v)) if v else float("nan")
        print("{:<16} {:>12.1f} {:>14.3f} {:>12.4f} {:>9.3f} {:>13.3f} {:>7.2f}".format(
            rep, m("total_tokens"), m("fertility_cell"), m("structural_token_share"),
            m("number_frag_mean"), m("length_inflation_ratio"), m("evidence_token_count")))
    n_missing = sum(1 for r in records if r["evidence_status"] == "missing")
    print(f"\nevidence coverage: {len(records)-n_missing}/{len(records)} located "
          f"({n_missing} missing)")


def parse_args():
    p = argparse.ArgumentParser(description="Tokenization metrics across representations.")
    p.add_argument("--input", required=True, help="Input JSON (post-s4_1, with raw_tables/verbalized_data).")
    p.add_argument("--output", required=True, help="Output JSONL of per-(sample,rep) metrics.")
    p.add_argument("--tokenizer", default=None, help="Tokenizer path (default: Qwen3-32B).")
    p.add_argument("--max_length", type=int, default=None)
    p.add_argument("--split", default=None, help="Optional split label stored on each record.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output, args.tokenizer, args.max_length, args.split)
