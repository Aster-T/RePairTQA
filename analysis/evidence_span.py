#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Locate the gold-answer "evidence" token span inside a prompt.

This is the linchpin that ties tokenization to attention: because the same cell
verbalizes differently across representations, the set of token indices that
carry the answer is what makes "same content, different representation"
measurable. Attention metrics (mass, entropy targeting) key off this span.

We deliberately use a *fast* tokenizer's offset mapping so character spans map
back to exact token indices. A QC flag records whether the match was exact,
fuzzy (number-normalized), or missing (e.g. the unstructured view paraphrased
"34500" as "thirty-four thousand five hundred").
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


def _flatten(x: Any) -> List[Any]:
    out: List[Any] = []
    if isinstance(x, (list, tuple)):
        for it in x:
            out.extend(_flatten(it))
    elif isinstance(x, dict):
        for it in x.values():
            out.extend(_flatten(it))
    else:
        out.append(x)
    return out


def atomic_answer_values(gold: Any) -> List[str]:
    """Flatten a (possibly nested) gold answer into atomic non-empty strings."""
    vals = []
    for v in _flatten(gold):
        if v is None:
            continue
        s = str(v).strip()
        if s:
            vals.append(s)
    # de-dup, keep order
    seen, uniq = set(), []
    for s in vals:
        if s.lower() not in seen:
            seen.add(s.lower())
            uniq.append(s)
    return uniq


def _number_variants(v: str) -> List[str]:
    """Variants of a numeric value to tolerate formatting drift
    (thousands separators, trailing .0)."""
    out = {v}
    nocomma = v.replace(",", "")
    out.add(nocomma)
    # x.0 -> x
    m = re.fullmatch(r"(-?\d+)\.0+", nocomma)
    if m:
        out.add(m.group(1))
    # integer -> with thousands separators
    if re.fullmatch(r"-?\d+", nocomma):
        try:
            out.add(f"{int(nocomma):,}")
        except ValueError:
            pass
    return [s for s in out if s]


def _find_char_spans(haystack_lower: str, needle: str) -> List[Tuple[int, int]]:
    """All non-overlapping occurrences of ``needle`` (case-insensitive) in text."""
    spans: List[Tuple[int, int]] = []
    if not needle:
        return spans
    n = needle.lower()
    start = 0
    while True:
        idx = haystack_lower.find(n, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(n)))
        start = idx + len(n)
    return spans


def evidence_token_indices(
    tokenizer,
    prompt: str,
    gold: Any,
    max_length: int | None = None,
) -> Dict[str, Any]:
    """Return the evidence token indices for ``gold`` inside ``prompt``.

    Returns dict with:
      token_indices : sorted list[int]  -- token positions carrying the answer
      status        : "exact" | "fuzzy" | "missing"
      matched_values: list[str]         -- which atomic values were located
      n_input_tokens: int               -- total tokens in the (truncated) prompt
    """
    enc = tokenizer(
        prompt,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=max_length is not None,
        max_length=max_length,
    )
    offsets = enc["offset_mapping"]
    n_tokens = len(enc["input_ids"])
    prompt_lower = prompt.lower()

    values = atomic_answer_values(gold)
    char_spans: List[Tuple[int, int]] = []
    matched: List[str] = []
    status = "missing"

    # Pass 1: exact (case-insensitive) substring match.
    for v in values:
        spans = _find_char_spans(prompt_lower, v)
        if spans:
            char_spans.extend(spans)
            matched.append(v)
    if matched:
        status = "exact"

    # Pass 2: number-normalized fallback for values not yet matched.
    if len(matched) < len(values):
        for v in values:
            if v in matched:
                continue
            for variant in _number_variants(v):
                spans = _find_char_spans(prompt_lower, variant)
                if spans:
                    char_spans.extend(spans)
                    matched.append(v)
                    if status != "exact":
                        status = "fuzzy"
                    break

    # Map char spans -> token indices via overlap with offset mapping.
    tok_idx = set()
    for (cs, ce) in char_spans:
        for i, (ts, te) in enumerate(offsets):
            if ts == te:  # special/empty token
                continue
            if ts < ce and te > cs:  # overlap
                tok_idx.add(i)

    return {
        "token_indices": sorted(tok_idx),
        "status": status,
        "matched_values": matched,
        "n_input_tokens": n_tokens,
    }
