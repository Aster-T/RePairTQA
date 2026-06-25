#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared configuration + a thin bridge to the existing pipeline code in ``src/``.

This package studies the association between
  (1) tokenization of the same table content under different representations,
  (2) the model's attention distribution while answering, and
  (3) table-sequence semantic invariance (the paper's controlled
      "same content, different representation" pairing),
using Qwen3-32B.

Nothing here loads a GPU model at import time, so it is safe to import on a
CPU-only box for the tokenization-only stage.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --- Repo layout -----------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
OUT_DIR = Path(os.environ.get("ANALYSIS_OUT_DIR", REPO_ROOT / "analysis_outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Model -----------------------------------------------------------------
# Local Qwen3-32B (downloaded from ModelScope). Override with QWEN3_PATH.
MODEL_PATH = os.environ.get("QWEN3_PATH", "/home/amax/models/Qwen3-32B")
MODEL_NAME = "Qwen3-32B"  # logical name passed to the vLLM endpoint

# The two RTX 4090 48GB cards (physical indices 2,3). A800s (0,1) are off-limits.
CUDA_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "2,3")

# --- Attention-capture limits (see plan: output_attentions OOMs ~L>1200, so we
#     use per-layer reduce-and-free hooks and still cap the sequence length). --
MAX_L = int(os.environ.get("ANALYSIS_MAX_L", "4096"))

# --- vLLM endpoint (accuracy / data-prep phases) ---------------------------
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8100"))
VLLM_BASE_URL = os.environ.get("OPENAI_BASE_URL", f"http://localhost:{VLLM_PORT}/v1")

# The three representations studied (paper's views). The string values match the
# ``data_type`` argument accepted by src/llm_infer_s5.estimate_prompt_tokens.
REPRESENTATIONS = ["structured", "semi-structured", "unstructured"]


# --- Bridge to existing pipeline code --------------------------------------
def load_s5():
    """Import src/llm_infer_s5.py (not a package) and return the module.

    Reuses its prompt builders and metric functions so every analysis prompt is
    byte-identical to what the accuracy run feeds the model.
    """
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    import llm_infer_s5  # noqa: E402  (path injected above)

    return llm_infer_s5


def build_prompt(sample: dict, data_type: str) -> str:
    """Return the exact prompt string for one sample under one representation.

    Mirrors the per-sample field extraction in src/llm_infer_s5.main():
      - structured / semi-structured read sample['tables'] (+ verbalized_data)
      - the structured *baseline* the paper reports is built from raw_tables;
        here we expose both via ``data_type`` and the ``use_raw`` switch below.
    """
    s5 = load_s5()
    verbal = [v["text"] for v in sample.get("verbalized_data", [])]
    cols_list = [t["table_columns"] for t in sample.get("tables", [])]
    rows_list = [t["table_content"] for t in sample.get("tables", [])]
    table_names = sample.get("table_names", None)
    _, prompt = s5.estimate_prompt_tokens(
        cols_list,
        rows_list,
        verbal,
        sample["Question"],
        data_type=data_type,
        model="gpt-4o",  # only affects tiktoken; we re-tokenize with Qwen3 ourselves
        table_names=table_names,
    )
    return prompt


def prompt_for(sample: dict, representation: str) -> str:
    """Map a representation name to the correct prompt, honoring the paper's
    controlled pairing after s4_1 verbalization:

      structured       -> original full table (sample['raw_tables']), no text
      semi-structured   -> verbalized table (sample['tables']) + verbalized text
      unstructured      -> verbalized text only

    Note: post-verbalization sample['tables'] are the *semi* tables and
    sample['raw_tables'] holds the *original* table, so the structured view must
    be built from raw_tables (this matches s5's structured-baseline pass).
    """
    if representation == "structured":
        return build_prompt_from_raw(sample)
    if representation in ("semi-structured", "unstructured"):
        return build_prompt(sample, representation)
    raise ValueError(f"Unknown representation: {representation}")


def build_prompt_from_raw(sample: dict) -> str:
    """Structured prompt built from sample['raw_tables'] (the paper's structured
    baseline). Falls back to top-level tables if raw_tables is absent."""
    s5 = load_s5()
    raw = sample.get("raw_tables") or {}
    table_entries = raw.get("tables") or sample.get("tables", [])
    table_names = raw.get("table_names") or sample.get("table_names", None)
    cols_list = [t["table_columns"] for t in table_entries]
    rows_list = [t["table_content"] for t in table_entries]
    _, prompt = s5.estimate_prompt_tokens(
        cols_list, rows_list, [], sample["Question"],
        data_type="structured", model="gpt-4o", table_names=table_names,
    )
    return prompt
