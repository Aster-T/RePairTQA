#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Produce a CLEAN ``example_verbalized_full.json`` that adds a full-text view for
the *unstructured* representation WITHOUT corrupting the semi-structured config.

Background
----------
``build_unstructured_full.py`` ran s4_1 at ratio=1.0, which verbalizes *every*
row into text. A side effect is that the semi-structured table (``tables``) is
emptied (no rows remain) and the sample set differs from the canonical 0.5 run
(``--no_sql`` keeps a few extra samples). If that file were ever read for the
semi or structured view, it would be wrong.

The fix: ``verbalized_data`` is shared by the semi and unstructured prompts, so a
single file cannot hold both a half-text (semi) and a full-text (unstructured)
version of it. We therefore keep the canonical 0.5 file untouched and attach the
full text under a SEPARATE field, ``verbalized_data_full``, matched by ``id_``.

Result per split (overwrites example_verbalized_full.json):
  - tables / raw_tables / verbalized_data : IDENTICAL to the 0.5 file (semi &
    structured configs preserved exactly).
  - verbalized_data_full                  : every-row text, read only by the
    unstructured-FULL accuracy + attention passes (--verbal_field).
  - sample set = the 0.5 set (paired 1:1 with structured/semi/unstructured).

The raw ratio=1.0 output is snapshotted once to example_verbalized_full_raw.json
so this script is idempotent (re-runs read the immutable snapshot, not the clean
file it writes).
"""
from __future__ import annotations

import json
import os
import shutil

SPLITS = ["bird_S1", "bird_S3", "bird_S4", "bird_S5",
          "tableeval_S2", "mmqa_M1", "mmqa_M2"]


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    for split in SPLITS:
        base = f"pipeline_results/{split}/example_verbalized.json"          # 0.5, canonical
        full = f"pipeline_results/{split}/example_verbalized_full.json"     # ratio=1.0 (consumable)
        raw = f"pipeline_results/{split}/example_verbalized_full_raw.json"  # immutable snapshot
        if not (os.path.exists(base) and os.path.exists(full)):
            print(f"[SKIP] {split}: missing base or full file")
            continue

        # Snapshot the raw ratio=1.0 output exactly once, then always read from it.
        if not os.path.exists(raw):
            shutil.copyfile(full, raw)
        raw_data = _load(raw)
        full_text = {s.get("id_"): s.get("verbalized_data", []) for s in raw_data}

        base_data = _load(base)
        out = []
        missing = 0
        empty = 0
        for s in base_data:
            sid = s.get("id_")
            vdf = full_text.get(sid)
            if vdf is None:
                missing += 1
                vdf = s.get("verbalized_data", [])  # fall back to 0.5 text
            if not vdf:
                empty += 1
            o = dict(s)                      # shallow: tables/raw_tables/verbalized_data shared, not mutated
            o["verbalized_data_full"] = vdf  # full-row text for the unstructured-full view
            out.append(o)

        with open(full, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        # sanity: semi tables must still be non-empty (intact 0.5 config)
        semi_rows0 = sum(len(t.get("table_content", [])) for t in out[0].get("tables", []))
        vd0 = sum(len(v.get("text", "")) for v in out[0].get("verbalized_data", []))
        vdf0 = sum(len(v.get("text", "")) for v in out[0].get("verbalized_data_full", []))
        print(f"[{split}] {len(out)} samples (paired) | id-miss={missing} full-empty={empty} | "
              f"sample[0]: semi_rows={semi_rows0} vd_chars={vd0} vd_full_chars={vdf0}")


if __name__ == "__main__":
    main()
