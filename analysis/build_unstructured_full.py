#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a FULL unstructured representation: verbalize EVERY row of each table into
text (ratio=1.0), rather than the partial ratio=0.5 used for the semi-structured
view. Produces pipeline_results/<split>/example_verbalized_full.json whose
`verbalized_data` covers the whole (row-capped) table.

Why a separate file: the standard pipeline runs s4_1 once at ratio=0.5, so its
`verbalized_data` only holds ~half the rows (the rest stay as a table for the
semi view). The text-only "unstructured" view therefore silently dropped the
other half. This full pass fixes that for the unstructured representation only.

Rows are capped at CAP (default 1000, matching analysis.config.MAX_TABLE_ROWS) so
monster tables (MMQA M2, ~190k rows) stay tractable; --no_sql keeps the gold
answer (no SQL re-execution).
"""
from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, "src")
import verbalization_row_s4_1 as s4_1  # noqa: E402

SPLITS = {
    "bird_S1": "data_selection/BIRD/splits/bird_S1_short_lookup_with_answer.json",
    "bird_S3": "data_selection/BIRD/splits/bird_S3_short_compositional_with_answer.json",
    "bird_S4": "data_selection/BIRD/splits/bird_S4_long_lookup_with_answer.json",
    "bird_S5": "data_selection/BIRD/splits/bird_S5_long_compositional_with_answer.json",
    "tableeval_S2": "data_selection/TableEval/TableEval_S2_simple_short_flat.json",
    "mmqa_M1": "data_selection/MMQA/splits/MMQA_M1_short_multi.json",
    "mmqa_M2": "data_selection/MMQA/splits/MMQA_M2_long_multi.json",
}
CAP = int(os.environ.get("CAP", "1000"))


def main() -> None:
    for split, data in SPLITS.items():
        tmpl = f"pipeline_results/{split}/col_combinations_all_columns.csv"
        if not (os.path.exists(data) and os.path.exists(tmpl)):
            print(f"[SKIP] {split}: missing data or templates")
            continue
        # cap table rows so full verbalization stays tractable
        d = json.load(open(data, encoding="utf-8"))
        for sample in d:
            for t in sample.get("tables", []):
                t["table_content"] = t.get("table_content", [])[:CAP]
        capped = f"pipeline_results/{split}/_capped_data.json"
        json.dump(d, open(capped, "w", encoding="utf-8"), ensure_ascii=False)

        out = f"pipeline_results/{split}/example_verbalized_full.json"
        random.seed(42)  # reproducible row sampling (ratio=1.0 -> all rows anyway)
        s4_1.run(
            data_path=capped,
            template_path=tmpl,
            output_path=out,
            failed_path=f"pipeline_results/{split}/_full_failed.json",
            ratio=1.0,
            no_sql=True,
        )
        res = json.load(open(out, encoding="utf-8"))
        has_text = sum(1 for s in res if s.get("verbalized_data"))
        print(f"[{split}] {len(res)} samples, {has_text} with full verbalized_data")


if __name__ == "__main__":
    main()
