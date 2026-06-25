#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert the HuggingFace `table-benchmark/mmqa` processed jsonl files into the
RePairTQA "Synthesized" JSON that data_selection/MMQA/data_selection.py expects.

Background: the original GitHub `WuJian1995/MMQA` was deleted; the multi-table
MMQA (Wu et al. ICLR 2025) now lives at HF `table-benchmark/mmqa` as
  MMQA/processed/mmqa_two_table.jsonl
  MMQA/processed/mmqa_three_table.jsonl
Each line: {question, answer, table (a JSON *string* holding
{"table_names": [...], "tables": [{"table_columns": [...], "table_content": [...]}, ...]}),
original_dataset_id, dataset_name, table_title, text}.

Output (unified RePairTQA schema, a list of):
  {id_, db, Question, answer, table_names, tables[].{table_columns, table_content},
   primary_keys, foreign_keys, [SQL if present]}

NOTE: this processed mirror has NO SQL field. The verbalization step
src/verbalization_row_s4_1.py re-executes item["SQL"]; for MMQA we keep the gold
`answer` as-is, so run that step in a no-SQL mode (see --keep_answer there) or use
the column-based s4 with --no_sql for MMQA inputs.
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List


def _maybe_json(x: Any) -> Any:
    if isinstance(x, str):
        try:
            return json.loads(x)
        except (json.JSONDecodeError, ValueError):
            return x
    return x


def convert(in_path: str, out_path: str, prefix: str) -> int:
    out: List[Dict[str, Any]] = []
    with open(in_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tbl = _maybe_json(r.get("table", {})) or {}
            table_names = tbl.get("table_names", []) if isinstance(tbl, dict) else []
            tables = tbl.get("tables", []) if isinstance(tbl, dict) else []
            item: Dict[str, Any] = {
                "id_": f"{prefix}_{r.get('original_dataset_id', i)}",
                "db": r.get("dataset_name", "mmqa"),
                "Question": r.get("question"),
                "answer": r.get("answer"),
                "table_names": table_names,
                "tables": tables,
                # keys may live inside the `table` blob or at top level
                "primary_keys": (tbl.get("primary_keys") if isinstance(tbl, dict) else None)
                                or r.get("primary_keys", []),
                "foreign_keys": (tbl.get("foreign_keys") if isinstance(tbl, dict) else None)
                                or r.get("foreign_keys", []),
            }
            for sql_key in ("SQL", "sql", "query"):
                if r.get(sql_key):
                    item["SQL"] = r[sql_key]
                    break
            out.append(item)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    n_tables = [len(s["tables"]) for s in out[:200]]
    print(f"{in_path}: {len(out)} samples -> {out_path} "
          f"(first-200 table counts: min={min(n_tables) if n_tables else 0} "
          f"max={max(n_tables) if n_tables else 0})")
    return len(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Convert HF table-benchmark/mmqa -> RePairTQA Synthesized JSON.")
    p.add_argument("--two_in", default="data_selection/MMQA/_raw/MMQA/processed/mmqa_two_table.jsonl")
    p.add_argument("--three_in", default="data_selection/MMQA/_raw/MMQA/processed/mmqa_three_table.jsonl")
    p.add_argument("--two_out", default="data_selection/MMQA/Synthesized_two_table.json")
    p.add_argument("--three_out", default="data_selection/MMQA/Synthesized_three_table.json")
    args = p.parse_args()
    import os
    if os.path.exists(args.three_in):
        convert(args.three_in, args.three_out, "mmqa3")
    if os.path.exists(args.two_in):
        convert(args.two_in, args.two_out, "mmqa2")


if __name__ == "__main__":
    main()
