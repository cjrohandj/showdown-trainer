#!/usr/bin/env python3
"""Check whether a JSONL shard contains one game or many.

This is useful for files that have lots of `winner: null` decision rows and you
want to know whether that means:

1. one battle never reached a final result row, or
2. the file contains many separate battles.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize battle counts in a Showdown JSONL file."
    )
    parser.add_argument("input_path", help="Path to a .jsonl or .jsonl.gz file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze_jsonl(Path(args.input_path))

    print(f"rows: {summary['rows']}")
    print(f"battle_ids: {summary['battle_count']}")
    print(f"decision_rows: {summary['decision_rows']}")
    print(f"result_rows: {summary['result_rows']}")
    print(f"run_rows: {summary['run_rows']}")

    if summary["battle_count"] == 0:
        print("No battle rows found.")
        return

    if summary["battle_count"] == 1:
        battle_id = summary["battle_ids"][0]
        print(f"One battle found: {battle_id}")
        if battle_id in summary["unfinished_battles"]:
            print("This battle looks unfinished: no result row was written.")
        else:
            print("This battle has a result row.")
        return

    print("Multiple battles found:")
    for battle_id in summary["battle_ids"]:
        status = "finished" if battle_id in summary["finished_battles"] else "unfinished"
        decision_count = summary["decision_counts"].get(battle_id, 0)
        result_count = summary["result_counts"].get(battle_id, 0)
        print(
            f"  - {battle_id}: {status} "
            f"(decision rows={decision_count}, result rows={result_count})"
        )

    if summary["unfinished_battles"]:
        print("At least one battle did not reach a final result row.")
    else:
        print("Every battle has a final result row.")


def analyze_jsonl(path: Path) -> dict[str, Any]:
    decision_counts: Counter[str] = Counter()
    result_counts: Counter[str] = Counter()
    rows = 0
    run_rows = 0
    battle_order: list[str] = []
    seen_battles: set[str] = set()

    for record in iter_jsonl(path):
        rows += 1
        record_type = str(record.get("record_type") or "")
        battle_id = str(record.get("battle_id") or "")

        if record_type == "run":
            run_rows += 1
            continue

        if not battle_id:
            continue

        if battle_id not in seen_battles:
            seen_battles.add(battle_id)
            battle_order.append(battle_id)

        if record_type == "decision":
            decision_counts[battle_id] += 1
        elif record_type == "result":
            result_counts[battle_id] += 1

    finished_battles = {
        battle_id for battle_id in battle_order if result_counts[battle_id] > 0
    }
    unfinished_battles = set(battle_order) - finished_battles

    return {
        "rows": rows,
        "run_rows": run_rows,
        "battle_ids": battle_order,
        "battle_count": len(battle_order),
        "decision_counts": dict(decision_counts),
        "result_counts": dict(result_counts),
        "decision_rows": sum(decision_counts.values()),
        "result_rows": sum(result_counts.values()),
        "finished_battles": finished_battles,
        "unfinished_battles": unfinished_battles,
    }


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


if __name__ == "__main__":
    main()
