#!/usr/bin/env python3
"""Aggregate H3 management-partition benchmark CSVs across workload seeds."""
from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: summarize_h3_partition.py <seed_dir> <output_csv>")
        return 2
    seed_dir = Path(sys.argv[1])
    output = Path(sys.argv[2])
    files = sorted(seed_dir.glob("bench_*.csv"))
    if not files:
        print(f"no bench_*.csv in {seed_dir}")
        return 1

    rows: list[dict[str, str]] = []
    for path in files:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))

    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], int(row["threads"]))].append(row)

    def median(key: str, items: list[dict[str, str]]) -> float:
        return statistics.median(float(item[key]) for item in items)

    base = groups.get(("GLOBAL_H3", 1))
    base_ms = median("total_ms", base) if base else 0.0

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "threads",
        "seeds",
        "partitions",
        "nodes",
        "bytes",
        "build_ms",
        "update_ms",
        "total_ms",
        "relative_to_global_1t",
        "qps",
        "p95_ms",
        "p99_ms",
        "candidate_cells",
        "bucket_lookups",
        "leaves_visited",
        "candidates",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (method, threads), items in sorted(groups.items()):
            total = median("total_ms", items)
            writer.writerow(
                {
                    "method": method,
                    "threads": threads,
                    "seeds": len(files),
                    "partitions": median("partitions", items),
                    "nodes": median("nodes", items),
                    "bytes": median("bytes", items),
                    "build_ms": median("build_ms", items),
                    "update_ms": median("update_ms", items),
                    "total_ms": total,
                    "relative_to_global_1t": total / base_ms if base_ms else 0.0,
                    "qps": median("qps", items),
                    "p95_ms": median("p95_ms", items),
                    "p99_ms": median("p99_ms", items),
                    "candidate_cells": median("candidate_cells", items),
                    "bucket_lookups": median("bucket_lookups", items),
                    "leaves_visited": median("leaves_visited", items),
                    "candidates": median("candidates", items),
                }
            )
    print(f"wrote {output} from {len(files)} seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
