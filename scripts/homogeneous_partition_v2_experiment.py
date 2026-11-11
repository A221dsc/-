"""H0.1: lightweight homogeneous partition comparison (one shared Quad per leaf).

Compares the direct full-map Quad, the old PartitionIndex 1x1 (container
overhead control) and the new HomogeneousPartitionIndex for 1x1/4x4/8x8/16x16/
32x32 and the non-uniform partitions. Same traces, interleaved, dual anchors.
"""
import csv
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from prepare import json_file  # noqa: E402

config = json.load(open("configs/m4.json"))
selection = json.load(open("results/m4/calibration/selected.json"))
fine, quad = selection["fine_cell_m"], selection["quad_capacity"]
plans = Path("results/homogeneous_partition/plans")
traces = Path("data/processed/m4/traces")
out = Path("results/homogeneous_partition/v2")
out.mkdir(parents=True, exist_ok=True)
sentinel = traces / "train-0-brc202d.actual.m3bin"
variants = ["uniform-1", "uniform-4", "uniform-8", "uniform-16", "uniform-32", "entity",
            "query", "combined"]
names = ["brc100d", "brc203d", "den011d"]


def run(args):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(result.stderr)
    return dict(token.split("=", 1) for token in result.stdout.split())


def measure(trace, name, param):
    return float(run(["build_m3/spatial_m3.exe", "measure", name, str(fine), str(quad),
                      str(param), "0", str(trace), "NONE"])["online_ms"])


def drift(a, b):
    return max(a, b) / min(a, b) if min(a, b) > 0 else float("inf")


for _ in range(2):
    measure(sentinel, "Fixed Regional Scan", 1)
sent_start = measure(sentinel, "Fixed Regional Scan", 1)
rows, anchors = [], []
for index in range(3):
    ident = f"test-{index}-{names[index]}"
    for actual in range(3):
        trace = traces / (f"{ident}.actual.m3bin" if actual == 0
                          else f"{ident}.actual{actual}.m3bin")
        suffix = "" if actual == 0 else f".actual{actual}"
        jobs = [("new", name) for name in variants]
        jobs.append(("old1", "uniform-1"))
        jobs.append(("direct", "PR Quadtree"))
        random.Random(901 + index * 10 + actual).shuffle(jobs)
        start = measure(trace, "Fixed Regional Scan", 1)
        for kind, name in jobs:
            plan = plans / f"{ident}{suffix}.{name}.plan"
            if kind == "new":
                head = run(["build_m3/spatial_m3.exe", "hpartition", str(fine), str(quad), "0",
                            str(trace), str(plan), "0", "0"])
                label = f"new homogeneous {name}"
            elif kind == "old1":
                head = run(["build_m3/spatial_m3.exe", "partition", str(fine), str(quad), "0",
                            str(trace), str(plan), "0", "2"])
                label = "old PartitionIndex 1x1 (LOCAL_QUAD)"
            else:
                head = run(["build_m3/spatial_m3.exe", "measure", "PR Quadtree", str(fine),
                            str(quad), str(quad), "0", str(trace), "NONE"])
                label = "direct full-map Quad"
            rows.append({"scenario": ident, "actual": actual, "method": label,
                         "online_ms": float(head["online_ms"]),
                         "query_ms": float(head.get("query_ms", 0)),
                         "update_ms": float(head.get("update_ms", 0)),
                         "build_ms": float(head.get("build_ms", 0)),
                         "peak_mib": float(head.get("peak_mib", head.get("peak_rss_mb", 0))),
                         "index_nodes": int(float(head.get("index_nodes", 0))),
                         "touched_leaves": int(float(head.get("touched_leaves", 0))),
                         "route_cells": int(float(head.get("route_cells", 0))),
                         "p95_us": float(head.get("p95_us", 0)),
                         "mismatches": int(float(head.get("mismatches", 0)))})
        end = measure(trace, "Fixed Regional Scan", 1)
        anchors.append([ident, actual, start, end])
        print(f"{ident} a{actual} done", flush=True)
sent_end = measure(sentinel, "Fixed Regional Scan", 1)
with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
ratios = [(s, a, st, en, drift(st, en)) for s, a, st, en in anchors]
sentinel_ratio = drift(sent_start, sent_end)
valid = all(r[4] <= 1.15 for r in ratios) and sentinel_ratio <= 1.15
json_file(out / "manifest.json", {
    "per_trace": [{"scenario": s, "actual": a, "start": st, "end": en, "drift": d}
                  for s, a, st, en, d in ratios],
    "max_trace_drift": max(r[4] for r in ratios),
    "sentinel": {"start": sent_start, "end": sent_end, "drift": sentinel_ratio},
    "threshold": 1.15, "batch_valid": valid,
    "environment": {"platform": platform.platform(), "cpu_count": os.cpu_count()}})
print(f"max trace drift {max(r[4] for r in ratios):.3f}, sentinel {sentinel_ratio:.3f}, "
      f"valid={valid}")
by = {}
for row in rows:
    by.setdefault(row["method"], []).append(row)
print("\nmethod, median online, query, update, build, nodes, p95")
for method, group in sorted(by.items(), key=lambda kv: float(np.median([r["online_ms"] for r in kv[1]]))):
    print(f"{method}: {np.median([r['online_ms'] for r in group]):.1f} / "
          f"{np.median([r['query_ms'] for r in group]):.1f} / "
          f"{np.median([r['update_ms'] for r in group]):.1f} / "
          f"{np.median([r['build_ms'] for r in group]):.1f} / "
          f"{np.median([r['index_nodes'] for r in group]):.0f} / "
          f"{np.median([r['p95_us'] for r in group]):.1f}")
