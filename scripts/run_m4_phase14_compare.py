"""M4.14 amortized-selection comparison with dual anchors and a sentinel."""
import csv
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from prepare import json_file  # noqa: E402

config = json.load(open("configs/m4.json"))
selection = json.load(open("results/m4/calibration/selected.json"))
fine, quad = selection["fine_cell_m"], selection["quad_capacity"]
selector = Path("results/m4/calibration/selector_final.txt")
scen = Path("data/processed/m4/scenarios")
traces = Path("data/processed/m4/traces")
plans = Path("results/m4/phase7/plans")
out = Path("results/m4/phase14/compare")
out.mkdir(parents=True, exist_ok=True)
sentinel_trace = traces / "train-0-brc202d.actual.m3bin"


def run(args):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(result.stderr)
    return dict(token.split("=", 1) for token in result.stdout.strip().split())


def measure(trace):
    return float(run(["build_m3/spatial_m3.exe", "measure", "Fixed Regional Scan", str(fine),
                      str(quad), "1", "0", str(trace), "NONE"])["online_ms"])


def drift_ratio(start, end):
    return max(start, end) / min(start, end) if min(start, end) > 0 else float("inf")


for _ in range(2):
    measure(sentinel_trace)
sentinel_start = measure(sentinel_trace)
rows, anchors = [], []
for index in range(3):
    ident = f"test-{index}-{Path(config['maps']['test'][index]).stem}"
    for actual in range(3):
        trace = traces / (f"{ident}.actual.m3bin" if actual == 0 else f"{ident}.actual{actual}.m3bin")
        jobs = []
        for name, amortize in (("prior-h1", 1), ("prior-h2", 2), ("prior-h4", 4)):
            jobs.append(("greedy", 1, name, amortize))
        for name, amortize in (("history-h1", 1), ("history-h4", 4)):
            jobs.append(("greedy", 0, name, amortize))
        jobs.append(("scan", 0, "all-scan", 1))
        jobs.append(("quantile", 1, "quantile-feedforward", 1))
        random.Random(1400 + index * 10 + actual).shuffle(jobs)
        start = measure(trace)
        for kind, prior, name, amortize in jobs:
            if kind == "scan":
                head = run(["build_m3/spatial_m3.exe", "partition", str(fine), str(quad), "0",
                            str(trace), str(plans / f"{ident}.uniform-16.plan"), "0", "0"])
                label = "all-SCAN"
            elif kind == "greedy":
                head = run(["build_m3/spatial_m3.exe", "partition_greedy", str(fine), str(quad),
                            "0", str(trace), str(plans / f"{ident}.uniform-16.plan"), str(selector),
                            str(prior), "0", str(amortize)])
                label = f"{'prior' if prior else 'history'} greedy H{amortize}"
            else:
                head = run(["build_m3/spatial_m3.exe", "measure",
                            "Quantile-Cost-Penalized Feed-Forward Regional Index Selection",
                            str(fine), str(quad), "1", "0", str(trace), str(selector),
                            str(scen / f"{ident}.nominal_samples_m32.csv")])
                label = "quantile feed-forward"
            rows.append({"scenario": ident, "actual": actual, "role": kind, "method": label,
                         "online_ms": float(head["online_ms"]),
                         "decide_ms": float(head.get("decide_ms", 0)),
                         "promotions": head.get("promotions", 0),
                         "leaves": head.get("leaves", 0)})
        end = measure(trace)
        anchors.append([ident, actual, start, end])
        print(f"{ident} a{actual} done", flush=True)
sentinel_end = measure(sentinel_trace)
with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
ratios = [(s, a, st, en, drift_ratio(st, en)) for s, a, st, en in anchors]
sentinel = drift_ratio(sentinel_start, sentinel_end)
valid = all(r[4] <= 1.15 for r in ratios) and sentinel <= 1.15
json_file(out / "manifest.json", {
    "per_trace": [{"scenario": s, "actual": a, "start": st, "end": en, "drift": d}
                  for s, a, st, en, d in ratios],
    "max_trace_drift": max(r[4] for r in ratios), "sentinel": {"start": sentinel_start,
                                                              "end": sentinel_end, "drift": sentinel},
    "threshold": 1.15, "batch_valid": valid,
    "environment": {"platform": platform.platform(), "cpu_count": os.cpu_count()}})
print(f"max trace drift {max(r[4] for r in ratios):.3f}, sentinel {sentinel:.3f}, valid={valid}")
