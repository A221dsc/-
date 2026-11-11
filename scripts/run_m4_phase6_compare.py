"""Anchored cross-method comparison including the partition plans."""
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from prepare import json_file

config = json.load(open("configs/m4.json"))
selection = json.load(open("results/m4/calibration/selected.json"))
fine, quad = selection["fine_cell_m"], selection["quad_capacity"]
base = Path("results/m4/calibration/selector_final.txt")
quantile = Path("results/m4/phase3_v2/tmp/selector_2_1.00_1.txt")
scen = Path("data/processed/m4/scenarios")
traces = Path("data/processed/m4/traces")
plans = Path("results/m4/phase6/ablation/plans")
out = Path("results/m4/phase6/compare")
out.mkdir(parents=True, exist_ok=True)
methods = [
    ("Fixed Regional Scan", None, None, 1, None),
    ("PR Quadtree", None, None, selection["classic"]["PR Quadtree"], None),
    ("Cost-Aware Greedy Regional Index Selection", base, None, 1, None),
    ("Uncertainty-Aware Feed-Forward Regional Index Selection", base,
     "nominal_samples_m32.csv", 1, None),
    ("Quantile-Cost-Penalized Feed-Forward Regional Index Selection", quantile,
     "nominal_samples_m32.csv", 1, None),
    ("Adaptive Partition (uniform 16x16) + Cost Greedy", None, None, 1, "uniform-16"),
    ("Adaptive Partition (load-aware) + Cost Greedy", None, None, 1, "adaptive-combined"),
]
rows = []
anchors = []
for index in range(3):
    ident = f"test-{index}-{Path(config['maps']['test'][index]).stem}"
    for actual in range(3):
        trace = traces / (f"{ident}.actual.m3bin" if actual == 0 else f"{ident}.actual{actual}.m3bin")
        order = list(methods)
        random.Random(6100 + index * 10 + actual).shuffle(order)
        batch = [("anchor-start", "Fixed Regional Scan", None, None, 1, None)] + \
                [("method", m, cfg, prior, param, plan) for (m, cfg, prior, param, plan) in order] + \
                [("anchor-end", "Fixed Regional Scan", None, None, 1, None)]
        for kind, method, cfg, prior_name, param, plan_name in batch:
            if plan_name:
                plan = plans / f"{ident}.{plan_name}.plan"
                args = ["build_m3/spatial_m3.exe", "partition", "160", "32", "0",
                        str(trace), str(plan), "0"]
            else:
                args = ["build_m3/spatial_m3.exe", "measure", method, str(fine), str(quad),
                        str(param), "0", str(trace), str(cfg) if cfg else "NONE"]
                if prior_name:
                    args.append(str(scen / f"{ident}.{prior_name}"))
            result = subprocess.run(args, capture_output=True, text=True)
            if result.returncode:
                raise SystemExit(result.stderr)
            head = dict(token.split("=", 1) for token in result.stdout.strip().split())
            label = method if not plan_name else f"{method} [{plan_name}]"
            rows.append({"scenario": ident, "actual": actual, "role": kind,
                         "method": label, "online_ms": head["online_ms"]})
            if kind.startswith("anchor"):
                anchors.append(float(head["online_ms"]))
        print(f"{ident} a{actual} done", flush=True)
with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
json_file(out / "manifest.json", {"anchor_runs": len(anchors),
                                  "anchor_median": sorted(anchors)[len(anchors) // 2],
                                  "anchor_min": min(anchors), "anchor_max": max(anchors)})
print("done")
