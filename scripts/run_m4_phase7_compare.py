"""Same-policy partition ablation with the C++ online cost-greedy.

All partition rows use the same PartitionIndex engine and the same C++ greedy
(prior mode or history mode). The original AdaptiveRegionalIndex 16x16 selector
is measured alongside to separate layout from policy.
"""
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
selector = Path("results/m4/calibration/selector_final.txt")
scen = Path("data/processed/m4/scenarios")
traces = Path("data/processed/m4/traces")
plans = Path("results/m4/phase7/plans")
out = Path("results/m4/phase7/compare")
out.mkdir(parents=True, exist_ok=True)
variants = ["uniform-8", "uniform-16", "uniform-32", "adaptive-entity", "adaptive-query",
            "adaptive-combined"]
rows = []
anchors = []
for index in range(3):
    ident = f"test-{index}-{Path(config['maps']['test'][index]).stem}"
    for actual in range(3):
        trace = traces / (f"{ident}.actual.m3bin" if actual == 0 else f"{ident}.actual{actual}.m3bin")
        jobs = []
        for variant in variants:
            for prior in (1, 0):
                jobs.append(("partition", variant, prior, None, None))
        jobs += [("classic", "Cost-Aware Greedy Regional Index Selection", None, selector, None),
                 ("classic", "Cost-Aware Greedy Regional Index Selection (Scenario)", None, selector,
                  "nominal_samples_m32.csv"),
                 ("classic", "PR Quadtree", None, None, None)]
        random.Random(7100 + index * 10 + actual).shuffle(jobs)
        jobs = [("anchor", "scan", None, None, None)] + jobs + \
               [("anchor", "scan", None, None, None)]
        for kind, name, prior, cfg, prior_file in jobs:
            if kind == "anchor":
                args = ["build_m3/spatial_m3.exe", "measure", "Fixed Regional Scan", str(fine),
                        str(quad), "1", "0", str(trace), "NONE"]
                label = "Fixed Regional Scan"
            elif kind == "partition":
                args = ["build_m3/spatial_m3.exe", "partition_greedy", str(fine), str(quad), "0",
                        str(trace), str(plans / f"{ident}.{name}.plan"), str(selector),
                        str(prior), "0"]
                label = f"Partition {name} + C++ {'prior' if prior else 'history'} greedy"
            else:
                args = ["build_m3/spatial_m3.exe", "measure", name, str(fine), str(quad),
                        str(selection["classic"]["PR Quadtree"] if name == "PR Quadtree" else 1),
                        "0", str(trace), str(cfg) if cfg else "NONE"]
                if prior_file:
                    args.append(str(scen / f"{ident}.{prior_file}"))
                label = name
            result = subprocess.run(args, capture_output=True, text=True)
            if result.returncode:
                raise SystemExit(result.stderr)
            head = dict(token.split("=", 1) for token in result.stdout.strip().split())
            rows.append({"scenario": ident, "actual": actual, "role": kind, "method": label,
                         "online_ms": head["online_ms"],
                         "decide_ms": head.get("decide_ms", 0.0),
                         "leaves": head.get("leaves", 0),
                         "promotions": head.get("promotions", 0)})
            if kind == "anchor":
                anchors.append(float(head["online_ms"]))
        print(f"{ident} a{actual} done", flush=True)
with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
threshold = 1.15
drift = (max(anchors) / min(anchors)) if anchors and min(anchors) > 0 else 0.0
import platform
json_file(out / "manifest.json", {"anchor_runs": len(anchors),
                                  "anchor_median": sorted(anchors)[len(anchors) // 2],
                                  "anchor_min": min(anchors), "anchor_max": max(anchors),
                                  "anchor_drift": drift, "drift_threshold": threshold,
                                  "batch_valid": drift <= threshold,
                                  "environment": {"platform": platform.platform(),
                                                  "cpu_count": __import__("os").cpu_count()},
                                  "note": "partition rows include the C++ decision time"})
if drift > threshold:
    print(f"WARNING: anchor drift {drift:.2f} exceeds {threshold}, batch marked invalid")
print("done")
