"""Four-way partition RL comparison on the same QPC replay path.

A: uniform 16x16 + prior cost greedy
B: uniform 16x16 + prior Double DQN (saved executed plans)
C: adaptive-query partition + prior cost greedy
D: adaptive-query partition + prior Double DQN
plus the historical DQN variants and classic indexes. Anchors bracket the batch
and a drift above the pre-registered threshold invalidates it.
"""
import csv
import json
import os
import platform
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
eval_plans = Path("results/m4/phase8/eval/plans")
eval_rows = list(csv.DictReader(open("results/m4/phase8/eval/metrics.csv")))
inference = {(r["variant"], int(r["seed"]), r["partition"], r["scenario"], int(r["actual"])):
             float(r["inference_ms"]) for r in eval_rows}
out = Path("results/m4/phase8/compare")
out.mkdir(parents=True, exist_ok=True)
models = sorted({(r["variant"], int(r["seed"])) for r in eval_rows})
rows = []
anchors = []
for index in range(3):
    ident = f"test-{index}-{Path(config['maps']['test'][index]).stem}"
    for actual in range(3):
        trace = traces / (f"{ident}.actual.m3bin" if actual == 0 else f"{ident}.actual{actual}.m3bin")
        jobs = []
        for partition in ("uniform-16", "adaptive-query"):
            jobs.append(("greedy", partition, None, None))
        for variant, seed in models:
            for partition in ("uniform-16", "adaptive-query"):
                jobs.append(("dqn", partition, (variant, seed), None))
        for name, param, cfg, prior in (
                ("PR Quadtree", selection["classic"]["PR Quadtree"], None, None),
                ("Uniform Grid", selection["classic"]["Uniform Grid"], None, None),
                ("Fixed Regional Scan", 1, None, None),
                ("Fixed Local Quadtree", 1, None, None),
                ("Cost-Aware Greedy Regional Index Selection", 1, selector, None),
                ("Cost-Aware Greedy Regional Index Selection (Scenario)", 1, selector,
                 "nominal_samples_m32.csv")):
            jobs.append(("classic", name, (param, cfg, prior), None))
        random.Random(8100 + index * 10 + actual).shuffle(jobs)
        jobs = [("anchor", "a", None, None)] + jobs + [("anchor", "b", None, None)]
        for kind, name, extra, prior in jobs:
            if kind == "anchor":
                args = ["build_m3/spatial_m3.exe", "measure", "Fixed Regional Scan", str(fine),
                        str(quad), "1", "0", str(trace), "NONE"]
                label = "anchor"
            elif kind == "greedy":
                args = ["build_m3/spatial_m3.exe", "partition_greedy", str(fine), str(quad),
                        "0", str(trace), str(plans / f"{ident}.{name}.plan"), str(selector),
                        "1", "0"]
                label = f"{name} + prior cost greedy"
            elif kind == "dqn":
                partition = name
                variant, seed = extra
                plan = eval_plans / f"{variant}-{seed}-{ident}-{partition}-a{actual}.plan"
                args = ["build_m3/spatial_m3.exe", "partition", str(fine), str(quad), "0",
                        str(trace), str(plan), "0"]
                label = f"{partition} + prior Double DQN [{variant}-{seed}]"
            else:
                param, cfg, prior_name = extra
                args = ["build_m3/spatial_m3.exe", "measure", name, str(fine), str(quad),
                        str(param), "0", str(trace), str(cfg) if cfg else "NONE"]
                if prior_name:
                    args.append(str(scen / f"{ident}.{prior_name}"))
                label = name
            result = subprocess.run(args, capture_output=True, text=True)
            if result.returncode:
                raise SystemExit(result.stderr)
            head = dict(token.split("=", 1) for token in result.stdout.strip().split())
            online = float(head["online_ms"])
            if kind == "dqn":
                online += inference.get((extra[0], extra[1], name, ident, actual), 0.0)
            rows.append({"scenario": ident, "actual": actual, "role": kind,
                         "method": label, "online_ms": online,
                         "decide_ms": head.get("decide_ms", 0.0),
                         "leaves": head.get("leaves", 0)})
            if kind == "anchor":
                anchors.append(online)
        print(f"{ident} a{actual} done", flush=True)
with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
threshold = 1.15
drift = (max(anchors) / min(anchors)) if anchors and min(anchors) > 0 else 0.0
json_file(out / "manifest.json", {"anchor_runs": len(anchors),
                                  "anchor_median": sorted(anchors)[len(anchors) // 2],
                                  "anchor_min": min(anchors), "anchor_max": max(anchors),
                                  "anchor_drift": drift, "drift_threshold": threshold,
                                  "batch_valid": drift <= threshold,
                                  "environment": {"platform": platform.platform(),
                                                  "cpu_count": os.cpu_count()},
                                  "note": "DQN rows add the model inference to the replay time"})
if drift > threshold:
    print(f"WARNING: anchor drift {drift:.2f} exceeds {threshold}, batch marked invalid")
print("done")
