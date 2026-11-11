"""Single-session phase-5 comparison with interleaved scan anchors.

RL rows add the measured per-trace model inference time to the index replay time.
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
base = Path("results/m4/calibration/selector_final.txt")
tmp = Path("results/m4/phase3_v2/tmp")
quantile = tmp / "selector_2_1.00_1.txt"
scen = Path("data/processed/m4/scenarios")
traces = Path("data/processed/m4/traces")
plans = Path("results/m4/phase5/eval/plans")
out = Path("results/m4/phase5/compare")
out.mkdir(parents=True, exist_ok=True)

eval_rows = list(csv.DictReader(open("results/m4/phase5/eval/metrics.csv")))
inference = {}
for r in eval_rows:
    inference[(r["variant"], int(r["seed"]), r["scenario"], int(r["actual"]))] = \
        float(r["inference_ms"])
rl = [("Historical-Load Double DQN Regional Index Selection", "historical", seed)
      for seed in (11, 22, 33)] + \
     [("Scenario-Forward-Enhanced Double DQN Regional Index Selection", "scenario", seed)
      for seed in (11, 22, 33)]
methods = [("Fixed Regional Scan", None, None, 1),
           ("PR Quadtree", None, None, selection["classic"]["PR Quadtree"]),
           ("Cost-Aware Greedy Regional Index Selection", base, None, 1),
           ("Uncertainty-Aware Feed-Forward Regional Index Selection", base,
            "nominal_samples_m32.csv", 1),
           ("Quantile-Cost-Penalized Feed-Forward Regional Index Selection", quantile,
            "nominal_samples_m32.csv", 1)]
rows = []
anchors = []
for index in range(3):
    ident = f"test-{index}-{Path(config['maps']['test'][index]).stem}"
    for actual in range(3):
        trace = traces / (f"{ident}.actual.m3bin" if actual == 0 else f"{ident}.actual{actual}.m3bin")
        order = list(methods)
        random.Random(9100 + index * 10 + actual).shuffle(order)
        batch = [("anchor-start", "Fixed Regional Scan", None, None, 1)] + \
                [("method", m, cfg, prior, param) for (m, cfg, prior, param) in order] + \
                [("anchor-end", "Fixed Regional Scan", None, None, 1)]
        for kind_name, method, cfg, prior_name, param in batch:
            args = ["build_m3/spatial_m3.exe", "measure", method, str(fine), str(quad),
                    str(param), "0", str(trace), str(cfg) if cfg else "NONE"]
            if prior_name:
                args.append(str(scen / f"{ident}.{prior_name}"))
            result = subprocess.run(args, capture_output=True, text=True)
            if result.returncode:
                raise SystemExit(result.stderr)
            head = dict(token.split("=", 1) for token in result.stdout.strip().split())
            rows.append({"scenario": ident, "actual": actual, "role": kind_name,
                         "method": method, "online_ms": head["online_ms"],
                         "index_ms": head["online_ms"], "inference_ms": 0.0})
            if kind_name.startswith("anchor"):
                anchors.append(float(head["online_ms"]))
        for label, variant, seed in rl:
            plan = plans / f"{variant}-{seed}-{ident}-a{actual}.txt"
            result = subprocess.run(["build_m3/spatial_m3.exe", "replayplan", "160", "32", "0",
                                     str(trace), str(plan)], capture_output=True, text=True)
            if result.returncode:
                raise SystemExit(result.stderr)
            head = dict(token.split("=", 1) for token in result.stdout.strip().split())
            extra = inference.get((variant, seed, ident, actual), 0.0)
            rows.append({"scenario": ident, "actual": actual, "role": "rl",
                         "method": f"{label} [{variant}-{seed}]",
                         "online_ms": float(head["online_ms"]) + extra,
                         "index_ms": head["online_ms"], "inference_ms": extra})
        print(f"{ident} a{actual} done", flush=True)
with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
json_file(out / "manifest.json", {"anchor_runs": len(anchors),
                                  "anchor_median": sorted(anchors)[len(anchors) // 2],
                                  "anchor_min": min(anchors), "anchor_max": max(anchors),
                                  "note": "RL online includes model inference"})
print("done")
