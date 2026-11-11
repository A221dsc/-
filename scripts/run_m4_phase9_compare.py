"""M4.9 corrected four-way comparison with per-trace and sentinel anchors."""
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
sentinel_trace = traces / "train-0-brc202d.actual.m3bin"
eval_sets = [("uniform", Path("results/m4/phase9/eval_uniform")),
             ("nonuniform", Path("results/m4/phase9/eval_nonuniform"))]
out = Path("results/m4/phase9/compare")
out.mkdir(parents=True, exist_ok=True)

inference = {}
rl_plans = []
for train_set, directory in eval_sets:
    metrics = list(csv.DictReader(open(directory / "metrics.csv")))
    for r in metrics:
        inference[(train_set, r["variant"], int(r["seed"]), r["partition"], r["scenario"],
                   int(r["actual"]))] = float(r["inference_ms"])
    for path in (directory / "plans").glob("*.plan"):
        parts = path.stem.split("-")
        rl_plans.append((train_set, parts[0], int(parts[1]), path))

rows = []
anchor_pairs = []
sentinel = []
def measure(trace):
    result = subprocess.run(["build_m3/spatial_m3.exe", "measure", "Fixed Regional Scan",
                             str(fine), str(quad), "1", "0", str(trace), "NONE"],
                            capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(result.stderr)
    head = dict(token.split("=", 1) for token in result.stdout.strip().split())
    return float(head["online_ms"])

sentinel.append(("start", measure(sentinel_trace)))
for index in range(3):
    ident = f"test-{index}-{Path(config['maps']['test'][index]).stem}"
    for actual in range(3):
        trace = traces / (f"{ident}.actual.m3bin" if actual == 0 else f"{ident}.actual{actual}.m3bin")
        start = measure(trace)
        jobs = []
        for partition in ("uniform-16", "adaptive-query"):
            jobs.append(("greedy", partition, None, None))
        for train_set, variant, seed, path in rl_plans:
            if not path.name.startswith(f"{variant}-{seed}-{ident}-"):
                continue
            partition = "uniform-16" if "-uniform-16-" in path.name else "adaptive-query"
            if not path.name.endswith(f"-a{actual}.plan"):
                continue
            jobs.append(("dqn", partition, (train_set, variant, seed), path))
        jobs.append(("classic", "PR Quadtree", (selection["classic"]["PR Quadtree"], None, None), None))
        jobs.append(("classic", "Uniform Grid", (selection["classic"]["Uniform Grid"], None, None), None))
        jobs.append(("classic", "Fixed Regional Scan", (1, None, None), None))
        jobs.append(("classic", "Fixed Local Quadtree", (1, None, None), None))
        random.Random(9100 + index * 10 + actual).shuffle(jobs)
        for kind, name, extra, path in jobs:
            if kind == "greedy":
                args = ["build_m3/spatial_m3.exe", "partition_greedy", str(fine), str(quad),
                        "0", str(trace), str(plans / f"{ident}.{name}.plan"), str(selector),
                        "1", "0"]
                label = f"{name} + prior cost greedy"
            elif kind == "dqn":
                train_set, variant, seed = extra
                args = ["build_m3/spatial_m3.exe", "partition", str(fine), str(quad), "0",
                        str(trace), str(path), "0"]
                label = f"{name} + {variant} DQN [trained {train_set}, seed {seed}]"
            else:
                param, _, _ = extra
                args = ["build_m3/spatial_m3.exe", "measure", name, str(fine), str(quad),
                        str(param), "0", str(trace), "NONE"]
                label = name
            result = subprocess.run(args, capture_output=True, text=True)
            if result.returncode:
                raise SystemExit(result.stderr)
            head = dict(token.split("=", 1) for token in result.stdout.strip().split())
            online = float(head["online_ms"])
            if kind == "dqn":
                online += inference.get((extra[0], extra[1], extra[2], name, ident, actual), 0.0)
            rows.append({"scenario": ident, "actual": actual, "role": kind, "method": label,
                         "online_ms": online, "leaves": head.get("leaves", 0)})
        end = measure(trace)
        anchor_pairs.append((ident, actual, start, end))
        print(f"{ident} a{actual} done", flush=True)
sentinel.append(("end", measure(sentinel_trace)))
with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
ratios = [(s, a, s0, e0, e0 / s0 if s0 > 0 else 0.0) for s, a, s0, e0 in anchor_pairs]
threshold = 1.15
valid = all(r[4] <= threshold for r in ratios) and sentinel[1][1] / sentinel[0][1] <= threshold
json_file(out / "manifest.json", {
    "per_trace_anchor": [{"scenario": s, "actual": a, "start": s0, "end": e0, "ratio": r}
                         for s, a, s0, e0, r in ratios],
    "max_per_trace_ratio": max(r[4] for r in ratios),
    "sentinel": {"start": sentinel[0][1], "end": sentinel[1][1],
                 "ratio": sentinel[1][1] / sentinel[0][1]},
    "threshold": threshold, "batch_valid": valid,
    "environment": {"platform": platform.platform(), "cpu_count": os.cpu_count()}})
print(f"max per-trace ratio {max(r[4] for r in ratios):.3f}, sentinel "
      f"{sentinel[1][1]/sentinel[0][1]:.3f}, valid={valid}")
