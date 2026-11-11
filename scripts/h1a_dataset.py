"""H1-A dataset: controlled scenarios x seeds x candidate granularities.

Eight frozen scenarios, three independent load seeds, five candidate partitions
(1/4/8/16/32) plus the direct full-map Quad. Every row stores the scenario
parameters (the nominal "想定"), the measured C++ cost and the index size.
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from m4_scenario import write_trace  # noqa: E402

WORLD = (0.0, 0.0, 4096.0, 4096.0)
STEPS = 1200
WINDOW = 50
N = 20000
RECTS = {}
for cells in (1, 4, 8, 16, 32):
    step = 64 // cells
    RECTS[cells] = [(bx * step, by * step, (bx + 1) * step, (by + 1) * step, 1)
                    for by in range(cells) for bx in range(cells)]
SCENARIOS = {
    "A_uniform_uniform": dict(dist="uniform", qbox=200, qps=20, ups=5, move=2.0, hot=None,
                              hp=None),
    "B_concentrated": dict(dist="concentrated", qbox=200, qps=20, ups=5, move=2.0, hot=None,
                           hp=None),
    "C_small_hot": dict(dist="uniform", qbox=20, qps=40, ups=5, move=2.0, hot=4, hp=300),
    "D_large_query": dict(dist="uniform", qbox=1000, qps=10, ups=5, move=2.0, hot=None,
                          hp=None),
    "E_update_heavy": dict(dist="uniform", qbox=20, qps=20, ups=200, move=2.0, hot=None,
                           hp=None),
    "F_migrating_hot": dict(dist="uniform", qbox=30, qps=40, ups=5, move=2.0, hot=1, hp=300,
                            migrate=True),
    "G_hot_vs_dense": dict(dist="concentrated", qbox=30, qps=40, ups=5, move=2.0, hot=1,
                           hp=300, disjoint=True),
    "H_mobility": dict(dist="uniform", qbox=100, qps=20, ups=100, move=40.0, hot=None,
                       hp=None),
}


def build(seed, spec):
    import random
    rng = random.Random(seed)
    if spec["dist"] == "uniform":
        points = [(rng.random() * 4096, rng.random() * 4096) for _ in range(N)]
    else:
        points = []
        for _ in range(N):
            if rng.random() < 0.8:
                points.append((300 + 1000 * rng.random(), 300 + 1000 * rng.random()))
            else:
                points.append((rng.random() * 4096, rng.random() * 4096))
    initial = list(points)
    events = []
    for step in range(STEPS):
        if step and step % WINDOW == 0:
            events.append(("P", step, 0.0, 0.0, 0.0, 0.0))
        window = step // WINDOW
        if spec.get("migrate"):
            centers = [(400 + (window % 4) * 1000.0, 400 + ((window * 7) % 4) * 1000.0)]
        elif spec.get("disjoint"):
            centers = [(3000.0, 3000.0)]
        elif spec["hot"]:
            centers = [(700 + 900 * (i % 2), 700 + 900 * (i // 2))
                       for i in range(spec["hot"])]
        else:
            centers = None
        for _ in range(spec["qps"]):
            if centers:
                cx, cy = centers[rng.randrange(len(centers))]
                cx += rng.uniform(-spec["hp"] / 2, spec["hp"] / 2)
                cy += rng.uniform(-spec["hp"] / 2, spec["hp"] / 2)
            else:
                cx, cy = rng.uniform(200, 3896), rng.uniform(200, 3896)
            half = spec["qbox"] / 2
            events.append(("R", 0, cx - half, cy - half, cx + half, cy + half))
        for _ in range(spec["ups"]):
            ident = rng.randrange(1, N + 1)
            x, y = points[ident - 1]
            x = min(4095.0, max(1.0, x + rng.uniform(-spec["move"], spec["move"])))
            y = min(4095.0, max(1.0, y + rng.uniform(-spec["move"], spec["move"])))
            points[ident - 1] = (x, y)
            events.append(("U", ident, x, y, 0.0, 0.0))
    return initial, events


def write_plans(prefix):
    plans = {}
    for cells, rects in RECTS.items():
        path = prefix / f"{cells}.plan"
        path.write_text("\n".join(f"L {x0} {y0} {x1} {y1} {level}"
                                  for (x0, y0, x1, y1, level) in rects) + "\n",
                        encoding="utf-8")
        plans[cells] = path
    return plans


def run(binary, trace, plan):
    result = subprocess.run([str(binary), "hpartition", "160", "32", "0", str(trace),
                             str(plan), "0", "0"], capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(result.stderr)
    return {k: float(v.split("/")[0]) for k, v in
            (token.split("=", 1) for token in result.stdout.split())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plans = write_plans(args.output)
    rows = []
    for name, spec in SCENARIOS.items():
        for seed in range(args.seeds):
            points, events = build(1000 * (list(SCENARIOS).index(name) + 1) + seed, spec)
            trace = args.output / f"{name}.s{seed}.m3bin"
            write_trace(trace, WORLD, points, events)
            direct = subprocess.run([str(args.binary), "measure", "PR Quadtree", "160", "32",
                                     "32", "0", str(trace), "NONE"],
                                    capture_output=True, text=True)
            if direct.returncode:
                raise SystemExit(direct.stderr)
            head = dict(token.split("=", 1) for token in direct.stdout.split())
            base = dict(scenario=name, seed=seed, dist=spec["dist"], qbox=spec["qbox"],
                        qps=spec["qps"], ups=spec["ups"], move=spec["move"],
                        hot=spec["hot"] or 0, hp=spec["hp"] or 0,
                        migrate=int(bool(spec.get("migrate"))),
                        disjoint=int(bool(spec.get("disjoint"))))
            rows.append({**base, "cells": 1, "method": "direct Quad",
                         "online_ms": float(head["online_ms"]),
                         "query_ms": float(head.get("query_ms", 0)),
                         "update_ms": float(head.get("update_ms", 0)),
                         "build_ms": float(head.get("build_ms", 0)),
                         "nodes": 0, "index_delta_mib": 0.0})
            for cells, plan in plans.items():
                out = run(args.binary, trace, plan)
                rows.append({**base, "cells": cells, "method": f"partition {cells}x{cells}",
                             "online_ms": out["online_ms"], "query_ms": out["query_ms"],
                             "update_ms": out["update_ms"], "build_ms": out["build_ms"],
                             "nodes": out["index_nodes"],
                             "index_delta_mib": out.get("index_delta_mib", 0.0)})
            print(f"{name} s{seed} done", flush=True)
    with (args.output / "dataset.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"dataset rows {len(rows)}")


if __name__ == "__main__":
    main()
