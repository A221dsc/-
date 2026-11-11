"""H0.1 controlled workloads (frozen parameters) for the partition comparison.

Eight scenarios share a 4096 m world, 20k entities and 24 windows x 50 steps.
Only the controlled factor changes per scenario so the effect is attributable.
Queries are rectangles; updates are position changes.
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
WINDOWS = STEPS // WINDOW
N = 20000
RECTS = {1: [(0, 0, 64, 64, 1)], 4: [], 8: [], 16: [], 32: []}
for cells in (4, 8, 16, 32):
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
    points = []
    if spec["dist"] == "uniform":
        points = [(rng.random() * 4096, rng.random() * 4096) for _ in range(N)]
    else:
        for _ in range(N):
            if rng.random() < 0.8:
                x, y = 300 + 1000 * rng.random(), 300 + 1000 * rng.random()
            else:
                x, y = rng.random() * 4096, rng.random() * 4096
            points.append((x, y))
    initial = list(points)
    events = []
    hot_centers = None
    for step in range(STEPS):
        if step and step % WINDOW == 0:
            events.append(("P", step, 0.0, 0.0, 0.0, 0.0))
        window = step // WINDOW
        if spec.get("migrate"):
            cx = 400 + (window % 4) * 1000.0
            cy = 400 + ((window * 7) % 4) * 1000.0
            centers = [(cx, cy)]
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
                                  for (x0, y0, x1, y1, level) in rects) + "\n", encoding="utf-8")
        plans[cells] = path
    return plans


def run(binary, trace, plan, profile=0):
    result = subprocess.run([str(binary), "hpartition", "160", "32", "0", str(trace),
                             str(plan), "0", str(profile)], capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(result.stderr)
    return {k: float(v.split("/")[0]) for k, v in
            (token.split("=", 1) for token in result.stdout.split())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, spec in SCENARIOS.items():
        points, events = build(17 + len(name), spec)
        trace = args.output / f"{name}.m3bin"
        write_trace(trace, WORLD, points, events)
        plans = write_plans(args.output)
        # direct full-map Quad reference
        direct = subprocess.run([str(args.binary), "measure", "PR Quadtree", "160", "32", "32",
                                 "0", str(trace), "NONE"], capture_output=True, text=True)
        if direct.returncode:
            raise SystemExit(direct.stderr)
        head = dict(token.split("=", 1) for token in direct.stdout.split())
        rows.append({"scenario": name, "method": "direct Quad", "online_ms": float(head["online_ms"]),
                     "query_ms": float(head.get("query_ms", 0)),
                     "update_ms": float(head.get("update_ms", 0)), "nodes": 0})
        for cells, plan in plans.items():
            out = run(args.binary, trace, plan)
            rows.append({"scenario": name, "method": f"partition {cells}x{cells}",
                         "online_ms": out["online_ms"], "query_ms": out["query_ms"],
                         "update_ms": out["update_ms"], "nodes": out["index_nodes"]})
        print(f"{name} done", flush=True)
    with (args.output / "controlled.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("controlled loads done")


if __name__ == "__main__":
    main()
