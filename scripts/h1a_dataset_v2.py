"""H1-A.1 dataset: original 8 exploratory scenarios + frozen held-out combos.

Geometry: cells is the per-axis count k (1/4/8/16/32); leaves = k^2; leaf side =
WORLD/k. The direct full-map Quad is an independent candidate whose geometry is
the k=1 limit (one leaf of the whole world), so the physical features stay the
same across candidates. Methods are shuffled per trace and a sentinel trace is
anchored before/after the batch.
"""
import argparse
import csv
import json
import random
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
ORIG = {
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
                            migrate=True, period=1),
    "G_hot_vs_dense": dict(dist="concentrated", qbox=30, qps=40, ups=5, move=2.0, hot=1,
                           hp=300, disjoint=True),
    "H_mobility": dict(dist="uniform", qbox=100, qps=20, ups=100, move=40.0, hot=None,
                       hp=None),
}
NEW = {
    "N1_qbox100": dict(dist="uniform", qbox=100, qps=20, ups=5, move=2.0, hot=None, hp=None),
    "N2_qbox500": dict(dist="uniform", qbox=500, qps=15, ups=5, move=2.0, hot=None, hp=None),
    "N3_hot2_qps60": dict(dist="uniform", qbox=50, qps=60, ups=5, move=2.0, hot=2, hp=300),
    "N4_ups50": dict(dist="uniform", qbox=200, qps=30, ups=50, move=2.0, hot=None, hp=None),
    "N5_move20": dict(dist="uniform", qbox=100, qps=20, ups=5, move=20.0, hot=None, hp=None),
    "N6_hot8": dict(dist="uniform", qbox=50, qps=40, ups=5, move=2.0, hot=8, hp=200),
    "N7_small_ups20": dict(dist="uniform", qbox=30, qps=40, ups=20, move=2.0, hot=None,
                           hp=None),
    "N8_large_ups20": dict(dist="uniform", qbox=1000, qps=10, ups=20, move=2.0, hot=None,
                           hp=None),
    "N9_mix05": dict(dist="concentrated", qbox=200, qps=20, ups=5, move=2.0, hot=None,
                     hp=None, density_mix=0.5),
    "N10_fast_migrate": dict(dist="uniform", qbox=50, qps=30, ups=5, move=2.0, hot=1, hp=300,
                             migrate=True, period=3),
    "N11_slow_migrate": dict(dist="uniform", qbox=30, qps=30, ups=5, move=2.0, hot=1, hp=300,
                             migrate=True, period=12),
    "N12_qbox300_ups200": dict(dist="uniform", qbox=300, qps=20, ups=200, move=2.0, hot=None,
                               hp=None),
}


def build(seed, spec):
    rng = random.Random(seed)
    mix = spec.get("density_mix", 0.8)
    if spec["dist"] == "uniform":
        points = [(rng.random() * 4096, rng.random() * 4096) for _ in range(N)]
    else:
        points = []
        for _ in range(N):
            if rng.random() < mix:
                points.append((300 + 1000 * rng.random(), 300 + 1000 * rng.random()))
            else:
                points.append((rng.random() * 4096, rng.random() * 4096))
    initial = list(points)
    events = []
    period = spec.get("period", 6)
    for step in range(STEPS):
        if step and step % WINDOW == 0:
            events.append(("P", step, 0.0, 0.0, 0.0, 0.0))
        window = step // WINDOW
        if spec.get("migrate"):
            centers = [(400 + ((window // period) % 4) * 1000.0,
                        400 + (((window // period) * 7) % 4) * 1000.0)]
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


def run_hpartition(binary, trace, plan):
    result = subprocess.run([str(binary), "hpartition", "160", "32", "0", str(trace),
                             str(plan), "0", "0"], capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(result.stderr)
    return {k: float(v.split("/")[0]) for k, v in
            (token.split("=", 1) for token in result.stdout.split())}


def drift(a, b):
    return max(a, b) / min(a, b) if min(a, b) > 0 else float("inf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--set", choices=["orig", "new"], default="new")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plans = write_plans(args.output)
    scenarios = ORIG if args.set == "orig" else NEW
    sentinel = None
    sent_start = None
    rows = []
    for name, spec in scenarios.items():
        for seed in range(args.seeds):
            points, events = build(1000 * (list(scenarios).index(name) + 1) + seed, spec)
            trace = args.output / f"{name}.s{seed}.m3bin"
            write_trace(trace, WORLD, points, events)
            if sentinel is None:
                sentinel = args.output / f"sentinel.m3bin"
                write_trace(sentinel, WORLD, points, events)
            def scan(t):
                out = subprocess.run([str(args.binary), "measure", "Fixed Regional Scan",
                                      "160", "32", "1", "0", str(t), "NONE"],
                                     capture_output=True, text=True)
                if out.returncode:
                    raise SystemExit(out.stderr)
                return float(dict(token.split("=", 1)
                                  for token in out.stdout.split())["online_ms"])
            trace_start = scan(trace)
            order = [("direct", 0)] + [("partition", c) for c in (1, 4, 8, 16, 32)]
            random.Random(7100 + list(scenarios).index(name) * 10 + seed).shuffle(order)
            base = dict(scenario=name, set=args.set, seed=seed, dist=spec["dist"],
                        qbox=spec["qbox"], qps=spec["qps"], ups=spec["ups"],
                        move=spec["move"], hot=spec["hot"] or 0, hp=spec["hp"] or 0,
                        migrate=int(bool(spec.get("migrate"))),
                        period=spec.get("period", 6),
                        disjoint=int(bool(spec.get("disjoint"))),
                        density_mix=spec.get("density_mix", 0.8))
            for kind, cells in order:
                if kind == "direct":
                    direct = subprocess.run([str(args.binary), "measure", "PR Quadtree",
                                             "160", "32", "32", "0", str(trace), "NONE"],
                                            capture_output=True, text=True)
                    if direct.returncode:
                        raise SystemExit(direct.stderr)
                    head = dict(token.split("=", 1) for token in direct.stdout.split())
                    rows.append({**base, "candidate": 0, "method": "direct Quad",
                                 "online_ms": float(head["online_ms"]),
                                 "query_ms": float(head.get("query_ms", 0)),
                                 "update_ms": float(head.get("update_ms", 0)),
                                 "build_ms": float(head.get("build_ms", 0)),
                                 "nodes": 0,
                                 "index_delta_mib": float(head.get("prebuild_rss_mb", 0))})
                else:
                    out = run_hpartition(args.binary, trace, plans[cells])
                    rows.append({**base, "candidate": cells, "method": f"partition {cells}x{cells}",
                                 "online_ms": out["online_ms"], "query_ms": out["query_ms"],
                                 "update_ms": out["update_ms"], "build_ms": out["build_ms"],
                                 "nodes": out["index_nodes"],
                                 "index_delta_mib": out.get("index_delta_mib", 0.0)})
            trace_end = scan(trace)
            print(f"{name} s{seed} done anchor {trace_start:.1f}->{trace_end:.1f}", flush=True)
    # Direct Quad memory proxy: the same Quad as the 1x1 partition, so copy the
    # measured 1x1 live delta and node count for the direct rows of that trace.
    by_trace = {}
    for r in rows:
        if r["candidate"] == 1:
            by_trace[(r["scenario"], r["seed"])] = r
    for r in rows:
        if r["candidate"] == 0:
            ref = by_trace.get((r["scenario"], r["seed"]))
            if ref is not None:
                r["index_delta_mib"] = ref["index_delta_mib"]
                r["nodes"] = ref["nodes"]
    # Sentinel anchors: fixed reference scan on the sentinel trace before/after.
    def scan():
        out = subprocess.run([str(args.binary), "measure", "Fixed Regional Scan", "160", "32",
                              "1", "0", str(sentinel), "NONE"], capture_output=True, text=True)
        if out.returncode:
            raise SystemExit(out.stderr)
        head = dict(token.split("=", 1) for token in out.stdout.split())
        return float(head["online_ms"])
    scan()
    start = scan()
    end = scan()
    ratio = drift(start, end) if end else 0.0
    out_path = args.output / ("dataset_v2_new.csv" if args.set == "new" else "dataset_v2_orig.csv")
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / f"manifest_{args.set}.json").write_text(
        json.dumps({"set": args.set, "rows": len(rows), "scenarios": len(scenarios),
                    "sentinel": {"start": start, "end": end, "drift": ratio},
                    "threshold": 1.15, "valid": ratio <= 1.15}, indent=1) + "\n",
        encoding="utf-8")
    print(f"{args.set} rows {len(rows)} sentinel drift {ratio:.3f} -> {out_path}")


if __name__ == "__main__":
    main()
