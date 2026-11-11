"""Controlled separability scenarios: query-dominant vs update-dominant leaves.

A 2x2 partition over a 2 km world. Leaf 0 holds the query-dominant workload
(800 entities, 20 repeated queries per step, few updates); leaf 1 holds the
update-dominant workload (800 entities, 200 updates per step, few queries). The
fixed-mode replays compare SCAN / FINE_GRID / LOCAL_QUAD on the real index, and
the counterfactual harness measures the switch benefit for the target leaf.
"""
import argparse
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from m4_scenario import write_trace  # noqa: E402

STEPS = 1200
WINDOW = 50
WINDOWS = STEPS // WINDOW
WORLD = (0.0, 0.0, 2000.0, 2000.0)


def build(kind, seed=7):
    rng = random.Random(seed)
    n0, n1 = 800, 800
    points = []
    for _ in range(n0):
        points.append((200 + 600 * rng.random(), 200 + 600 * rng.random()))
    for _ in range(n1):
        points.append((1200 + 600 * rng.random(), 200 + 600 * rng.random()))
    ids0 = list(range(1, n0 + 1))
    ids1 = list(range(n0 + 1, n0 + n1 + 1))
    events = []
    for step in range(STEPS):
        if step and step % WINDOW == 0:
            events.append(("P", step, 0.0, 0.0, 0.0, 0.0))
        if kind == "query":
            for _ in range(20):
                cx = 300 + 400 * rng.random()
                cy = 300 + 400 * rng.random()
                events.append(("R", 0, cx - 40, cy - 40, cx + 40, cy + 40))
            for ident in rng.sample(ids0, 6):
                x, y = points[ident - 1]
                points[ident - 1] = (x + rng.uniform(-2, 2), y + rng.uniform(-2, 2))
                events.append(("U", ident, points[ident - 1][0], points[ident - 1][1], 0, 0))
        else:
            for ident in rng.sample(ids1, 200):
                x, y = points[ident - 1]
                points[ident - 1] = (x + rng.uniform(-2, 2), y + rng.uniform(-2, 2))
                events.append(("U", ident, points[ident - 1][0], points[ident - 1][1], 0, 0))
            for _ in range(2):
                cx = 300 + 400 * rng.random()
                cy = 300 + 400 * rng.random()
                events.append(("R", 0, cx - 40, cy - 40, cx + 40, cy + 40))
    return points, events


def leaf_plan(path):
    rects = [(0, 0, 32, 32), (32, 0, 64, 32), (0, 32, 32, 64), (32, 32, 64, 64)]
    lines = [f"L {x0} {y0} {x1} {y1} 1" for (x0, y0, x1, y1) in rects]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rects


def mode_plan(path, rects, mode):
    lines = [f"L {x0} {y0} {x1} {y1} 1" for (x0, y0, x1, y1) in rects]
    for leaf in range(len(rects)):
        if mode != 0:
            lines.append(f"M 0 {leaf} {mode} 1.0")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(binary, trace, plan):
    result = subprocess.run([str(binary), "partition", "160", "32", "0", str(trace), str(plan),
                             "0"], capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(result.stderr)
    return {k: float(v) for k, v in (token.split("=", 1) for token in result.stdout.split())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind, target_leaf in (("query", 0), ("update", 1)):
        points, events = build(kind)
        trace = args.output / f"controlled-{kind}.m3bin"
        write_trace(trace, WORLD, points, events)
        base = args.output / f"controlled-{kind}.base.plan"
        rects = leaf_plan(base)
        for mode, name in ((0, "SCAN"), (1, "FINE_GRID"), (2, "LOCAL_QUAD")):
            plan = args.output / f"controlled-{kind}.{name}.plan"
            mode_plan(plan, rects, mode)
            out = run(args.binary, trace, plan)
            rows.append({"scenario": kind, "mode": name, "online_ms": out["online_ms"],
                         "query_ms": out["query_ms"], "update_ms": out["update_ms"],
                         "switch_ms": out["switch_ms"]})
            print(f"{kind} {name}: {out['online_ms']:.2f} ms "
                  f"(q {out['query_ms']:.2f}, u {out['update_ms']:.2f})", flush=True)
        for horizon in (1, 4):
            for pair in range(5):
                order = "1" if pair % 2 else "0"
                result = subprocess.run(
                    [str(args.binary), "partition_cf", "160", "32", "0", str(trace), str(base),
                     "8", str(target_leaf), "0", "2", str(horizon), "0", "0", order],
                    capture_output=True, text=True)
                if result.returncode:
                    raise SystemExit(result.stderr)
                out = {}
                for token in result.stdout.split():
                    key, value = token.split("=", 1)
                    if key in ("window_q", "window_u"):
                        continue
                    if "/" in value:
                        value = value.split("/")[0]
                    out[key] = float(value)
                rows.append({"scenario": kind, "mode": f"CF_QUAD_h{horizon}", "pair": pair,
                             "online_ms": out["net_ms"], "query_ms": out["query_a_ms"],
                             "update_ms": out["query_b_ms"],
                             "switch_ms": out["switch_b_ms"],
                             "obs_prev_q": out["obs_prev_q"], "prior_q": out["prior_q"],
                             "actual_future_q": out["actual_future_q"],
                             "state_equal": out["state_equal"],
                             "mismatches": out["mismatches"]})
                print(f"{kind} CF quad h{horizon} pair {pair}: net {out['net_ms']:+.2f} "
                      f"(obs {out['obs_prev_q']:.0f}, prior {out['prior_q']:.0f}, "
                      f"actual {out['actual_future_q']:.0f})", flush=True)
    import csv
    fieldnames = sorted({key for row in rows for key in row})
    with (args.output / "controlled_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
