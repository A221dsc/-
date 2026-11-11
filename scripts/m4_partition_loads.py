"""Recompute per-leaf nominal loads with query-touch semantics.

For every planned nominal query the leaf receives one touch when the query
bounding box (rectangle half width, or the actual circle radius bounding box)
intersects the leaf rectangle, and the covered fraction uses the leaf area as
denominator. Entity counts and update touches come from the nominal reporting
simulation at 64x64 sub-cell resolution. The resulting P lines are consumed by
the C++ online cost-greedy partition benchmark.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from m4_maps import GameMap  # noqa: E402
from m4_scenario import (SUBCELL, build_scenario, planned_query_schedule, prepare,  # noqa: E402
                         simulate)

SIDE = 16
SUB = SUBCELL


def read_plan(path):
    rects, levels = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line or line[0] == "#":
            continue
        f = line.split()
        if f[0] == "L":
            rects.append(tuple(int(v) for v in f[1:5]))
            levels.append(int(f[5]))
    return rects, levels


def leaf_grid(rects):
    grid = np.full((SUB, SUB), -1, dtype=np.int32)
    for index, (x0, y0, x1, y1) in enumerate(rects):
        grid[y0:y1, x0:x1] = index
    return grid


def leaf_query_loads(scenario, game, rects, windows):
    x0, y0, x1, y1 = scenario["world"]
    w, h = (x1 - x0) / SUB, (y1 - y0) / SUB
    boxes = np.array([[x0 + r[0] * w, y0 + r[1] * h, x0 + r[2] * w, y0 + r[3] * h]
                      for r in rects])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    grid = leaf_grid(rects)
    touches = np.zeros((windows, len(rects)), dtype=np.float64)
    coverage = np.zeros((windows, len(rects)), dtype=np.float64)
    schedule = planned_query_schedule(scenario, game)
    window = scenario["decision_window"]
    for step, kind, point, half, tag in schedule:
        bx0, by0 = point[0] - half, point[1] - half
        bx1, by1 = point[0] + half, point[1] + half
        sx0 = int(np.clip((bx0 - x0) / w, 0, SUB - 1))
        sx1 = int(np.clip((bx1 - x0) / w, 0, SUB - 1))
        sy0 = int(np.clip((by0 - y0) / h, 0, SUB - 1))
        sy1 = int(np.clip((by1 - y0) / h, 0, SUB - 1))
        leaves = np.unique(grid[sy0:sy1 + 1, sx0:sx1 + 1])
        wk = min(windows - 1, step // window)
        for leaf in leaves:
            if leaf < 0:
                continue
            lx0, ly0, lx1, ly1 = boxes[leaf]
            ox = max(0.0, min(lx1, bx1) - max(lx0, bx0))
            oy = max(0.0, min(ly1, by1) - max(ly0, by0))
            if ox > 0 and oy > 0:
                touches[wk, leaf] += 1.0
                coverage[wk, leaf] += min(1.0, ox * oy / max(areas[leaf], 1e-9))
    return touches, coverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("scenarios", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ident", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    split, index = "test", int(args.ident.split("-")[1])
    map_name = config["maps"]["test"][index]
    scenario = json.loads((args.scenarios / "scenarios" / f"{args.ident}.json").read_text(encoding="utf-8"))
    game, rebuilt, entities = build_scenario(config, map_name, split, index, scenario["seeds"])
    prepared = prepare(scenario, game, entities)
    state = simulate(scenario, game, prepared, reporting=True)
    windows = state["window_entities_sub"].shape[0]
    rects, levels = read_plan(args.plan)
    grid = leaf_grid(rects)
    flat = grid.reshape(-1)
    leaf_of_sub = np.where(flat >= 0, flat, 0)
    entities = np.zeros((windows, len(rects)))
    updates = np.zeros((windows, len(rects)))
    for w in range(windows):
        entities[w] = np.bincount(leaf_of_sub, weights=state["window_entities_sub"][w],
                                  minlength=len(rects))
        updates[w] = np.bincount(leaf_of_sub, weights=state["window_updates_sub"][w],
                                 minlength=len(rects))
    touches, coverage = leaf_query_loads(scenario, game, rects, windows)
    lines = [f"L {x0} {y0} {x1} {y1} {level}" for (x0, y0, x1, y1), level in zip(rects, levels)]
    for w in range(windows):
        for leaf in range(len(rects)):
            lines.append(f"P {w} {leaf} {touches[w, leaf]:.6f} {entities[w, leaf]:.3f} "
                         f"{updates[w, leaf]:.3f} {coverage[w, leaf]:.6f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{args.ident} {args.plan.name}: {len(rects)} leaves, "
          f"touches/window {touches.sum(axis=1).mean():.1f}, "
          f"entities/window {entities.sum(axis=1).mean():.0f}, "
          f"updates/window {updates.sum(axis=1).mean():.0f}")


if __name__ == "__main__":
    main()
