"""Partition ablation: uniform vs load-aware adaptive partitions with the same
cost-greedy per-leaf index selection, measured through the real C++ partition
index on the same test traces.
"""
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from prepare import json_file, sha  # noqa: E402

SIDE = 16
SUB = 64
MODES = 3


def read_prior(path):
    """Mean prior load per 16x16 cell per window: q, entity, update, coverage."""
    groups = {}
    with Path(path).open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            f = line.split(",")
            window, region = int(f[1]), int(f[2])
            groups.setdefault((window, region), []).append(
                (float(f[3]), float(f[4]), float(f[5]), float(f[6])))
    windows = max(w for w, _ in groups) + 1
    out = np.zeros((windows, SIDE * SIDE, 4))
    for (window, region), values in groups.items():
        out[window, region] = np.mean(np.asarray(values), axis=0)
    return out


def cost(model, mode, load):
    q, n, u, c = load
    query = np.array([q, q * n, n * c, q * math.sqrt(max(n, 0.0)), c])
    update = np.array([u, u * math.sqrt(max(n, 0.0))])
    return float(query @ np.array(model["query"][mode]) + update @ np.array(model["update"][mode]))


def switch_cost(model, source, target):
    names = ["SCAN", "FINE_GRID", "LOCAL_QUAD"]
    if source == target:
        return 0.0
    return float(np.array([1.0, 1.0]) @ np.array(model["switch"][names[source]][names[target]]))


def partition_uniform(cells):
    step = SUB // cells
    rects = []
    for by in range(cells):
        for bx in range(cells):
            rects.append((bx * step, by * step, (bx + 1) * step, (by + 1) * step, 1))
    return rects


def score_cells(prior, weight_query, weight_entity, weight_update):
    q = prior[:, :, 0].mean(axis=0)
    n = prior[:, :, 1].mean(axis=0)
    u = prior[:, :, 2].mean(axis=0)
    def norm(v):
        v = v.astype(float)
        return v / (v.mean() + 1e-9) if v.mean() > 0 else v
    score = weight_query * norm(q) + weight_entity * norm(n) + weight_update * norm(u)
    return score.reshape(SIDE, SIDE)


def partition_adaptive(prior, weight_query, weight_entity, weight_update):
    """Cold 2x2 base-cell blocks merge; hot base cells split into 2x2 sub-cells."""
    score = score_cells(prior, weight_query, weight_entity, weight_update)
    low = np.percentile(score, 25)
    high = np.percentile(score, 75)
    rects = []
    for block_y in range(0, SIDE, 2):
        for block_x in range(0, SIDE, 2):
            block = [score[block_y + dy, block_x + dx] for dy in (0, 1) for dx in (0, 1)]
            if max(block) < low:
                rects.append((block_x * 4, block_y * 4, (block_x + 2) * 4, (block_y + 2) * 4, 1))
                continue
            for dy in (0, 1):
                for dx in (0, 1):
                    cell_x, cell_y = block_x + dx, block_y + dy
                    value = score[cell_y, cell_x]
                    if value > high:
                        for qy in (0, 1):
                            for qx in (0, 1):
                                x0 = cell_x * 4 + qx * 2
                                y0 = cell_y * 4 + qy * 2
                                rects.append((x0, y0, x0 + 2, y0 + 2, 2))
                    else:
                        rects.append((cell_x * 4, cell_y * 4, cell_x * 4 + 4,
                                      cell_y * 4 + 4, 1))
    return rects


def leaf_loads(prior, rects):
    """Aggregate the per-cell prior to each leaf by area share."""
    windows = prior.shape[0]
    out = np.zeros((windows, len(rects), 4))
    for index, (x0, y0, x1, y1, _) in enumerate(rects):
        cells = []
        for cy in range(y0 // 4, (y1 + 3) // 4):
            for cx in range(x0 // 4, (x1 + 3) // 4):
                cells.append(cy * SIDE + cx)
        area_share = ((x1 - x0) * (y1 - y0)) / (4.0 * 4.0) / len(cells)
        for w in range(windows):
            out[w, index] = prior[w, cells].sum(axis=0) * area_share
    return out


def greedy_schedule(loads, model):
    """Myopic cost greedy per leaf per window, with switch cost and benefit."""
    windows, leaves, _ = loads.shape
    modes = [0] * leaves
    entries = []
    for w in range(windows):
        for leaf in range(leaves):
            current = modes[leaf]
            best, best_mode = None, current
            for mode in range(MODES):
                value = cost(model, mode, loads[w, leaf]) + switch_cost(model, current, mode)
                if best is None or value < best:
                    best, best_mode = value, mode
            if best_mode != current:
                benefit = (cost(model, current, loads[w, leaf]) -
                           (best - switch_cost(model, current, current)))
                entries.append((w, leaf, best_mode, max(0.0, benefit)))
                modes[leaf] = best_mode
    return entries


def write_plan(path, rects, entries):
    lines = [f"L {x0} {y0} {x1} {y1} {level}" for (x0, y0, x1, y1, level) in rects]
    lines += [f"M {w} {leaf} {mode} {benefit:.6f}" for (w, leaf, mode, benefit) in entries]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("scenarios", type=Path)
    parser.add_argument("binary", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model = json.loads((args.calibration / "cost_model.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "plans").mkdir(exist_ok=True)
    rows = []
    variants = [
        ("uniform-8", lambda p: partition_uniform(8)),
        ("uniform-16", lambda p: partition_uniform(16)),
        ("uniform-32", lambda p: partition_uniform(32)),
        ("adaptive-entity", lambda p: partition_adaptive(p, 0.0, 1.0, 0.0)),
        ("adaptive-query", lambda p: partition_adaptive(p, 1.0, 0.0, 0.0)),
        ("adaptive-combined", lambda p: partition_adaptive(p, 0.5, 0.3, 0.2)),
    ]
    for index in range(3):
        ident = f"test-{index}-{Path(config['maps']['test'][index]).stem}"
        prior = read_prior(args.scenarios / "scenarios" / f"{ident}.nominal_samples_m32.csv")
        for name, builder in variants:
            rects = builder(prior)
            loads = leaf_loads(prior, rects)
            entries = greedy_schedule(loads, model)
            plan = args.output / "plans" / f"{ident}.{name}.plan"
            write_plan(plan, rects, entries)
            for actual in range(3):
                trace = args.scenarios / "traces" / (
                    f"{ident}.actual.m3bin" if actual == 0 else f"{ident}.actual{actual}.m3bin")
                result = subprocess.run([str(args.binary), "partition", "160", "32", "0",
                                         str(trace), str(plan), "0"],
                                        capture_output=True, text=True)
                if result.returncode:
                    raise SystemExit(result.stderr)
                head = dict(token.split("=", 1) for token in result.stdout.strip().split())
                rows.append({"scenario": ident, "actual": actual, "partition": name,
                             "leaves": int(head["leaves"]), "online_ms": head["online_ms"],
                             "query_ms": head["query_ms"], "update_ms": head["update_ms"],
                             "switch_ms": head["switch_ms"],
                             "promotions": head["promotions"],
                             "plan_sha256": sha(plan)[:16]})
                print(f"{ident} a{actual} {name}: {head['online_ms']} ms "
                      f"({head['leaves']} leaves)", flush=True)
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_file(args.output / "manifest.json", {"variants": [v[0] for v in variants],
                                              "plan_format": "L x0 y0 x1 y1 level; M window leaf mode benefit",
                                              "scheduled_by": "myopic cost greedy on the 32-sample prior"})
    print("done")


if __name__ == "__main__":
    main()
