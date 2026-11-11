"""Controlled heterogeneous scenarios with per-leaf cross configurations.

A single 2x2 instance: leaf 0 is query-dominant (repeated queries, few updates)
and leaf 1 is update-dominant (heavy updates, no queries); leaves 2 and 3 are
cold. The initial positions are kept as a separate copy so the update stream
cannot overwrite the replay origin. The plan P lines carry the *planned*
per-window loads (legal prior, not the actual future events).
"""
import argparse
import csv
import itertools
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from m4_scenario import write_trace  # noqa: E402

STEPS = 1200
WINDOW = 50
WINDOWS = STEPS // WINDOW
WORLD = (0.0, 0.0, 2000.0, 2000.0)
RECTS = [(0, 0, 32, 32), (32, 0, 64, 32), (0, 32, 32, 64), (32, 32, 64, 64)]


def build(seed=7):
    rng = random.Random(seed)
    n0, n1 = 800, 800
    points = [(200 + 600 * rng.random(), 200 + 600 * rng.random()) for _ in range(n0)]
    points += [(1200 + 600 * rng.random(), 200 + 600 * rng.random()) for _ in range(n1)]
    initial = list(points)  # the replay origin must not be the post-update state
    ids0 = list(range(1, n0 + 1))
    ids1 = list(range(n0 + 1, n0 + n1 + 1))
    events = []
    planned = {}

    def bump(window, leaf, dq, du):
        cell = planned.setdefault((window, leaf), [0, 0])
        cell[0] += dq
        cell[1] += du

    for step in range(STEPS):
        if step and step % WINDOW == 0:
            events.append(("P", step, 0.0, 0.0, 0.0, 0.0))
        window = step // WINDOW
        for _ in range(20):
            cx = 300 + 400 * rng.random()
            cy = 300 + 400 * rng.random()
            events.append(("R", 0, cx - 40, cy - 40, cx + 40, cy + 40))
            bump(window, 0, 1, 0)
        for ident in rng.sample(ids0, 6):
            x, y = points[ident - 1]
            points[ident - 1] = (x + rng.uniform(-2, 2), y + rng.uniform(-2, 2))
            events.append(("U", ident, points[ident - 1][0], points[ident - 1][1], 0, 0))
            bump(window, 0, 0, 1)
        for ident in rng.sample(ids1, 200):
            x, y = points[ident - 1]
            points[ident - 1] = (x + rng.uniform(-2, 2), y + rng.uniform(-2, 2))
            events.append(("U", ident, points[ident - 1][0], points[ident - 1][1], 0, 0))
            bump(window, 1, 0, 1)
    return initial, events, planned


def write_plan(path, modes, planned, entities=(800, 800, 0, 0)):
    lines = [f"L {x0} {y0} {x1} {y1} 1" for (x0, y0, x1, y1) in RECTS]
    for leaf, mode in enumerate(modes):
        if mode != 0:
            lines.append(f"M 0 {leaf} {mode} 1.0")
    for window in range(WINDOWS):
        for leaf in range(4):
            q, u = planned.get((window, leaf), (0, 0))
            coverage = min(1.0, q / 20.0) if q else 0.0
            lines.append(f"P {window} {leaf} {q:.3f} {entities[leaf]:.1f} {u:.3f} {coverage:.4f}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(binary, trace, plan, profile):
    args = [str(binary), "partition", "160", "32", "0", str(trace), str(plan), "0",
            "1" if profile else "0"]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(result.stderr)
    out = {}
    for token in result.stdout.split():
        key, value = token.split("=", 1)
        if key == "leaf_profile":
            leaves = []
            for row in value.split("|"):
                name, rest = row.split("=")
                q, b, l, s, n = (float(v) for v in rest.split(","))
                leaves.append({"leaf": int(name[4:]), "q": q, "base": b, "local": l,
                               "switch": s, "entities": n})
            out["leaves"] = leaves
            continue
        if "/" in value:
            value = value.split("/")[0]
        out[key] = float(value)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    initial, events, planned = build()
    trace = args.output / "controlled-combined.m3bin"
    write_trace(trace, WORLD, initial, events)
    configs = {
        "all_scan": (0, 0, 0, 0), "all_fine": (1, 1, 1, 1), "all_quad": (2, 2, 2, 2),
        "query_fine": (1, 0, 0, 0), "query_quad": (2, 0, 0, 0),
        "update_fine": (0, 1, 0, 0), "update_quad": (0, 2, 0, 0),
        "query_quad_update_scan": (2, 0, 0, 0),
        "query_quad_update_fine": (2, 1, 0, 0),
    }
    rows = []
    for name, modes in configs.items():
        plan = args.output / f"combined.{name}.plan"
        write_plan(plan, modes, planned)
        out = run(args.binary, trace, plan, True)
        rows.append({"config": name, "modes": "|".join(str(m) for m in modes),
                     "online_ms": out["online_ms"], "query_ms": out["query_ms"],
                     "update_ms": out["update_ms"], "switch_ms": out["switch_ms"],
                     "leaves": "|".join(f"L{l['leaf']}:q{l['q']:.2f}/b{l['base']:.2f}/"
                                        f"m{l['local']:.2f}/s{l['switch']:.2f}"
                                        for l in out["leaves"])})
        print(f"{name}: {out['online_ms']:.2f} ms "
              f"(q {out['query_ms']:.2f}, u {out['update_ms']:.2f})", flush=True)
    best = None
    for modes in itertools.product(range(3), repeat=4):
        plan = args.output / "combined.enum.plan"
        write_plan(plan, modes, planned)
        out = run(args.binary, trace, plan, False)
        if best is None or out["online_ms"] < best[1]:
            best = (modes, out["online_ms"])
    print("best static config", best, flush=True)
    with (args.output / "controlled_configs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import json
    (args.output / "controlled_summary.json").write_text(json.dumps(
        {"best_static_modes": list(best[0]), "best_static_ms": best[1],
         "configs": rows}, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
