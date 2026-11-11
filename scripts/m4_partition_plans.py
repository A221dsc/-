"""Generate per-leaf nominal load plans (L + P lines) for all partition variants.

One nominal simulation per scenario is reused across the six partition plans.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from m4_maps import GameMap  # noqa: E402
from m4_partition_loads import leaf_grid, leaf_query_loads, read_plan  # noqa: E402
from m4_scenario import build_scenario, planned_query_schedule, prepare, simulate  # noqa: E402

VARIANTS = ["uniform-8", "uniform-16", "uniform-32", "adaptive-entity", "adaptive-query",
            "adaptive-combined"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("scenarios", type=Path)
    parser.add_argument("plans", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    for index, map_name in enumerate(config["maps"]["test"]):
        ident = f"test-{index}-{Path(map_name).stem}"
        scenario = json.loads((args.scenarios / "scenarios" / f"{ident}.json").read_text(encoding="utf-8"))
        game, _, entities = build_scenario(config, map_name, "test", index, scenario["seeds"])
        prepared = prepare(scenario, game, entities)
        state = simulate(scenario, game, prepared, reporting=True)
        windows = state["window_entities_sub"].shape[0]
        for variant in VARIANTS:
            source = args.plans / f"{ident}.{variant}.plan"
            rects, levels = read_plan(source)
            grid = leaf_grid(rects).reshape(-1)
            leaf_of_sub = np.where(grid >= 0, grid, 0)
            ent = np.zeros((windows, len(rects)))
            upd = np.zeros((windows, len(rects)))
            for w in range(windows):
                ent[w] = np.bincount(leaf_of_sub, weights=state["window_entities_sub"][w],
                                     minlength=len(rects))
                upd[w] = np.bincount(leaf_of_sub, weights=state["window_updates_sub"][w],
                                     minlength=len(rects))
            touches, coverage = leaf_query_loads(scenario, game, rects, windows)
            lines = [f"L {x0} {y0} {x1} {y1} {level}"
                     for (x0, y0, x1, y1), level in zip(rects, levels)]
            for w in range(windows):
                for leaf in range(len(rects)):
                    lines.append(f"P {w} {leaf} {touches[w, leaf]:.6f} {ent[w, leaf]:.3f} "
                                 f"{upd[w, leaf]:.3f} {coverage[w, leaf]:.6f}")
            (args.output / f"{ident}.{variant}.plan").write_text("\n".join(lines) + "\n",
                                                                 encoding="utf-8")
            print(f"{ident} {variant}: {len(rects)} leaves, "
                  f"touches/window {touches.sum(axis=1).mean():.0f}", flush=True)


if __name__ == "__main__":
    main()
