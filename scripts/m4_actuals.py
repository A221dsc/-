"""Generate additional actual replays for one scenario with independent disturbance seeds.

The nominal table is shared: the scenario is the same, only the actual execution
differs. This is exactly the leakage contract required by the specification.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from m4_scenario import SIDE, actual_events, build_scenario, prepare, sha, write_trace
from m4_maps import GameMap
from prepare import json_file


def actual_streams(scenario, seeds, perturbation):
    streams = [np.random.default_rng(s) for s in seeds]
    focus = np.random.default_rng([seeds[4], 991]).normal(
        0.0, perturbation["focus_offset_m"] if "focus_offset_m" in perturbation else 0.0,
        (len(scenario["phases"]), 2))
    return {"detour_rng": streams[0], "speed_rng": streams[1], "delay_rng": streams[2],
            "noise_rng": streams[3], "query_rng": streams[4],
            "phase_shift": perturbation["phase_shift"], "focus_offset": focus}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--actuals", type=int, default=3)
    parser.add_argument("--splits", nargs="+", default=["test", "validation"])
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    added = []
    for split in args.splits:
        for index, map_name in enumerate(config["maps"][split]):
            ident = f"{split}-{index}-{Path(map_name).stem}"
            scenario = json.loads((args.directory / "scenarios" / f"{ident}.json").read_text(encoding="utf-8"))
            game = GameMap(Path("data/maps") / map_name)
            rebuilt_game, rebuilt, entities = build_scenario(
                config, map_name, split, index, scenario["seeds"])
            assert rebuilt["scenario_sha256"] == scenario["scenario_sha256"], "scenario rebuild diverged"
            perturbation = scenario["perturbation"]
            canonical = scenario["seeds"]["actual"]
            for k in range(1, args.actuals):
                seeds = [s + 100003 * k for s in canonical]
                actual = actual_streams(scenario, seeds, perturbation)
                prepared = prepare(scenario, game, entities, actual=actual)
                execution = actual_events(scenario, game, prepared, actual)
                suffix = f".actual{k}"
                trace = args.directory / "traces" / f"{ident}{suffix}.m3bin"
                write_trace(trace, scenario["world"], execution["points"], execution["events"])
                loads = execution["actual_loads"]
                (args.directory / "loads_python_semantics").mkdir(parents=True, exist_ok=True)
                with (args.directory / "loads_python_semantics" / f"{ident}{suffix}.loads.csv").open(
                        "w", encoding="utf-8", newline="") as handle:
                    handle.write("window,region,query_count,entity_count,update_count,coverage\n")
                    for w in range(loads["queries"].shape[0]):
                        for region in range(SIDE * SIDE):
                            handle.write(f"{w},{region},{int(loads['queries'][w, region])},"
                                         f"{int(loads['entities'][w, region])},"
                                         f"{int(loads['updates'][w, region])},"
                                         f"{loads['coverage'][w, region]:.6f}\n")
                meta = {"scenario_id": ident, "split": split, "index": index,
                        "map": Path(map_name).stem, "actual_seed_set": seeds,
                        "mismatch_profile": scenario.get("mismatch_profile", "medium"),
                        "queries": int(sum(1 for event in execution["events"] if event[0] in "RC")),
                        "updates": int(sum(1 for event in execution["events"] if event[0] == "U")),
                        "trace_sha256": sha(trace),
                        "nominal_sha256": sha(args.directory / "scenarios" / f"{ident}.nominal.csv")}
                json_file(args.directory / "meta" / f"{ident}{suffix}.json", meta)
                added.append(meta)
                print(f"{ident}{suffix}: {meta['queries']} queries {meta['updates']} updates", flush=True)
    json_file(args.directory / "actuals_manifest.json", added)


if __name__ == "__main__":
    main()
