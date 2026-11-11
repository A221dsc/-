"""Build public-map scenarios, the nominal future-load derivation and the actual replay."""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from m4_scenario import (SIDE, actual_events, build_scenario, nominal_summary, prepare,
                         planned_query_schedule, sha, simulate, write_nominal_table,
                         write_trace)
from prepare import json_file


def seeds_for(config, split, index):
    scenario_seed = config["scenario_seeds"][split][index]
    actual = config["actual_seeds"][split][index]
    return {"scenario": scenario_seed, "objective": scenario_seed + 7, "route": scenario_seed + 13,
            "query": scenario_seed + 19, "actual": actual}


def make_actual(config, profile, seeds):
    perturbation = dict(config["mismatch_profiles"][profile])
    perturbation["phase_shift"] = list(perturbation["phase_shift"])
    perturbation.setdefault("exit_share", 0.0)
    perturbation.setdefault("exit_step", -1)
    streams = [np.random.default_rng(s) for s in seeds["actual"]]
    focus = np.random.default_rng([seeds["actual"][4], 991]).normal(
        0.0, config["mismatch_profiles"][profile]["focus_offset_m"], (len(config["phases"]), 2))
    return {"detour_rng": streams[0], "speed_rng": streams[1], "delay_rng": streams[2],
            "noise_rng": streams[3], "query_rng": streams[4], "phase_shift": perturbation["phase_shift"],
            "focus_offset": focus}, perturbation


def build_one(config, map_name, split, index, directory, mismatch="medium", overrides=None):
    directory = Path(directory)
    seed = seeds_for(config, split, index)
    cfg = dict(config)
    if overrides:
        cfg = dict(cfg)
        for key, value in overrides.items():
            cfg[key] = value
    game = build_game(map_name)
    _, scenario, entities = build_scenario(cfg, map_name, split, index, seed)
    scenario["perturbation"] = dict(config["mismatch_profiles"][mismatch])
    scenario["mismatch_profile"] = mismatch
    prepared_nominal = prepare(scenario, game, entities)
    nominal = nominal_summary(scenario, game, prepared_nominal)
    actual, perturbation = make_actual(config, mismatch, seed)
    scenario["perturbation"] = perturbation
    prepared_actual = prepare(scenario, game, entities, actual=actual)
    execution = actual_events(scenario, game, prepared_actual, actual)

    ident = f"{split}-{index}-{Path(map_name).stem}"
    scenarios_dir = directory / "scenarios"
    traces_dir = directory / "traces"
    loads_dir = directory / "loads_python_semantics"
    meta_dir = directory / "meta"
    for folder in (scenarios_dir, traces_dir, loads_dir, meta_dir):
        folder.mkdir(parents=True, exist_ok=True)
    json_file(scenarios_dir / f"{ident}.json", scenario)
    with (scenarios_dir / f"{ident}.entities.csv").open("w", encoding="utf-8", newline="") as handle:
        handle.write("entity_id,category,start_cell_x,start_cell_y,assembly,group,objectives,speed_mps,offset_x,offset_y\n")
        for entity in entities:
            handle.write(f"{entity['entity_id']},{entity['category']},{entity['start_cell'][0]},"
                         f"{entity['start_cell'][1]},{entity['assembly']},{entity['group']},"
                         f"{'|'.join(entity['objectives'])},{entity['speed_mps']:.3f},"
                         f"{entity['offset_cells'][0]:.3f},{entity['offset_cells'][1]:.3f}\n")
    write_nominal_table(scenarios_dir / f"{ident}.nominal.csv", nominal)
    schedule = nominal["schedule"]
    with (scenarios_dir / f"{ident}.nominal_queries.csv").open("w", encoding="utf-8", newline="") as handle:
        handle.write("step,kind,x,y,half_m,focus\n")
        for step, kind, point, half, tag in schedule:
            handle.write(f"{step},{kind},{point[0]:.3f},{point[1]:.3f},{half:.1f},{tag}\n")
    trace_path = traces_dir / f"{ident}.actual.m3bin"
    write_trace(trace_path, scenario["world"], execution["points"], execution["events"])
    loads = execution["actual_loads"]
    with (loads_dir / f"{ident}.actual_loads.csv").open("w", encoding="utf-8", newline="") as handle:
        handle.write("window,region,query_count,entity_count,update_count,coverage\n")
        for w in range(loads["queries"].shape[0]):
            for region in range(SIDE * SIDE):
                handle.write(f"{w},{region},{int(loads['queries'][w, region])},"
                             f"{int(loads['entities'][w, region])},"
                             f"{int(loads['updates'][w, region])},"
                             f"{loads['coverage'][w, region]:.6f}\n")
    meta = {
        "scenario_id": ident, "split": split, "index": index, "map": Path(map_name).stem,
        "mismatch_profile": mismatch, "seeds": seed,
        "entity_count": len(entities), "steps": scenario["steps"],
        "queries": int(sum(1 for event in execution["events"] if event[0] in "RC")),
        "updates": int(sum(1 for event in execution["events"] if event[0] == "U")),
        "phase_at": execution["phase_at"].tolist(),
        "scenario_sha256": sha(scenarios_dir / f"{ident}.json"),
        "entities_sha256": sha(scenarios_dir / f"{ident}.entities.csv"),
        "nominal_sha256": sha(scenarios_dir / f"{ident}.nominal.csv"),
        "nominal_queries_sha256": sha(scenarios_dir / f"{ident}.nominal_queries.csv"),
        "trace_sha256": sha(trace_path),
    }
    json_file(meta_dir / f"{ident}.json", meta)
    return meta


def build_game(map_name):
    from m4_maps import GameMap
    return GameMap(Path("data/maps") / map_name)


def leakage_check(config, directory):
    """Same scenario, different actual disturbance seeds => identical nominal data."""
    map_name = config["maps"]["test"][0]
    original = config["actual_seeds"]["test"][0]
    results = []
    for trial, actual_seeds in enumerate((original, [s + 90000 for s in original])):
        cfg = dict(config)
        cfg["actual_seeds"] = dict(config["actual_seeds"])
        cfg["actual_seeds"]["test"] = [actual_seeds, actual_seeds]
        with tempfile.TemporaryDirectory() as temp:
            meta = build_one(cfg, map_name, "test", 0, temp, "medium")
            nominal = (Path(temp) / "scenarios" / f"test-0-{Path(map_name).stem}.nominal.csv").read_bytes()
            trace = (Path(temp) / "traces" / f"test-0-{Path(map_name).stem}.actual.m3bin").read_bytes()
            results.append({"trial": trial, "actual_seeds": actual_seeds,
                            "nominal_sha256": meta["nominal_sha256"],
                            "trace_sha256": meta["trace_sha256"],
                            "nominal_bytes": len(nominal), "trace_bytes": len(trace)})
    assert results[0]["nominal_sha256"] == results[1]["nominal_sha256"], "nominal load leaked actual disturbance"
    assert results[0]["trace_sha256"] != results[1]["trace_sha256"], "disturbance seeds produced identical traces"
    json_file(Path(directory) / "leakage_check.json",
              {"map": map_name, "trials": results,
               "verdict": "nominal identical, actual traces differ"})
    print("leakage check passed:", results[0]["nominal_sha256"])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--entities", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--leakage", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    overrides = {}
    if args.entities:
        config = json.loads(json.dumps(config))
        config["scenario"]["entity_count"] = args.entities
        overrides["scenario"] = config["scenario"]
    if args.steps:
        config = json.loads(json.dumps(config))
        config["steps"] = args.steps
        config["phase_steps"] = args.steps // 4
        overrides["steps"] = args.steps
        overrides["phase_steps"] = config["phase_steps"]
    if args.leakage:
        leakage_check(config, args.output)
        return
    manifest = []
    for split in args.splits:
        for index, map_name in enumerate(config["maps"][split]):
            meta = build_one(config, map_name, split, index, args.output, "medium", overrides)
            manifest.append(meta)
            print(f"{meta['scenario_id']}: {meta['entity_count']} entities, "
                  f"{meta['queries']} queries, {meta['updates']} updates", flush=True)
    json_file(Path(args.output) / "manifest.json", manifest)


if __name__ == "__main__":
    main()
