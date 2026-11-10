"""Calibrate the M4 selector on the public-map scenario load.

Index parameters and the cost model use the training scenarios only. The policy
hyper-parameters use the validation scenarios only. Thresholds and the greedy
margin are tuned once and reused unchanged by both the history and the scenario
variants, so the scenario prior is the only difference between them.
"""
import argparse
import json
import random
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from calibrate_m3 import fit_cost_model, fit_switch_model, write_config
from prepare import json_file, sha

METHOD_FIXED = {"SCAN": "Fixed Regional Scan", "FINE_GRID": "Fixed Local Fine Grid",
                "LOCAL_QUAD": "Fixed Local Quadtree"}
METHOD_POLICY = {"history": "", "scenario": " (Scenario)"}
POLICY_NAMES = {"threshold": "Threshold-Based Adaptive Regional Index Selection",
                "greedy": "Cost-Aware Greedy Regional Index Selection"}


def invoke(binary, args):
    result = subprocess.run([str(binary), *map(str, args)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{binary} {args} failed: {result.stderr.strip()}")
    return result.stdout


def parse_kv(text):
    row = {}
    for token in text.strip().split():
        if "=" not in token:
            raise RuntimeError(f"bad token {token}")
        key, value = token.split("=", 1)
        row[key] = float(value) if any(c in value for c in ".eE") else int(value)
    return row


def write_csv(path, rows):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select(rows, binary, cpu, method, fine, quad, param, config, repeats):
    values = []
    for _ in range(repeats):
        head = parse_kv(invoke(binary, ["measure", method, fine, quad, param, cpu, rows,
                                        config or "NONE"]))
        values.append(float(head["online_ms"]))
    return statistics.median(values), values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("binary", type=Path)
    parser.add_argument("scenarios", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "tmp").mkdir(exist_ok=True)

    def trace(split, scenario, actual=0):
        ident = f"{split}-{scenario}-{Path(config['maps'][split][scenario]).stem}"
        return args.scenarios / "traces" / (f"{ident}.actual.m3bin" if actual == 0
                                            else f"{ident}.actual{actual}.m3bin")

    train = [trace("train", i) for i in range(len(config["maps"]["train"]))]
    validation = [trace("validation", i) for i in range(len(config["maps"]["validation"]))]
    validation_actual = [trace("validation", i, 1) for i in range(len(config["maps"]["validation"]))]

    sweeps = {
        "Fixed Local Fine Grid": ("fine_cell_m", [80, 160, 320, 640]),
        "Fixed Local Quadtree": ("quad_capacity", [8, 16, 32, 64]),
        "Uniform Grid": ("grid_side_m", [125, 250, 500, 1000]),
        "PR Quadtree": ("quad_capacity", [8, 16, 32, 64]),
        "Boost R*-Tree": ("node_capacity", [8, 16, 32, 64]),
    }
    rng = random.Random(0x4D4)
    sweep_rows, best = [], {}
    for method, (label, candidates) in sweeps.items():
        trials = list(candidates)
        times = defaultdict(list)
        for repeat in range(args.repeats):
            rng.shuffle(trials)
            for candidate in trials:
                fine = candidate if method == "Fixed Local Fine Grid" else 320
                quad = candidate if method in ("Fixed Local Quadtree", "PR Quadtree",
                                               "Boost R*-Tree") else 64
                param = candidate if method in ("Uniform Grid", "PR Quadtree", "Boost R*-Tree") else 1
                for path in train:
                    head = parse_kv(invoke(args.binary, ["measure", method, fine, quad, param,
                                                         args.cpu, path, "NONE"]))
                    times[candidate].append(float(head["online_ms"]))
                    sweep_rows.append({"sweep": label, "method": method, "candidate": candidate,
                                       "database": path.name, "repeat": repeat,
                                       "online_ms": head["online_ms"], "query_ms": head["query_ms"],
                                       "update_ms": head["update_ms"]})
        best[method] = min(trials, key=lambda c: statistics.median(times[c]))
        print(f"{method} -> {best[method]} ({statistics.median(times[best[method]]):.1f} ms)", flush=True)
    write_csv(args.output / "index_parameter_sweep.csv", sweep_rows)

    fine = best["Fixed Local Fine Grid"]
    quad = best["Fixed Local Quadtree"]
    classic = {"Uniform Grid": best["Uniform Grid"], "PR Quadtree": best["PR Quadtree"],
               "Boost R*-Tree": best["Boost R*-Tree"]}
    json_file(args.output / "index_parameters.json",
              {"fine_cell_m": fine, "quad_capacity": quad, **classic})

    rows_by_mode = defaultdict(list)
    for mode, method in METHOD_FIXED.items():
        param = quad if mode == "LOCAL_QUAD" else 1
        for seed, path in enumerate(train):
            head = parse_kv(invoke(args.binary, ["profile", method, fine, quad, param, args.cpu,
                                                 path, "NONE",
                                                 args.output / "tmp" / f"profile_{seed}_{mode}"]))
            assert int(head["decision_calls"]) == 0
            prefix = args.output / "tmp" / f"profile_{seed}_{mode}"
            import csv
            with (prefix.parent / f"{prefix.name}_regions.csv").open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    row["seed"] = str(seed)
                    rows_by_mode[mode].append(row)
            print(f"profile train {seed} {mode} online {head['online_ms']:.1f} ms", flush=True)
    model, fit_report = fit_cost_model(rows_by_mode, {0, 1}, [0, 1])
    write_csv(args.output / "cost_model_fit.csv", fit_report)

    probe_rows = []
    for seed, path in enumerate(train):
        text = invoke(args.binary, ["switchprobe", fine, quad, args.cpu, path])
        lines = text.strip().splitlines()
        for line in lines[1:]:
            from_mode, to_mode, entities, seq, switch_ns = line.split(",")
            probe_rows.append({"seed": seed, "from_mode": from_mode, "to_mode": to_mode,
                               "entities": int(entities), "seq": int(seq),
                               "switch_ns": int(switch_ns)})
    write_csv(args.output / "switch_probe.csv", probe_rows)
    switch_model, switch_fit = fit_switch_model(probe_rows)
    model["switch"] = switch_model
    write_csv(args.output / "switch_model_fit.csv", switch_fit)

    thresholds = [tuple(row) for row in config["threshold_candidates"]]
    margins = list(config["greedy_margins"])
    alphas = list(config.get("alpha_candidates", [1.0]))

    def evaluate(policy, threshold, margin, alpha):
        path = write_config(args.output / "tmp" / "selector.txt", model, threshold, margin, alpha)
        values = []
        for repeat in range(args.repeats):
            for trace_path in validation + validation_actual:
                head = parse_kv(invoke(args.binary, ["measure", POLICY_NAMES[policy],
                                                     fine, quad, 1, args.cpu, trace_path,
                                                     str(path)]))
                values.append(float(head["online_ms"]))
        return statistics.median(values), values

    threshold_rows, threshold_best = [], None
    for alpha in alphas:
        for candidate in thresholds:
            median, values = evaluate("threshold", candidate, 0.1, alpha)
            threshold_rows.append({"alpha": alpha, "candidate": list(candidate),
                                   "median_online_ms": median,
                                   "runs": ";".join(f"{v:.4f}" for v in values)})
            if threshold_best is None or median < threshold_best[0]:
                threshold_best = (median, candidate, alpha)
            print(f"threshold alpha={alpha} {candidate} -> {median:.1f} ms", flush=True)
    write_csv(args.output / "threshold_search.csv", threshold_rows)

    margin_rows, margin_best = [], None
    for margin in margins:
        median, values = evaluate("greedy", threshold_best[1], margin, threshold_best[2])
        margin_rows.append({"margin": margin, "median_online_ms": median,
                            "runs": ";".join(f"{v:.4f}" for v in values)})
        if margin_best is None or median < margin_best[0]:
            margin_best = (median, margin)
        print(f"margin {margin} -> {median:.1f} ms", flush=True)
    write_csv(args.output / "margin_search.csv", margin_rows)

    selected = {"fine_cell_m": fine, "quad_capacity": quad, "classic": classic,
                "threshold": list(threshold_best[1]), "alpha": threshold_best[2],
                "margin": margin_best[1],
                "threshold_validation_online_ms": threshold_best[0],
                "greedy_validation_online_ms": margin_best[0]}
    json_file(args.output / "selected.json", selected)
    final = write_config(args.output / "selector_final.txt", model, threshold_best[1],
                         margin_best[1], threshold_best[2])
    json_file(args.output / "cost_model.json", model)
    print(f"wrote {final}")


if __name__ == "__main__":
    main()
