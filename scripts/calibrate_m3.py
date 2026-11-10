"""Calibrate the M3 index parameters, the cost model and the policy hyper-parameters.

Only the two training traces are used for the index parameter sweeps, the cost
model fit and the switch model. The two validation traces select the threshold
policy and the greedy margin. The five test traces are never read here.
"""
import argparse
import csv
import json
import math
import random
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from prepare import json_file, sha

MODES = ("SCAN", "FINE_GRID", "LOCAL_QUAD")
ORDERED_PAIRS = [(a, b) for a in MODES for b in MODES]
QUERY_BASIS = ("Q", "QN", "NC", "QROOTN", "C")
UPDATE_BASIS = ("U", "UROOTN")


def invoke(binary, args):
    result = subprocess.run([str(binary), *map(str, args)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{binary} {args} failed: {result.stderr.strip()}")
    return result.stdout


def parse_kv(text):
    row = {}
    for token in text.strip().split():
        if "=" not in token:
            raise RuntimeError(f"bad benchmark token {token!r}")
        key, value = token.split("=", 1)
        try:
            row[key] = float(value) if any(c in value for c in ".eE") else int(value)
        except ValueError:
            row[key] = value
    return row


def measure(binary, cpu, trace, method, fine, quad, param, config, repeats=1):
    runs = []
    for _ in range(repeats):
        runs.append(parse_kv(invoke(binary, ["measure", method, fine, quad, param, cpu, trace,
                                             config or "NONE"])))
    keys = [k for k in runs[0] if all(isinstance(r[k], (int, float)) for r in runs)]
    return {k: statistics.median(r[k] for r in runs) for k in keys}, runs


def profile(binary, cpu, trace, method, fine, quad, param, config, prefix):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    head = parse_kv(invoke(binary, ["profile", method, fine, quad, param, cpu, trace,
                                    config or "NONE", prefix]))
    rows = list(csv.DictReader((prefix.parent / f"{prefix.name}_regions.csv").open(encoding="utf-8")))
    return head, rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def config_text(model, threshold, margin, alpha):
    lines = ["version 1", f"margin {margin:.6f}", f"alpha {alpha:.6f}",
             "threshold " + " ".join(f"{v:.6f}" for v in threshold)]
    for i, mode in enumerate(MODES):
        lines.append(f"query {mode} " + " ".join(f"{v:.6e}" for v in model["query"][i]))
    for i, mode in enumerate(MODES):
        lines.append(f"update {mode} " + " ".join(f"{v:.6e}" for v in model["update"][i]))
    for (a, b) in ORDERED_PAIRS:
        row = [0.0, 0.0] if a == b else model["switch"][a][b]
        lines.append(f"switch {a} {b} {row[0]:.6f} {row[1]:.9f}")
    return "\n".join(lines) + "\n"


def write_config(path, model, threshold, margin, alpha):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config_text(model, threshold, margin, alpha), encoding="utf-8")
    return path


def fit_nnls(A, b, ridge=1e-6):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    scale = np.linalg.norm(A, axis=0)
    scale[scale == 0] = 1.0
    An = A / scale
    k = An.shape[1]
    augmented = np.vstack([An, math.sqrt(ridge) * np.eye(k)])
    target = np.concatenate([b, np.zeros(k)])
    result = lsq_linear(augmented, target, bounds=(0, np.inf), lsq_solver="exact")
    return result.x / scale


def r2(predicted, observed):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    ss_res = float(np.sum((observed - predicted) ** 2))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


def query_features(row):
    q, n, c = float(row["query_count"]), float(row["entities"]), float(row["coverage"])
    return [q, q * n, n * c, q * math.sqrt(n), c]


def update_features(row):
    u, n = float(row["update_count"]), float(row["entities"])
    return [u, u * math.sqrt(n)]


def fit_row(mode, target, basis, rows, holdout_seed, coef, prediction, observed):
    row = {"mode": mode, "component": target, "rows": rows, "holdout_seed": holdout_seed,
           "basis": "+".join(basis), "r2": r2(prediction, observed),
           "rmse_ns": float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(observed)) ** 2)))}
    for index in range(5):
        row[f"coef_{index}"] = float(coef[index]) if index < len(coef) else ""
    return row


def fit_cost_model(rows_by_mode, train_seeds, profile_seeds):
    """Fit per-mode query and update cost; report in-sample and cross-seed error."""
    model = {"query": [[0.0] * 5 for _ in MODES], "update": [[0.0] * 2 for _ in MODES],
             "switch": {a: {b: [0.0, 0.0] for b in MODES} for a in MODES}}
    report = []
    for i, mode in enumerate(MODES):
        rows = rows_by_mode[mode]
        for target, feature_fn, basis, key in (
                ("query", query_features, QUERY_BASIS, "query_ms"),
                ("update", update_features, UPDATE_BASIS, "local_update_ms")):
            train = [r for r in rows if int(r["seed"]) in train_seeds]
            A = [feature_fn(r) for r in train]
            b = [float(r[key]) * 1e6 for r in train]
            coef = fit_nnls(A, b)
            model[target][i] = list(coef)
            prediction = np.asarray(A) @ coef
            report.append(fit_row(mode, target, basis, len(train), "", coef, prediction, b))
            for seed in profile_seeds:
                hold = [r for r in rows if int(r["seed"]) == seed]
                if not hold:
                    continue
                Ah = [feature_fn(r) for r in hold]
                bh = [float(r[key]) * 1e6 for r in hold]
                ph = np.asarray(Ah) @ coef
                report.append(fit_row(mode, target, basis, len(hold), seed, coef, ph, bh))
    return model, report


def fit_switch_model(probe_rows):
    model = {a: {b: [0.0, 0.0] for b in MODES} for a in MODES}
    fit_rows = []
    for a in MODES:
        for b in MODES:
            if a == b:
                continue
            sample = [r for r in probe_rows if r["from_mode"] == a and r["to_mode"] == b]
            if len(sample) < 3:
                raise RuntimeError(f"switch probe has too few {a}->{b} samples")
            A = np.asarray([[1.0, float(r["entities"])] for r in sample])
            targets = np.asarray([float(r["switch_ns"]) for r in sample])
            coef = fit_nnls(A, targets, ridge=1e-8)
            model[a][b] = list(coef)
            prediction = A @ coef
            fit_rows.append({"from_mode": a, "to_mode": b, "samples": len(sample),
                             "intercept_ns": float(coef[0]), "per_entity_ns": float(coef[1]),
                             "r2": r2(prediction, targets),
                             "median_abs_error_ns": float(np.median(np.abs(prediction - targets)))})
    return model, fit_rows


def bin_fine_grid(fine_rows, scan_rows, quad_rows, test_seeds):
    """Cost bins that show where FINE_GRID beats SCAN and LOCAL_QUAD."""
    key = lambda r: (int(r["window_index"]), int(r["region_id"]), int(r["seed"]))
    scan = {key(r): r for r in scan_rows}
    quad = {key(r): r for r in quad_rows}
    bins = defaultdict(lambda: defaultdict(list))
    for row in fine_rows:
        if int(row["seed"]) in test_seeds:
            continue
        k = key(row)
        if k not in scan or k not in quad:
            continue
        n, q = int(row["entities"]), int(row["query_count"])
        coverage = float(row["coverage"]) / q if q else 0.0
        density = "0-16" if n <= 16 else "17-64" if n <= 64 else "65-256" if n <= 256 else "257+"
        footprint = "0" if q == 0 else "<=0.05" if coverage <= .05 else "<=0.2" if coverage <= .2 else "<=0.5" if coverage <= .5 else ">0.5"
        ratio = "0" if q == 0 else "<=0.25" if q <= .25 * int(row["update_count"]) else "<=1" if q <= int(row["update_count"]) else ">1"
        costs = {
            "FINE_GRID": float(row["query_ms"]) + float(row["local_update_ms"]),
            "SCAN": float(scan[k]["query_ms"]) + float(scan[k]["local_update_ms"]),
            "LOCAL_QUAD": float(quad[k]["query_ms"]) + float(quad[k]["local_update_ms"]),
        }
        bins[(density, footprint, ratio)][row["seed"]].append(costs)
    rows = []
    for (density, footprint, ratio), per_seed in sorted(bins.items()):
        merged = defaultdict(list)
        for costs_list in per_seed.values():
            for costs in costs_list:
                for mode, value in costs.items():
                    merged[mode].append(value)
        if not merged["SCAN"]:
            continue
        wins = {mode: sum(1 for i in range(len(merged["SCAN"]))
                          if merged[mode][i] == min(merged[m][i] for m in merged)) for mode in merged}
        rows.append({
            "region_density": density, "query_footprint": footprint, "query_update_ratio": ratio,
            "paired_region_windows": len(merged["SCAN"]),
            **{f"{mode}_median_ms": statistics.median(values) for mode, values in merged.items()},
            **{f"{mode}_wins": wins.get(mode, 0) for mode in MODES},
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("binary", type=Path)
    parser.add_argument("traces", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--reuse", action="store_true",
                        help="reuse index_parameters.json from a previous run")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "tmp").mkdir(exist_ok=True)
    traces = {}
    for split in ("train", "validation", "test"):
        for seed in config[f"{split}_seeds"]:
            base = args.traces / f"{split}_{seed}"
            meta = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
            if sha(base.with_suffix(".m3bin")) != meta["sha256"]:
                raise RuntimeError(f"trace hash mismatch for {base}")
            traces[split, seed] = base.with_suffix(".m3bin")
    train = [(s, traces["train", s]) for s in config["train_seeds"]]
    validation = [(s, traces["validation", s]) for s in config["validation_seeds"]]
    print(f"train {[s for s,_ in train]} validation {[s for s,_ in validation]}", flush=True)

    # ---- 1. index parameters on the training traces -------------------------
    sweeps = {
        "S1_ALL_FINE": ("fine_cell_m", config["fine_candidates_m"]),
        "S2_ALL_QUAD": ("quad_capacity", config["quad_candidates"]),
        "UNIFORM_GRID": ("grid_side_m", config["global_grid_candidates_m"]),
        "PR_QUADTREE": ("quad_capacity", config["global_tree_candidates"]),
        "RSTAR_BOOST": ("node_capacity", config["global_tree_candidates"]),
    }
    sweep_rows = []
    best = {}
    rng = random.Random(0x3C3F)
    previous = args.output / "index_parameters.json"
    if args.reuse and previous.exists():
        stored = json.loads(previous.read_text(encoding="utf-8"))
        best = {"S1_ALL_FINE": stored["fine_cell_m"], "S2_ALL_QUAD": stored["quad_capacity"],
                "UNIFORM_GRID": stored["UNIFORM_GRID"], "PR_QUADTREE": stored["PR_QUADTREE"],
                "RSTAR_BOOST": stored["RSTAR_BOOST"]}
        print("reusing index parameters", best, flush=True)
        sweeps = {}
    for method, (name, candidates) in sweeps.items():
        trials = list(candidates)
        times = defaultdict(list)
        for repeat in range(args.repeats):
            rng.shuffle(trials)
            for candidate in trials:
                fine = candidate if method == "S1_ALL_FINE" else config["fine_candidates_m"][0]
                quad = candidate if method in ("S2_ALL_QUAD", "PR_QUADTREE", "RSTAR_BOOST") else config["quad_candidates"][0]
                param = candidate if method in ("UNIFORM_GRID", "PR_QUADTREE", "RSTAR_BOOST") else 1
                for seed, path in train:
                    head, _ = measure(args.binary, args.cpu, path, method, fine, quad, param, None)
                    times[candidate].append(float(head["online_ms"]))
                    sweep_rows.append({"sweep": name, "method": method, "seed": seed,
                                       "repeat": repeat, "candidate": candidate,
                                       **{k: head[k] for k in ("build_ms", "online_ms", "query_ms",
                                                               "update_ms", "p95_us", "p99_us")}})
        best[method] = min(trials, key=lambda c: statistics.median(times[c]))
        print(f"{method} -> {best[method]} "
              f"({statistics.median(times[best[method]]):.1f} ms)", flush=True)
    if sweep_rows:
        write_csv(args.output / "index_parameter_sweep.csv", sweep_rows)

    fine_cell = best["S1_ALL_FINE"]
    quad_capacity = best["S2_ALL_QUAD"]
    classic = {"UNIFORM_GRID": best["UNIFORM_GRID"], "PR_QUADTREE": best["PR_QUADTREE"],
               "RSTAR_BOOST": best["RSTAR_BOOST"]}
    json_file(args.output / "index_parameters.json", {
        "fine_cell_m": fine_cell, "quad_capacity": quad_capacity, **classic,
        "selection": "median online ms over the training traces, "
                     f"{args.repeats} repeats, candidate sets from configs/m3.json"})

    # ---- 2. fixed-mode profiles for the cost model and the FINE_GRID bins ---
    fixed = {"SCAN": "S0_ALL_SCAN", "FINE_GRID": "S1_ALL_FINE", "LOCAL_QUAD": "S2_ALL_QUAD"}
    rows_by_mode = defaultdict(list)
    for split, group in (("train", train), ("validation", validation)):
        for mode, method in fixed.items():
            for seed, path in group:
                param = quad_capacity if mode == "LOCAL_QUAD" else 1
                head, rows = profile(args.binary, args.cpu, path, method, fine_cell,
                                     quad_capacity, param, None,
                                     args.output / "tmp" / f"profile_{split}_{seed}_{mode}")
                assert int(head["query_count"]) == sum(config["queries_per_step"]) * config["phase_steps"]
                if mode in ("SCAN", "FINE_GRID", "LOCAL_QUAD") and method.startswith("S"):
                    assert int(head["decision_calls"]) == 0
                for row in rows:
                    row["seed"] = str(seed)
                    row["split"] = split
                    rows_by_mode[mode].append(row)
                print(f"profile {split} {seed} {mode} online {head['online_ms']:.1f} ms", flush=True)

    # ---- 3. cost model on training traces only ------------------------------
    model, fit_report = fit_cost_model(rows_by_mode, set(config["train_seeds"]),
                                       config["train_seeds"])
    write_csv(args.output / "cost_model_fit.csv", fit_report)

    # ---- 4. switch model from the probe on training traces ------------------
    probe_rows = []
    for seed, path in train:
        text = invoke(args.binary, ["switchprobe", fine_cell, quad_capacity, args.cpu, path])
        lines = text.strip().splitlines()
        if lines[0] != "from,to,entities,seq,switch_ns":
            raise RuntimeError("unexpected switch probe header")
        for line in lines[1:]:
            from_mode, to_mode, entities, seq, switch_ns = line.split(",")
            if from_mode == to_mode:
                continue
            probe_rows.append({"seed": seed, "from_mode": from_mode, "to_mode": to_mode,
                               "entities": int(entities), "seq": int(seq),
                               "switch_ns": int(switch_ns)})
    write_csv(args.output / "switch_probe.csv", probe_rows)
    switch_model, switch_fit = fit_switch_model(probe_rows)
    model["switch"] = switch_model
    write_csv(args.output / "switch_model_fit.csv", switch_fit)

    # ---- 5. FINE_GRID advantage bins on train + validation ------------------
    bins = bin_fine_grid(rows_by_mode["FINE_GRID"], rows_by_mode["SCAN"], rows_by_mode["LOCAL_QUAD"],
                         set(config["test_seeds"]))
    write_csv(args.output / "fine_grid_bins.csv", bins)

    # ---- 6. policy hyper-parameters on the validation traces ----------------
    threshold_candidates = [tuple(row) for row in config["threshold_candidates"]]
    margin_candidates = list(config["greedy_margins"])
    alpha_candidates = list(config["alpha_candidates"])
    assert all(len(row) == 7 for row in threshold_candidates)

    def evaluate(method, thresholds, margin, alpha):
        path = write_config(args.output / "tmp" / "selector.txt", model, thresholds, margin, alpha)
        values = []
        for repeat in range(args.repeats):
            for seed, trace in validation:
                head, _ = measure(args.binary, args.cpu, trace, method, fine_cell,
                                  quad_capacity, 1, str(path))
                assert head["decision_calls"] == config["steps"] // config["window_steps"] - 1
                values.append(float(head["online_ms"]))
        return statistics.median(values), values

    threshold_rows, threshold_best = [], None
    for alpha in alpha_candidates:
        for candidate in threshold_candidates:
            median, values = evaluate("S3_THRESHOLD", candidate, 0.1, alpha)
            threshold_rows.append({"alpha": alpha, "candidate": list(candidate),
                                   "median_online_ms": median,
                                   "runs": ";".join(f"{v:.6f}" for v in values)})
            if threshold_best is None or median < threshold_best[0]:
                threshold_best = (median, candidate, alpha)
            print(f"threshold alpha={alpha} {candidate} -> {median:.1f} ms", flush=True)
    write_csv(args.output / "threshold_search.csv", threshold_rows)

    margin_rows, margin_best = [], None
    for margin in margin_candidates:
        median, values = evaluate("S4_GREEDY", threshold_best[1], margin, threshold_best[2])
        margin_rows.append({"margin": margin, "median_online_ms": median,
                            "runs": ";".join(f"{v:.6f}" for v in values)})
        if margin_best is None or median < margin_best[0]:
            margin_best = (median, margin)
        print(f"margin {margin} -> {median:.1f} ms", flush=True)
    write_csv(args.output / "margin_search.csv", margin_rows)

    # ---- 7. final selector configuration ------------------------------------
    selected = {"fine_cell_m": fine_cell, "quad_capacity": quad_capacity,
                "threshold": list(threshold_best[1]), "alpha": threshold_best[2],
                "margin": margin_best[1],
                "threshold_validation_online_ms": threshold_best[0],
                "greedy_validation_online_ms": margin_best[0],
                "classic_parameters": classic}
    json_file(args.output / "selected.json", selected)
    final = write_config(args.output / "selector_final.txt", model, threshold_best[1],
                         margin_best[1], threshold_best[2])
    json_file(args.output / "cost_model.json", model)
    print(f"wrote {final} and selected.json", flush=True)
    print("FINE_GRID bin report:")
    for row in bins:
        print(" ", row["region_density"], row["query_footprint"], row["query_update_ratio"],
              row["paired_region_windows"], "F", row["FINE_GRID_wins"],
              "S", row["SCAN_wins"], "Q", row["LOCAL_QUAD_wins"], flush=True)


if __name__ == "__main__":
    main()
