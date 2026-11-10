"""Run the M3 fair comparison on five paired test seeds.

One process per (method, seed) for the primary timing, one for the exhaustive
oracle validation and one profile replay per regional method. The profile replay
supplies the decision log and the per-region counterfactual costs; it never
contributes to the headline timings.
"""
import argparse
import csv
import json
import platform
import random
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

import psutil
import pyarrow

from prepare import json_file, sha

METHODS = ["S0_ALL_SCAN", "S1_ALL_FINE", "S2_ALL_QUAD", "S3_THRESHOLD", "S4_GREEDY",
           "UNIFORM_GRID", "PR_QUADTREE", "RSTAR_BOOST"]
FIXED = {"S0_ALL_SCAN": "S0_ALL_SCAN", "S1_ALL_FINE": "S1_ALL_FINE", "S2_ALL_QUAD": "S2_ALL_QUAD"}
POLICIES = {"S3_THRESHOLD", "S4_GREEDY"}
REGIONAL = set(FIXED) | POLICIES
CLASSIC = {"UNIFORM_GRID", "PR_QUADTREE", "RSTAR_BOOST"}
METRIC = ["build_ms", "online_ms", "query_ms", "update_ms", "switch_ms", "policy_ms",
          "p50_us", "p95_us", "p99_us", "peak_rss_mb", "prebuild_rss_mb", "trace_load_ms"]


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


def parse_validation(text):
    fields = dict(part.split("=") for part in text.strip().split())
    return {key: int(value) for key, value in fields.items()}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def bootstrap_interval(values, rng, draws=10000):
    values = list(values)
    medians = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(draws))
    return [medians[int(0.025 * draws)], medians[int(0.975 * draws)]]


def summarize(rows, output):
    by_method = defaultdict(dict)
    for row in rows:
        by_method[row["method"]][int(row["seed"])] = row
    seeds = sorted(by_method[METHODS[0]])
    medians = {method: {field: statistics.median(float(by_method[method][s][field]) for s in seeds)
                        for field in METRIC if all(by_method[method][s][field] != "" for s in seeds)}
               for method in by_method}
    reference = min(medians, key=lambda m: medians[m]["online_ms"])
    static_reference = min(FIXED, key=lambda m: medians[m]["online_ms"])
    rng = random.Random(0x5EE1)
    intervals = {}
    for method in by_method:
        entry = {"median_online_ms": medians[method]["online_ms"]}
        for label, base in (("best_method", reference), ("best_static_regional", static_reference),
                            ("all_scan", "S0_ALL_SCAN")):
            ratio = {s: float(by_method[method][s]["online_ms"]) / float(by_method[base][s]["online_ms"])
                     for s in seeds}
            entry[f"median_ratio_to_{label}"] = statistics.median(ratio.values())
            entry[f"bootstrap95_ratio_to_{label}"] = bootstrap_interval(ratio.values(), rng)
        intervals[method] = entry
    json_file(output, {"seeds": seeds, "reference_best_method": reference,
                       "reference_best_static_regional": static_reference,
                       "medians": medians, "paired_intervals": intervals})
    return reference, static_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("selection", type=Path, help="results/m3/calibration/selected.json")
    parser.add_argument("selector", type=Path, help="results/m3/calibration/selector_final.txt")
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--traces", type=Path, default=Path("data/processed/m3_traces"))
    parser.add_argument("--cpu", type=int, default=0)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    fine = int(selection["fine_cell_m"])
    quad = int(selection["quad_capacity"])
    classic = selection["classic_parameters"]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "profile").mkdir(exist_ok=True)

    def param_for(method):
        return classic[method] if method in CLASSIC else 1

    def selector_for(method):
        return str(args.selector) if method in POLICIES else "NONE"

    traces, manifest, schedules, expected = {}, [], {}, {}
    for seed in config["test_seeds"]:
        base = args.traces / f"test_{seed}"
        meta = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
        if sha(base.with_suffix(".m3bin")) != meta["sha256"]:
            raise RuntimeError(f"trace hash mismatch for {base}")
        traces[seed] = base.with_suffix(".m3bin")
        expected[seed] = meta
        manifest.append({"seed": seed, "sha256": meta["sha256"], "split": meta["split"],
                         "steps": meta["steps"], "entities": meta["entities"],
                         "queries": meta["queries"], "updates": meta["updates"],
                         "boundaries": meta["boundaries"]})
        schedules[str(seed)] = meta["hotspot_schedule"]
    json_file(args.output / "trace_manifest.json", manifest)
    json_file(args.output / "trace_schedule.json", schedules)
    json_file(args.output / "config.json", config)
    json_file(args.output / "selection.json", selection)
    (args.output / "selector.txt").write_bytes(args.selector.read_bytes())

    def version(*command):
        return subprocess.check_output(command, text=True).splitlines()[0]

    json_file(args.output / "environment.json", {
        "system": platform.platform(), "processor": platform.processor(),
        "logical_cores": psutil.cpu_count(), "python": platform.python_version(),
        "pyarrow": pyarrow.__version__, "compiler": version("g++", "--version"),
        "flags": "-std=c++20 -O3 -DNDEBUG", "cpu": args.cpu, "threads": 1,
        "deps": {"boost": "1.82.0", "libspatialindex": "1.9.3"},
        "git_commit": version("git", "rev-parse", "HEAD"),
        "binary_sha256": sha(args.binary), "config_sha256": sha(args.config),
        "selector_sha256": sha(args.selector), "selection_sha256": sha(args.selection),
        "index": {"fine_cell_m": fine, "quad_capacity": quad, "classic": classic},
        "processes": "one pinned process per method and seed for timing, validation and profiling",
        "files_sha256": {p.as_posix(): sha(p) for d in ("scripts", "src", "tests")
                         for p in sorted(Path(d).rglob("*"))
                         if p.is_file() and "__pycache__" not in p.parts},
    })

    metrics, decisions, regions, timeline = [], [], [], []
    for seed in config["test_seeds"]:
        trace = traces[seed]
        order = list(METHODS)
        random.Random(seed ^ 0xA31F).shuffle(order)
        for method in order:
            param = param_for(method)
            selector = selector_for(method)
            truth = parse_validation(invoke(args.binary, ["validate", method, fine, quad, param,
                                                          args.cpu, trace, selector]))
            assert truth["mismatches"] == 0
            assert truth["queries"] == expected[seed]["queries"]
            assert truth["updates"] == expected[seed]["updates"]
            head = parse_kv(invoke(args.binary, ["measure", method, fine, quad, param,
                                                 args.cpu, trace, selector]))
            assert head["query_count"] == truth["queries"] and head["update_count"] == truth["updates"]
            expected_calls = config["steps"] // config["window_steps"] - 1
            assert head["decision_calls"] == (expected_calls if method in POLICIES else 0)
            row = {"method": method, "seed": seed, "split": "test", "parameter": param,
                   "fine_cell_m": fine, "quad_capacity": quad,
                   "train_seeds": ";".join(map(str, config["train_seeds"])),
                   "validation_seeds": ";".join(map(str, config["validation_seeds"])),
                   "trace_sha256": expected[seed]["sha256"],
                   "selector_sha256": sha(args.selector) if method in POLICIES else "",
                   "correctness_pass": True,
                   **{field: head[field] for field in METRIC},
                   "total_ms": float(head["build_ms"]) + float(head["online_ms"]),
                   "query_count": head["query_count"], "update_count": head["update_count"],
                   "result_count": head["result_count"], "candidate_count": head["candidate_count"],
                   "switches": head["switches"], "decision_calls": head["decision_calls"],
                   "decision_hash": head["decision_hash"],
                   "promotions": head["promotions"], "demotions": head["demotions"],
                   "retunes": head["retunes"], "blocked": head["blocked"],
                   "final_scan": head["final_scan"], "final_fine": head["final_fine"],
                   "final_quad": head["final_quad"],
                   "final_indexed_entities": head["final_indexed_entities"],
                   "trace_bytes": head["trace_bytes"]}
            metrics.append(row)
            if method in REGIONAL:
                prefix = args.output / "profile" / f"test_{seed}_{method}"
                profile = parse_kv(invoke(args.binary, ["profile", method, fine, quad, param,
                                                        args.cpu, trace, selector, prefix]))
                assert profile["query_count"] == head["query_count"]
                assert profile["switches"] == head["switches"]
                if method in POLICIES:
                    assert profile["decision_hash"] == head["decision_hash"], "decision replay diverged"
                for item in read_csv(prefix.parent / f"{prefix.name}_regions.csv"):
                    regions.append({"method": method, "seed": seed, **item})
                for item in read_csv(prefix.parent / f"{prefix.name}_timeline.csv"):
                    timeline.append({"method": method, "seed": seed, **item})
                if method in POLICIES:
                    for item in read_csv(prefix.parent / f"{prefix.name}_decisions.csv"):
                        decisions.append({"method": method, "seed": seed, **item})
            print(f"seed {seed} {method} online {head['online_ms']:.1f} ms", flush=True)
        write_csv(args.output / "metrics.csv", metrics)

    write_csv(args.output / "regions.csv", regions)
    write_csv(args.output / "timeline.csv", timeline)
    if decisions:
        write_csv(args.output / "decisions.csv", decisions)
    reference, static_reference = summarize(metrics, args.output / "summary.json")
    print(f"best method {reference}, best static regional {static_reference}")


if __name__ == "__main__":
    main()
