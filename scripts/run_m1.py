"""Run one validated M1 group with paired seeds, pinned single-core processes."""
import argparse
import csv
import json
import platform
import random
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import psutil
import pyarrow

from prepare import json_file, sha
from trace import write_trace

MEASUREMENT = (
    "build_ms total_wall_ms query_wall_ms base_update_ms position_update_ms "
    "insert_delete_ms query_p50_us query_p95_us query_p99_us update_per_sec "
    "update_count insert_delete_count query_count candidate_count node_visits "
    "result_count peak_rss_mb prebuild_peak_rss_mb trace_load_ms"
).split()


def invoke(binary, mode, algorithm, param, cpu, trace):
    args = [str(binary.resolve()), mode, algorithm, str(param), str(cpu), str(trace)]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{args} exited {result.returncode}: {result.stderr.strip()}")
    return result.stdout.strip().split(",")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measure(args, config, trace, meta, algorithm, param, digest, verified=None):
    if verified is None:
        verified = invoke(args.binary, "validate", algorithm, param, args.cpu, trace)
    checks, peak = map(int, verified)
    fields = invoke(args.binary, "measure", algorithm, param, args.cpu, trace)
    observed = dict(zip(MEASUREMENT, fields, strict=True))
    counts = meta["counts"]
    assert checks == int(observed["query_count"]) == counts["R"] + counts["C"]
    assert counts.get("U", 0) == int(observed["update_count"])
    assert counts.get("I", 0) + counts.get("D", 0) == int(observed["insert_delete_count"])
    return {
        "experiment_id": config["experiment_id"], "trace_id": meta["trace_id"],
        "seed": meta["seed"], "dataset": config["dataset"], "dataset_variant": config["dataset_variant"],
        "policy": "", "baseline": algorithm, "parameter": param, "entity_count": peak,
        "initial_entities": meta["initial_entities"], "steps": meta["steps"],
        "trajectory_records": meta["trajectory_records"], **observed,
        "exact_refinements": observed["candidate_count"],
        "query_per_sec": checks * 1000 / float(observed["query_wall_ms"]),
        "total_with_build_ms": float(observed["total_wall_ms"]) + float(observed["build_ms"]),
        "local_update_ms": 0, "build_switch_ms": 0, "policy_ms": 0, "switches": 0,
        "correctness_pass": True, "hardware_id": args.hardware_id, "git_commit": args.commit,
        "config_sha256": digest, "trace_sha256": meta["trace_sha256"],
    }


def summarize(rows, out):
    by_algo = defaultdict(dict)
    for row in rows:
        by_algo[row["baseline"]][row["seed"]] = row
    stats = {}
    for algo, seeds in by_algo.items():
        stats[algo] = {
            field: statistics.median(float(r[field]) for r in seeds.values())
            for field in [*MEASUREMENT, "query_per_sec", "total_with_build_ms", "exact_refinements"]
            if all(r[field] != "" for r in seeds.values())
        }
    best = min(stats, key=lambda a: stats[a]["total_wall_ms"])
    rng = random.Random(10781)
    seeds = sorted(by_algo[best])
    intervals = {}
    for algo, group in by_algo.items():
        ratio = {s: float(group[s]["total_wall_ms"]) / float(by_algo[best][s]["total_wall_ms"])
                 for s in seeds}
        draws = sorted(statistics.median(ratio[s] for s in rng.choices(seeds, k=len(seeds)))
                       for _ in range(10000))
        intervals[algo] = {
            "median_paired_online_time_ratio_to_reference": statistics.median(ratio.values()),
            "bootstrap95_percentile_paired": [draws[249], draws[9749]],
        }
    json_file(out, {"medians": stats, "reference_lowest_median_online": best, "pairwise": intervals})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("binary", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--source", type=Path, help="directory containing normalized train/validation/test CSVs")
    p.add_argument("--traces", type=Path, default=Path("data/processed/traces"))
    p.add_argument("--cpu", type=int, default=min(psutil.Process().cpu_affinity()))
    p.add_argument("--hardware-id", default="local-machine")
    args = p.parse_args()
    args.commit = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["dataset"] != "SYNTHETIC" and args.source is None:
        p.error("--source is required for a real trajectory group")
    args.output.mkdir(parents=True, exist_ok=False)
    digest = sha(args.config)
    (args.output / "config.json").write_bytes(args.config.read_bytes())
    if args.source:
        for name in ("metadata.json", "quality.json"):
            (args.output / name).write_bytes((args.source / name).read_bytes())
    json_file(args.output / "environment.json", {
        "system": platform.platform(), "processor": platform.processor(),
        "physical_cores": psutil.cpu_count(False), "logical_cores": psutil.cpu_count(),
        "memory_bytes": psutil.virtual_memory().total, "python": platform.python_version(),
        "pyarrow": pyarrow.__version__, "pinned_logical_cpu": args.cpu, "threads": 1,
        "config_sha256": digest, "binary_sha256": sha(args.binary), "repo_head": args.commit,
        "source_hashes": {x.as_posix():sha(x) for base in ("src","scripts","tests")
                          for x in sorted(Path(base).rglob("*"))
                          if x.is_file() and "__pycache__" not in x.parts},
        "worktree_note": "experiment code is local; repo HEAD alone does not identify it",
        "cache_state": "trace parsed into process memory before timing; no OS cache flush",
        "reproduction_argv": sys.argv,
    })

    validation = []
    selected = {}
    traces = []
    trace,meta = write_trace(config,config["validation_seed"],"validation",args.source,args.traces)
    traces.append(meta)
    trials = [(algorithm,param) for algorithm, params in config["parameter_search"].items()
              for param in params]
    verified = {(a,param): invoke(args.binary,"validate",a,param,args.cpu,trace)
                for a,param in trials}
    timings = defaultdict(list)
    for repeat in range(5):
        random.Random(3109 + repeat).shuffle(trials)
        for algorithm, param in trials:
            row = measure(args,config,trace,meta,algorithm,param,digest,verified[algorithm,param])
            validation.append({**row, "repeat": repeat})
            timings[algorithm,param].append(float(row["total_wall_ms"]))
    for algorithm, params in config["parameter_search"].items():
        selected[algorithm] = min(params,key=lambda param:statistics.median(timings[algorithm,param]))
    write_csv(args.output / "tuning.csv", validation)
    json_file(args.output / "selected.json", selected)
    rows = []
    for seed in config["test_seeds"]:
        trace,meta = write_trace(config,seed,"test",args.source,args.traces)
        traces.append(meta)
        order = list(selected)
        random.Random(seed ^ 0xFEBE).shuffle(order)
        for algo in order:
            row = measure(args,config,trace,meta,algo,selected[algo],digest)
            rows.append(row)
            print(seed, algo, row["query_count"], row["total_wall_ms"], flush=True)
    write_csv(args.output / "metrics.csv", rows)
    json_file(args.output / "trace_manifest.json", traces)
    summarize(rows,args.output / "summary.json")


if __name__ == "__main__":
    main()
