"""Fair comparison on public-map scenarios: history vs scenario feed-forward."""
import argparse
import csv
import json
import random
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

import psutil

from prepare import json_file, sha

METHODS = [
    "Fixed Regional Scan",
    "Fixed Local Fine Grid",
    "Fixed Local Quadtree",
    "Threshold-Based Adaptive Regional Index Selection",
    "Threshold-Based Adaptive Regional Index Selection (Scenario)",
    "Cost-Aware Greedy Regional Index Selection",
    "Cost-Aware Greedy Regional Index Selection (Scenario)",
    "Uniform Grid",
    "PR Quadtree",
    "Boost R*-Tree",
]
POLICIES = {method for method in METHODS if "Adaptive" in method or "Greedy" in method}
SCENARIO = {method for method in POLICIES if method.endswith("(Scenario)")}
METRIC = ["build_ms", "online_ms", "query_ms", "update_ms", "switch_ms", "policy_ms",
          "p50_us", "p95_us", "p99_us", "peak_rss_mb", "prebuild_rss_mb"]


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


def parse_validation(text):
    return {key: int(value) for key, value in (part.split("=") for part in text.strip().split())}


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap(values, rng, draws=10000):
    values = list(values)
    medians = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(draws))
    return [medians[int(0.025 * draws)], medians[int(0.975 * draws)]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("selector", type=Path)
    parser.add_argument("binary", type=Path)
    parser.add_argument("scenarios", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--actuals", type=int, default=3)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    fine = int(selection["fine_cell_m"])
    quad = int(selection["quad_capacity"])
    classic = selection["classic"]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "profile").mkdir(exist_ok=True)

    traces = []
    for split in args.splits:
        for index, map_name in enumerate(config["maps"][split]):
            ident = f"{split}-{index}-{Path(map_name).stem}"
            count = args.actuals if split == "test" else min(2, args.actuals)
            for actual in range(count):
                name = (f"{ident}.actual.m3bin" if actual == 0
                        else f"{ident}.actual{actual}.m3bin")
                traces.append({"split": split, "scenario": ident, "actual": actual,
                               "trace": args.scenarios / "traces" / name,
                               "prior": args.scenarios / "scenarios" / f"{ident}.nominal.csv",
                               "map": Path(map_name).stem})

    metrics, decisions, timeline, regions = [], [], [], []
    for entry in traces:
        order = list(METHODS)
        random.Random(f"{entry['scenario']}-{entry['actual']}").shuffle(order)
        for method in order:
            param = 1
            if method == "Uniform Grid":
                param = classic["Uniform Grid"]
            elif method in ("PR Quadtree", "Boost R*-Tree"):
                param = classic[method]
            prior = entry["prior"] if method in SCENARIO else "NONE"
            truth = parse_validation(invoke(args.binary, ["validate", method, fine, quad, param,
                                                          args.cpu, entry["trace"], str(args.selector), prior]))
            assert truth["mismatches"] == 0, (method, entry["trace"])
            head = parse_kv(invoke(args.binary, ["measure", method, fine, quad, param,
                                                 args.cpu, entry["trace"], str(args.selector), prior]))
            assert head["query_count"] == truth["queries"]
            row = {"method": method, "split": entry["split"], "scenario": entry["scenario"],
                   "actual": entry["actual"], "map": entry["map"], "parameter": param,
                   "trace_sha256": sha(entry["trace"]),
                   "prior_sha256": sha(entry["prior"]) if method in SCENARIO else "",
                   "correctness_pass": True,
                   **{field: head[field] for field in METRIC},
                   "total_ms": float(head["build_ms"]) + float(head["online_ms"]),
                   "switches": head["switches"], "promotions": head["promotions"],
                   "demotions": head["demotions"], "retunes": head["retunes"],
                   "final_scan": head["final_scan"], "final_fine": head["final_fine"],
                   "final_quad": head["final_quad"], "decision_calls": head["decision_calls"],
                   "decision_hash": head["decision_hash"]}
            metrics.append(row)
            if method in POLICIES:
                prefix = args.output / "profile" / f"{entry['scenario']}-a{entry['actual']}-{len(metrics)}"
                profile = parse_kv(invoke(args.binary, ["profile", method, fine, quad, param,
                                                        args.cpu, entry["trace"], str(args.selector),
                                                        prefix, prior]))
                assert profile["query_count"] == head["query_count"]
                assert profile["decision_hash"] == head["decision_hash"]
                for item in read_csv(prefix.parent / f"{prefix.name}_decisions.csv"):
                    decisions.append({"method": method, "split": entry["split"],
                                      "scenario": entry["scenario"], "actual": entry["actual"], **item})
                for item in read_csv(prefix.parent / f"{prefix.name}_timeline.csv"):
                    timeline.append({"method": method, "split": entry["split"],
                                     "scenario": entry["scenario"], "actual": entry["actual"], **item})
                for item in read_csv(prefix.parent / f"{prefix.name}_regions.csv"):
                    regions.append({"method": method, "split": entry["split"],
                                    "scenario": entry["scenario"], "actual": entry["actual"], **item})
            print(f"{entry['scenario']} a{entry['actual']} {method} online {head['online_ms']:.1f} ms",
                  flush=True)
        write_csv(args.output / "metrics.csv", metrics)
    write_csv(args.output / "decisions.csv", decisions)
    write_csv(args.output / "timeline.csv", timeline)
    write_csv(args.output / "regions.csv", regions)
    json_file(args.output / "config.json", config)
    json_file(args.output / "selection.json", selection)
    (args.output / "selector.txt").write_bytes(args.selector.read_bytes())

    summary = {}
    for split in args.splits:
        by_method = defaultdict(list)
        for row in metrics:
            if row["split"] == split:
                by_method[row["method"]].append(row)
        summary[split] = {method: {field: statistics.median(float(r[field]) for r in rows)
                                   for field in METRIC} | {"total_ms": statistics.median(
                                       float(r["total_ms"]) for r in rows),
                                       "switches": statistics.median(r["switches"] for r in rows),
                                       "promotions": statistics.median(r["promotions"] for r in rows),
                                       "demotions": statistics.median(r["demotions"] for r in rows)}
                           for method, rows in by_method.items()}
    json_file(args.output / "summary.json", summary)

    # Paired ratios over the test traces: feed-forward vs history and vs the best static.
    pairs = defaultdict(dict)
    for row in metrics:
        if row["split"] == "test":
            pairs[(row["scenario"], row["actual"])][row["method"]] = float(row["online_ms"])
    rng = random.Random(0x4D4D)
    paired = {}
    static_methods = ["Fixed Regional Scan", "Fixed Local Fine Grid", "Fixed Local Quadtree",
                      "Uniform Grid", "PR Quadtree", "Boost R*-Tree"]
    best_static = min(static_methods,
                      key=lambda m: statistics.median(pairs[k][m] for k in pairs))
    for scenario, history in (("Cost-Aware Greedy Regional Index Selection (Scenario)",
                               "Cost-Aware Greedy Regional Index Selection"),
                              ("Threshold-Based Adaptive Regional Index Selection (Scenario)",
                               "Threshold-Based Adaptive Regional Index Selection")):
        ratios = [pairs[k][scenario] / pairs[k][history] for k in pairs]
        paired[f"{scenario} / history"] = {
            "median_ratio": statistics.median(ratios),
            "bootstrap95": bootstrap(ratios, rng),
            "wins": sum(1 for r in ratios if r < 1.0), "n": len(ratios)}
    for method in METHODS:
        ratios = [pairs[k][method] / pairs[k][best_static] for k in pairs]
        paired[f"{method} / {best_static}"] = {
            "median_ratio": statistics.median(ratios),
            "bootstrap95": bootstrap(ratios, rng),
            "wins": sum(1 for r in ratios if r < 1.0), "n": len(ratios)}
    json_file(args.output / "paired.json", {"best_static": best_static, "pairs": paired})
    print("best static:", best_static)


if __name__ == "__main__":
    main()
