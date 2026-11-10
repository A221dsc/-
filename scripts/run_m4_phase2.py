"""Phase-2 feed-forward policies: calibrated point, distribution expectation and risk."""
import argparse
import json
import statistics
import subprocess
from pathlib import Path

from prepare import json_file, sha

METHODS = {
    "calibrated": "Footprint-Calibrated Feed-Forward Regional Index Selection",
    "expected": "Uncertainty-Aware Feed-Forward Regional Index Selection",
    "risk": "Risk-Constrained Uncertainty-Aware Feed-Forward Regional Index Selection",
}
PRIOR = {"calibrated": "nominal_calibrated.csv", "expected": "nominal_samples.csv",
         "risk": "nominal_samples.csv"}
METRIC = ["build_ms", "online_ms", "query_ms", "update_ms", "switch_ms", "policy_ms",
          "p50_us", "p95_us", "p99_us", "peak_rss_mb"]


def invoke(binary, args):
    result = subprocess.run([str(binary), *map(str, args)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout


def parse_kv(text):
    row = {}
    for token in text.strip().split():
        key, value = token.split("=", 1)
        row[key] = float(value) if any(c in value for c in ".eE") else int(value)
    return row


def parse_validation(text):
    return {key: int(value) for key, value in (part.split("=") for part in text.strip().split())}


def write_csv(path, rows):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def config_with_risk(base: Path, directory: Path, risk_lambda: float) -> Path:
    lines = []
    for line in base.read_text(encoding="utf-8").splitlines():
        if line.startswith("risk_lambda"):
            continue
        lines.append(line)
        if line.startswith("uncertainty"):
            lines.append(f"risk_lambda {risk_lambda:.6f}")
    if not any(line.startswith("uncertainty") for line in lines):
        lines.append("uncertainty 2")
        lines.append(f"risk_lambda {risk_lambda:.6f}")
    path = directory / f"selector_risk_{risk_lambda:.2f}.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("selector", type=Path)
    parser.add_argument("binary", type=Path)
    parser.add_argument("scenarios", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--risk-values", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    fine, quad = int(selection["fine_cell_m"]), int(selection["quad_capacity"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "tmp").mkdir(exist_ok=True)
    traces = []
    for split in ("validation", "test"):
        for index, map_name in enumerate(config["maps"][split]):
            ident = f"{split}-{index}-{Path(map_name).stem}"
            count = 3 if split == "test" else 2
            for actual in range(count):
                name = (f"{ident}.actual.m3bin" if actual == 0
                        else f"{ident}.actual{actual}.m3bin")
                traces.append({"split": split, "scenario": ident, "actual": actual,
                               "trace": args.scenarios / "traces" / name,
                               "prior": args.scenarios / "scenarios" / f"{ident}.nominal_calibrated.csv",
                               "samples": args.scenarios / "scenarios" / f"{ident}.nominal_samples.csv"})
    validation = [t for t in traces if t["split"] == "validation"]
    risk_configs = {value: config_with_risk(args.selector, args.output / "tmp", value)
                    for value in args.risk_values}
    sweep = []
    for value, path in risk_configs.items():
        values = []
        for entry in validation:
            head = parse_kv(invoke(args.binary, ["measure", METHODS["risk"], fine, quad, 1,
                                                 args.cpu, entry["trace"], path, entry["samples"]]))
            values.append(float(head["online_ms"]))
        sweep.append({"risk_lambda": value, "median_online_ms": statistics.median(values),
                      "runs": ";".join(f"{v:.4f}" for v in values)})
        print(f"risk lambda {value} -> {statistics.median(values):.1f} ms", flush=True)
    write_csv(args.output / "risk_sweep.csv", sweep)
    best = min(sweep, key=lambda row: row["median_online_ms"])
    json_file(args.output / "risk_selection.json", best)

    rows = []
    for entry in traces:
        for key, method in METHODS.items():
            prior = entry["samples"] if key == "risk" else entry["prior"]
            selector = risk_configs[best["risk_lambda"]] if key == "risk" else args.selector
            truth = parse_validation(invoke(args.binary, ["validate", method, fine, quad, 1,
                                                          args.cpu, entry["trace"], selector, prior]))
            assert truth["mismatches"] == 0
            head = parse_kv(invoke(args.binary, ["measure", method, fine, quad, 1, args.cpu,
                                                 entry["trace"], selector, prior]))
            assert head["query_count"] == truth["queries"]
            rows.append({"method": key, "full_name": method, "split": entry["split"],
                         "scenario": entry["scenario"], "actual": entry["actual"],
                         "trace_sha256": sha(entry["trace"]),
                         "prior_sha256": sha(prior),
                         "correctness_pass": True, **{field: head[field] for field in METRIC},
                         "total_ms": float(head["build_ms"]) + float(head["online_ms"]),
                         "switches": head["switches"], "promotions": head["promotions"],
                         "demotions": head["demotions"],
                         "decision_hash": head["decision_hash"]})
            print(f"{entry['scenario']} a{entry['actual']} {key} {head['online_ms']:.1f} ms",
                  flush=True)
        write_csv(args.output / "metrics_phase2.csv", rows)
    print("done")


if __name__ == "__main__":
    main()
