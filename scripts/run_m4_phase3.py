"""Phase-3 risk and horizon study: expected cost, quantile penalty and CVaR."""
import argparse
import json
import statistics
import subprocess
from pathlib import Path

from prepare import json_file, sha

METHODS = {
    "expected": "Uncertainty-Aware Feed-Forward Regional Index Selection",
    "quantile": "Quantile-Cost-Penalized Feed-Forward Regional Index Selection",
    "cvar": "CVaR-Risk-Constrained Feed-Forward Regional Index Selection",
}
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
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_config(base, directory, uncertainty, risk_lambda, horizon, alpha=0.8):
    lines = []
    for line in base.read_text(encoding="utf-8").splitlines():
        if line.startswith(("uncertainty", "risk_lambda", "cvar_alpha", "horizon")):
            continue
        lines.append(line)
    lines += [f"uncertainty {uncertainty}", f"risk_lambda {risk_lambda:.4f}",
              f"cvar_alpha {alpha:.4f}", f"horizon {horizon}"]
    path = directory / f"selector_{uncertainty}_{risk_lambda:.2f}_{horizon}.txt"
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
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--samples-tag", default="", help="suffix for the sample prior, e.g. _m32")
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
            count = 0
            while True:
                candidate = (args.scenarios / "traces" /
                             (f"{ident}.actual.m3bin" if count == 0
                              else f"{ident}.actual{count}.m3bin"))
                if not candidate.exists():
                    break
                count += 1
            for actual in range(count):
                name = (f"{ident}.actual.m3bin" if actual == 0
                        else f"{ident}.actual{actual}.m3bin")
                traces.append({"split": split, "scenario": ident, "actual": actual,
                               "trace": args.scenarios / "traces" / name,
                               "prior": args.scenarios / "scenarios" /
                                        f"{ident}.nominal_samples{args.samples_tag}.csv"})
    validation = [entry for entry in traces if entry["split"] == "validation"]

    def measure(entry, method, cfg):
        return parse_kv(invoke(args.binary, ["measure", method, fine, quad, 1, args.cpu,
                                             entry["trace"], cfg, entry["prior"]]))

    tuning = []
    for key, uncertainty in (("quantile", 2), ("cvar", 3)):
        for lam in args.lambdas:
            cfg = make_config(args.selector, args.output / "tmp", uncertainty, lam, 1)
            values = [float(measure(entry, METHODS[key], cfg)["online_ms"]) for entry in validation]
            tuning.append({"method": key, "risk_lambda": lam, "horizon": 1,
                           "median_online_ms": statistics.median(values),
                           "runs": ";".join(f"{v:.4f}" for v in values)})
            print(f"{key} lambda {lam} -> {statistics.median(values):.1f} ms", flush=True)
    best_lambda = {}
    for key in ("quantile", "cvar"):
        rows = [row for row in tuning if row["method"] == key]
        best_lambda[key] = min(rows, key=lambda row: row["median_online_ms"])["risk_lambda"]
    for horizon in args.horizons:
        if horizon == 1:
            continue
        cfg = make_config(args.selector, args.output / "tmp", 3, best_lambda["cvar"], horizon)
        values = [float(measure(entry, METHODS["cvar"], cfg)["online_ms"]) for entry in validation]
        tuning.append({"method": "cvar", "risk_lambda": best_lambda["cvar"], "horizon": horizon,
                       "median_online_ms": statistics.median(values),
                       "runs": ";".join(f"{v:.4f}" for v in values)})
        print(f"cvar horizon {horizon} -> {statistics.median(values):.1f} ms", flush=True)
    write_csv(args.output / "tuning.csv", tuning)
    json_file(args.output / "selected.json", {"best_lambda": best_lambda})
    horizon_rows = [row for row in tuning if row["method"] == "cvar" and
                    row["risk_lambda"] == best_lambda["cvar"]]
    best_horizon = min(horizon_rows, key=lambda row: row["median_online_ms"])["horizon"]

    rows = []
    for entry in traces:
        jobs = [("expected", 1, 0.0, 1), ("quantile", 2, best_lambda["quantile"], 1),
                ("cvar", 3, best_lambda["cvar"], 1), ("cvar", 3, best_lambda["cvar"], best_horizon)]
        for key, uncertainty, lam, horizon in jobs:
            cfg = make_config(args.selector, args.output / "tmp", uncertainty, lam, horizon)
            truth = parse_validation(invoke(args.binary, ["validate", METHODS[key], fine, quad, 1,
                                                          args.cpu, entry["trace"], cfg, entry["prior"]]))
            assert truth["mismatches"] == 0
            head = measure(entry, METHODS[key], cfg)
            rows.append({"method": key, "horizon": horizon, "risk_lambda": lam,
                         "split": entry["split"], "scenario": entry["scenario"],
                         "actual": entry["actual"], "trace_sha256": sha(entry["trace"]),
                         "prior_sha256": sha(entry["prior"]),
                         "correctness_pass": True,
                         **{field: head[field] for field in METRIC},
                         "switches": head["switches"], "promotions": head["promotions"],
                         "decision_calls": head["decision_calls"]})
            print(f"{entry['scenario']} a{entry['actual']} {key} h{horizon} {head['online_ms']:.1f} ms",
                  flush=True)
        write_csv(args.output / "metrics.csv", rows)
    print("best lambda", best_lambda, "best horizon", best_horizon)


if __name__ == "__main__":
    main()
