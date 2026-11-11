"""Counterfactual single-region benefit labels from the real C++ partition index.

For each (trace, window, leaf, candidate mode, horizon) the harness rebuilds the
prefix twice and measures the horizon cost of keeping SCAN versus switching the
target leaf at the window start, with identical future events and all other
leaves unchanged. Labels are offline diagnostics only; online policies never
read future events.
"""
import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from prepare import json_file  # noqa: E402

SIDE = 16


def read_plan(path):
    rects, levels, prior = [], [], {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line or line[0] == "#":
            continue
        f = line.split()
        if f[0] == "L":
            rects.append(tuple(int(v) for v in f[1:5]))
            levels.append(int(f[5]))
        elif f[0] == "P":
            prior[(int(f[1]), int(f[2]))] = (float(f[3]), float(f[4]), float(f[5]), float(f[6]))
    return rects, levels, prior


def run_cf(binary, trace, plan, window, leaf, mode_b, horizon, verify=False):
    args = [str(binary), "partition_cf", "160", "32", "0", str(trace), str(plan),
            str(window), str(leaf), "0", str(mode_b), str(horizon), "1" if verify else "0"]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    out = {}
    for token in result.stdout.split():
        key, value = token.split("=", 1)
        if "/" in value:
            value = value.split("/")[0]
        out[key] = float(value)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("plans", type=Path)
    parser.add_argument("traces", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--windows", nargs="+", type=int, default=[4, 12, 20])
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--noise-ms", type=float, default=2.0, help="null-arm measurement noise floor")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(3):
        ident = f"train-{index}-{['brc202d','orz100d','den101d'][index]}"
        trace = args.traces / f"{ident}.actual.m3bin"
        plan = args.plans / f"{ident}.uniform-16.plan"
        rects, levels, prior = read_plan(plan)
        if not trace.exists():
            continue
        for window in args.windows:
            loads = [(prior.get((window, leaf), (0, 0, 0, 0))[0], leaf)
                     for leaf in range(len(rects))]
            loads.sort(reverse=True)
            hot = [leaf for _, leaf in loads[: args.top]]
            for leaf in hot:
                for mode_b in (1, 2):
                    for horizon in args.horizons:
                        values = []
                        for _ in range(args.repeats):
                            out = run_cf(args.binary, trace, plan, window, leaf, mode_b, horizon)
                            values.append(out)
                        nets = [v["net_ms"] for v in values]
                        median = statistics.median(nets)
                        spread = max(nets) - min(nets)
                        rows.append({
                            "trace": ident, "window": window, "leaf": leaf,
                            "area": (rects[leaf][2] - rects[leaf][0]) *
                                    (rects[leaf][3] - rects[leaf][1]) / 16.0,
                            "level": levels[leaf], "mode_from": 0, "mode_to": mode_b,
                            "horizon": horizon,
                            "prior_q": prior.get((window, leaf), (0, 0, 0, 0))[0],
                            "prior_u": prior.get((window, leaf), (0, 0, 0, 0))[2],
                            "entities": values[0]["entities"],
                            "load_q": values[0]["load_q"], "load_u": values[0]["load_u"],
                            "load_c": values[0]["load_c"],
                            "cost_keep_ms": statistics.median([v["cost_a_ms"] for v in values]),
                            "cost_switch_ms": statistics.median([v["cost_b_ms"] for v in values]),
                            "switch_ms": statistics.median([v["switch_b_ms"] for v in values]),
                            "net_ms": median, "net_spread_ms": spread,
                            "uncertain": abs(median) < max(args.noise_ms, 3 * spread),
                            "queries": values[0]["queries"],
                        })
                        print(f"{ident} w{window} leaf{leaf} ->{mode_b} h{horizon}: "
                              f"net {median:+.3f} ms (spread {spread:.3f})", flush=True)
    with (args.output / "counterfactual.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    certain = [r for r in rows if not r["uncertain"]]
    positive = [r for r in certain if r["net_ms"] > 0]
    json_file(args.output / "summary.json", {"labels": len(rows), "certain": len(certain),
                                             "positive": len(positive),
                                             "uncertain": len(rows) - len(certain),
                                             "repeats": args.repeats})
    print(f"labels {len(rows)}, certain {len(certain)}, positive {len(positive)}")


if __name__ == "__main__":
    main()
