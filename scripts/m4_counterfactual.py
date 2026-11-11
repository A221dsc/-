"""Counterfactual v2 driver: paired interleaved measurements, A/A null controls,
confidence intervals and three-way labels (positive, negative, near-zero,
unresolvable). Raw pairs are preserved; the full and local cost views are
separated (profile only for the local diagnostic subset).
"""
import argparse
import csv
import math
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from prepare import json_file  # noqa: E402

T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}


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


def run_cf(binary, trace, plan, window, leaf, mode_b, horizon, verify=False, profile=False,
           order=False):
    args = [str(binary), "partition_cf", "160", "32", "0", str(trace), str(plan),
            str(window), str(leaf), "0", str(mode_b), str(horizon),
            "1" if verify else "0", "1" if profile else "0", "1" if order else "0"]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    out = {}
    for token in result.stdout.split():
        key, value = token.split("=", 1)
        if key in ("window_q", "window_u"):
            continue
        if "/" in value:
            value = value.split("/")[0]
        out[key] = float(value)
    return out


def confidence(diffs):
    n = len(diffs)
    mean = statistics.fmean(diffs)
    if n < 2:
        return mean, mean, mean
    sd = statistics.stdev(diffs)
    se = sd / math.sqrt(n)
    t = T95.get(n, 1.96)
    return mean, mean - t * se, mean + t * se


def classify(mean, low, high, noise, near_zero=0.5):
    if low > noise:
        return "reliable_positive"
    if high < -noise:
        return "reliable_negative"
    if abs(mean) <= near_zero and (high - low) <= 1.0:
        return "near_zero_reproducible"
    return "unresolvable"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("plans", type=Path)
    parser.add_argument("traces", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--windows", nargs="+", type=int, default=[12])
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--null-pairs", type=int, default=3)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = ["brc202d", "orz100d", "den101d"]
    raw, summary = [], []
    for index in range(3):
        ident = f"train-{index}-{names[index]}"
        trace = args.traces / f"{ident}.actual.m3bin"
        plan = args.plans / f"{ident}.uniform-16.plan"
        if not trace.exists():
            continue
        rects, levels, prior = read_plan(plan)
        for window in args.windows:
            loads = sorted(((prior.get((window, leaf), (0, 0, 0, 0))[0], leaf)
                            for leaf in range(len(rects))), reverse=True)
            for leaf in [leaf for _, leaf in loads[: args.top]]:
                for mode_b in (1, 2):
                    for horizon in args.horizons:
                        diffs, state_ok = [], True
                        for pair in range(args.pairs):
                            order = pair % 2 == 1
                            out = run_cf(args.binary, trace, plan, window, leaf, mode_b,
                                         horizon, verify=False, profile=False, order=order)
                            state_ok = state_ok and out["state_equal"] == 1 and out["mismatches"] == 0
                            diffs.append(out["net_ms"])
                            raw.append({"trace": ident, "window": window, "leaf": leaf,
                                        "mode_to": mode_b, "horizon": horizon,
                                        "pair": pair, "order": int(order),
                                        "cost_a_ms": out["cost_a_ms"], "cost_b_ms": out["cost_b_ms"],
                                        "net_ms": out["net_ms"], "load_q": out["load_q"],
                                        "load_u": out["load_u"], "entities": out["entities"]})
                        null_diffs = []
                        for _ in range(args.null_pairs):
                            out = run_cf(args.binary, trace, plan, window, leaf, 0, horizon,
                                         verify=False, profile=False, order=False)
                            null_diffs.append(out["net_ms"])
                            if out["state_equal"] != 1:
                                state_ok = False
                        noise = statistics.median([abs(v) for v in null_diffs]) if null_diffs else 0.0
                        mean, low, high = confidence(diffs)
                        summary.append({
                            "trace": ident, "window": window, "leaf": leaf, "mode_to": mode_b,
                            "horizon": horizon, "pairs": args.pairs,
                            "mean_net_ms": mean, "ci_low_ms": low, "ci_high_ms": high,
                            "null_noise_ms": noise, "state_ok": int(state_ok),
                            "label": classify(mean, low, high, max(noise, 0.25)),
                            "load_q": raw[-1]["load_q"], "load_u": raw[-1]["load_u"],
                            "entities": raw[-1]["entities"],
                        })
                        print(f"{ident} w{window} leaf{leaf} ->{mode_b} h{horizon}: "
                              f"{mean:+.2f} [{low:+.2f},{high:+.2f}] null {noise:.2f} "
                              f"{summary[-1]['label']}", flush=True)
    with (args.output / "raw_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw[0]))
        writer.writeheader()
        writer.writerows(raw)
    with (args.output / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    from collections import Counter
    counts = Counter(r["label"] for r in summary)
    json_file(args.output / "summary.json", {"candidates": len(summary), "labels": counts,
                                             "pairs": args.pairs, "null_pairs": args.null_pairs,
                                             "state_ok": all(r["state_ok"] for r in summary)})
    print("labels", dict(counts))


if __name__ == "__main__":
    main()
