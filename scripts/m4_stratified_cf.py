"""Stratified counterfactual candidates on the training maps.

Candidates are selected from the decision-time observable state: the observed
window w-1 leaf loads (from the trace load labels), the entity count, the
nominal prior for window w (from the plan P lines) and the direction of change.
This avoids the M4.12 flaw of picking the top prior leaves only.
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from m4_counterfactual import classify, confidence, read_plan, run_cf  # noqa: E402


def observed_loads(path):
    table = {}
    with Path(path).open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            f = line.split(",")
            table[(int(f[0]), int(f[1]))] = (float(f[2]), float(f[3]), float(f[4]))
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("plans", type=Path)
    parser.add_argument("traces", type=Path)
    parser.add_argument("loads", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--null-pairs", type=int, default=3)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 4])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = ["brc202d", "orz100d", "den101d"]
    raw, labels = [], []
    for index in range(1):
        ident = f"train-{index}-{names[index]}"
        trace = args.traces / f"{ident}.actual.m3bin"
        plan = args.plans / f"{ident}.uniform-16.plan"
        load_path = args.loads / f"{ident}.actual_loads.csv"
        if not trace.exists():
            continue
        rects, levels, prior = read_plan(plan)
        observed = observed_loads(load_path)
        for window in (12,):
            q = [observed.get((window - 1, leaf), (0, 0, 0))[0] for leaf in range(len(rects))]
            u = [observed.get((window - 1, leaf), (0, 0, 0))[2] for leaf in range(len(rects))]
            n = [observed.get((window - 1, leaf), (0, 0, 0))[1] for leaf in range(len(rects))]
            p = [prior.get((window, leaf), (0, 0, 0, 0))[0] for leaf in range(len(rects))]
            qm, um, nm, pm = (statistics.median(v) for v in (q, u, n, p))
            strata = {
                "query_hot": sorted(range(len(rects)), key=lambda l: -q[l])[:1],
                "update_hot": sorted(range(len(rects)), key=lambda l: -u[l])[:1],
                "both_hot": sorted(range(len(rects)),
                                   key=lambda l: -(q[l] + u[l]))[:1],
                "cold": [leaf for leaf in range(len(rects)) if q[leaf] == 0 and u[leaf] == 0][:1],
                "entity_dense": sorted(range(len(rects)), key=lambda l: -n[l])[:1],
                "prior_new_hot": sorted(range(len(rects)),
                                        key=lambda l: -(p[l] - q[l]))[:1],
                "prior_leaving": sorted(range(len(rects)),
                                        key=lambda l: -(q[l] - p[l]))[:1],
            }
            chosen = {leaf for leaves in strata.values() for leaf in leaves}
            for leaf in sorted(chosen):
                for mode_b in (2,):
                    for horizon in args.horizons:
                        diffs, state_ok = [], True
                        for pair in range(args.pairs):
                            order = pair % 2 == 1
                            out = run_cf(args.binary, trace, plan, window, leaf, mode_b,
                                         horizon, order=order)
                            state_ok = state_ok and out["state_equal"] == 1 and out["mismatches"] == 0
                            diffs.append(out["net_ms"])
                            raw.append({"trace": ident, "window": window, "leaf": leaf,
                                        "mode_to": mode_b, "horizon": horizon, "pair": pair,
                                        "order": int(order), "net_ms": out["net_ms"],
                                        "cost_a_ms": out["cost_a_ms"],
                                        "cost_b_ms": out["cost_b_ms"],
                                        "obs_prev_q": out["obs_prev_q"],
                                        "prior_q": out["prior_q"],
                                        "actual_future_q": out["actual_future_q"]})
                        nulls = []
                        for _ in range(args.null_pairs):
                            out = run_cf(args.binary, trace, plan, window, leaf, 0, horizon)
                            nulls.append(out["net_ms"])
                            if out["state_equal"] != 1:
                                state_ok = False
                        noise = statistics.median([abs(v) for v in nulls]) if nulls else 0.0
                        mean, low, high = confidence(diffs)
                        labels.append({
                            "trace": ident, "window": window, "leaf": leaf, "horizon": horizon,
                            "obs_prev_q": raw[-1]["obs_prev_q"], "prior_q": raw[-1]["prior_q"],
                            "actual_future_q": raw[-1]["actual_future_q"],
                            "mean_net_ms": mean, "ci_low_ms": low, "ci_high_ms": high,
                            "null_noise_ms": noise, "state_ok": int(state_ok),
                            "label": classify(mean, low, high, max(noise, 0.25)),
                        })
                        print(f"{ident} w{window} leaf{leaf} h{horizon}: {mean:+.2f} "
                              f"[{low:+.2f},{high:+.2f}] null {noise:.2f} "
                              f"{labels[-1]['label']}", flush=True)
    with (args.output / "raw_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw[0]))
        writer.writeheader()
        writer.writerows(raw)
    with (args.output / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(labels[0]))
        writer.writeheader()
        writer.writerows(labels)
    from collections import Counter
    print("labels", dict(Counter(r["label"] for r in labels)))


if __name__ == "__main__":
    main()
