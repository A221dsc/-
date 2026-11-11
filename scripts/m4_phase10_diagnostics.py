"""Decision-quality diagnostics for the partition Double DQN.

Runs a trained model on one trace, logs every leaf/window decision together with
the Q values, the requested and executed actions and the realized per-leaf cost,
then reports selection-quality metrics. Logging is a separate diagnostic run and
never contributes to the headline timing.
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from m4_dqn_partition import Agent, FEATURES, Env, features, leaf_geometry, read_plan  # noqa: E402


def load_agent(path, has_scenario):
    agent = Agent(0, has_scenario=has_scenario)
    data = np.load(path)
    for name, param in (("W1", agent.online.W1), ("b1", agent.online.b1),
                        ("W2", agent.online.W2), ("b2", agent.online.b2),
                        ("W3", agent.online.W3), ("b3", agent.online.b3)):
        param[...] = data[name]
    return agent


def spearman(a, b):
    if len(a) < 3:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("plans", type=Path)
    parser.add_argument("traces", type=Path)
    parser.add_argument("--scenario", default="test-0-brc100d")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    trace = args.traces / f"{args.scenario}.actual.m3bin"
    rows = []
    for seed in (11, 22, 33):
        path = args.weights / f"weights-scenario-{seed}.npz"
        agent = load_agent(path, True)
        for partition in ("uniform-16", "adaptive-query"):
            plan_path = args.plans / f"{args.scenario}.{partition}.plan"
            rects, levels, prior = read_plan(plan_path)
            areas, aspect = leaf_geometry(rects)
            env = Env(args.binary, trace, plan_path)
            previous_mode = np.asarray(env.state["modes"])
            previous_cost = None
            next_window = 0
            per_window = []
            while not env.state["done"]:
                state = env.state
                feats = features(state, min(23, next_window), areas, aspect, levels, prior, True)
                q = agent.online.forward(feats)
                mode = np.asarray(state["modes"])
                mask = agent.mask(state)
                qm = np.where(mask, q, -1e9)
                actions = qm.argmax(axis=1)
                benefit = qm[np.arange(len(mode)), actions] - qm[np.arange(len(mode)), mode]
                benefit = np.where(actions == mode, 0.0, benefit)
                result = env.step(actions, benefit)
                applied = np.asarray(result["applied"])
                rejected = np.asarray(result["rejected"])
                region_us = np.asarray(result["region_us"])
                for leaf in range(len(mode)):
                    rows.append({
                        "seed": seed, "partition": partition, "window": next_window,
                        "leaf": leaf, "entities": state["entities"][leaf],
                        "queries": state["queries"][leaf], "updates": state["updates"][leaf],
                        "coverage": state["coverage"][leaf], "mode": int(mode[leaf]),
                        "q_scan": float(q[leaf, 0]), "q_fine": float(q[leaf, 1]),
                        "q_quad": float(q[leaf, 2]), "requested": int(actions[leaf]),
                        "applied": int(applied[leaf]), "rejected": int(rejected[leaf]),
                        "benefit": float(benefit[leaf]), "cost_us": float(region_us[leaf]),
                        "delta_cost": (float(region_us[leaf] - previous_cost[leaf])
                                       if previous_cost is not None else 0.0)})
                previous_cost = region_us
                previous_mode = applied
                per_window.append({"promotions": int(((applied != 0) & (applied != mode)).sum()),
                                   "budget": 4.0})
                next_window += 1
            env.close()
    out = args.output / "decision_log.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # metrics
    import collections
    metrics = []
    for seed in (11, 22, 33):
        for partition in ("uniform-16", "adaptive-query"):
            subset = [r for r in rows if r["seed"] == seed and r["partition"] == partition]
            by_window = collections.defaultdict(list)
            for r in subset:
                by_window[r["window"]].append(r)
            hot_total = hot_upgraded = upd_total = upd_scan = 0
            promotions = 0
            short_release = 0
            per_leaf = collections.defaultdict(list)
            for r in subset:
                per_leaf[r["leaf"]].append((r["window"], r["applied"]))
            for leaf, seq in per_leaf.items():
                seq.sort()
                for i in range(len(seq) - 1):
                    if seq[i][1] != 0 and seq[i + 1][1] == 0 and seq[i + 1][0] - seq[i][0] <= 3:
                        short_release += 1
            budget = []
            benefits, deltas = [], []
            missed = 0
            for window, group in sorted(by_window.items()):
                queries = np.array([r["queries"] for r in group], dtype=float)
                updates = np.array([r["updates"] for r in group], dtype=float)
                modes = np.array([r["mode"] for r in group])
                if queries.max() > 0:
                    hot = queries >= np.percentile(queries, 90)
                    hot_total += int(hot.sum())
                    hot_upgraded += int((modes[hot] != 0).sum())
                if updates.max() > 0:
                    upd = (updates >= np.percentile(updates, 90)) & \
                          (queries <= np.median(queries))
                    upd_total += int(upd.sum())
                    upd_scan += int((modes[upd] == 0).sum())
                for r in group:
                    if r["applied"] != r["mode"] and r["applied"] != 0:
                        promotions += 1
                    if r["applied"] != r["mode"]:
                        benefits.append(r["benefit"])
                        deltas.append(-r["delta_cost"])
                budget.append(sum(1 for r in group if r["applied"] != r["mode"] and r["applied"] != 0))
                top = sorted(group, key=lambda r: -r["benefit"])[:4]
                applied_now = [r for r in group if r["applied"] != r["mode"]]
                missed += sum(1 for r in top if r["applied"] == r["mode"] and r["benefit"] > 0)
            metrics.append({
                "seed": seed, "partition": partition,
                "hot_upgrade_ratio": hot_upgraded / hot_total if hot_total else 0.0,
                "update_heavy_scan_ratio": upd_scan / upd_total if upd_total else 0.0,
                "promotions": promotions,
                "short_release": short_release,
                "budget_use": statistics.mean(budget) / 4.0 if budget else 0.0,
                "q_real_spearman": spearman(np.array(benefits), np.array(deltas)),
                "missed_top_candidates": missed,
                "mean_abs_q_spread": float(np.mean([abs(r["q_scan"] - r["q_quad"]) for r in subset])),
            })
    with (args.output / "selection_quality.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    for m in metrics:
        print(f"seed {m['seed']} {m['partition']}: hot_upgrade {m['hot_upgrade_ratio']:.2f} "
              f"upd_scan {m['update_heavy_scan_ratio']:.2f} promotions {m['promotions']} "
              f"budget {m['budget_use']:.2f} spearman {m['q_real_spearman']:.2f} "
              f"missed {m['missed_top_candidates']} q_spread {m['mean_abs_q_spread']:.3f}")


if __name__ == "__main__":
    main()
