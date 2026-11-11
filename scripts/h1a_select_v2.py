"""H1-A.1 corrected granularity selection.

Geometry features use the per-axis count k: leaves=k^2, leaf side=WORLD/k,
entities per leaf=N/k^2, touched leaves=(qbox/leaf_side+1)^2, cross-leaf moves
~ ups*move/leaf_side. The direct full-map Quad is the k=1 limit (one leaf), so
all candidates share one physical feature definition.

The cost model is a standardized ridge regression, fit both on the raw cost and
on log(cost); predictions are reported for every candidate, not only the chosen
one. Training is the original eight scenarios; the test is the twelve frozen
held-out combos, grouped by scenario parameter combination.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np

WORLD = 4096.0
N = 20000
ORIG = Path("results/partition_prediction/dataset/dataset_v2_orig.csv")
NEW = Path("results/partition_prediction/dataset/dataset_v2_new.csv")


def load(path):
    rows = list(csv.DictReader(open(path)))
    out = {}
    for r in rows:
        key = (r["scenario"], int(r["seed"]))
        out.setdefault(key, {})[int(r["candidate"])] = r
    return out


def geometry(cells):
    k = cells if cells > 0 else 1
    leaves = k * k
    leaf_side = WORLD / k
    leaf_n = N / leaves
    return k, leaves, leaf_side, leaf_n


def features(r, cells):
    k, leaves, leaf_side, leaf_n = geometry(cells)
    qbox = float(r["qbox"])
    qps = float(r["qps"])
    ups = float(r["ups"])
    move = float(r["move"])
    hot = float(r["hot"])
    hp = float(r["hp"])
    migrate = float(r["migrate"])
    disjoint = float(r["disjoint"])
    mix = float(r.get("density_mix", 0.8))
    touched = (qbox / leaf_side + 1.0) ** 2
    cross = ups * move / leaf_side
    return [1.0, qps, qps * touched, qps * math.sqrt(leaf_n), ups,
            ups * math.sqrt(leaf_n), leaves, cross, hot * hp / 1e6, migrate, disjoint,
            qbox / 1000.0, mix, k / 32.0]


def fit(train_rows, log_target, lam=1.0):
    X = np.array([features(r, c) for (r, c) in train_rows])
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Z = (X - mu) / sigma
    y = np.array([float(r["online_ms"]) for (r, _) in train_rows])
    if log_target:
        y = np.log(y)
    n = Z.shape[1]
    beta = np.linalg.solve(Z.T @ Z + lam * np.eye(n), Z.T @ y)
    return {"mu": mu, "sigma": sigma, "beta": beta, "log": log_target}


def predict(model, r, cells):
    x = np.array(features(r, cells))
    z = (x - model["mu"]) / model["sigma"]
    v = float(np.dot(model["beta"], z))
    return math.exp(v) if model["log"] else v


def rule(r):
    qbox, qps, ups = float(r["qbox"]), float(r["qps"]), float(r["ups"])
    move, hot = float(r["move"]), float(r["hot"])
    if qbox >= 500:
        return 0
    if hot > 0 and qbox <= 32:
        return 32
    if ups >= 100:
        return 8
    return 0


def binary_rule(r):
    return 32 if float(r["qbox"]) <= 50 else 0


orig = load(ORIG)
new = load(NEW)
candidates = [0, 1, 4, 8, 16, 32]
train_rows = [(r, c) for key in orig for c, r in orig[key].items() if c in candidates]
models = {}
for log_target in (False, True):
    models["log" if log_target else "linear"] = fit(train_rows, log_target)
# fixed best on the training scenarios
med = {c: float(np.median([float(orig[k][c]["online_ms"]) for k in orig])) for c in candidates}
fixed = min(med, key=med.get)
rows = []
for key in sorted(new):
    for c in candidates:
        r = new[key][c]
        pred = {m: predict(models[m], r, c) for m in models}
        rows.append({"scenario": key[0], "seed": key[1], "candidate": c,
                     "actual_ms": float(r["online_ms"]),
                     "pred_linear": pred["linear"], "pred_log": pred["log"],
                     "fixed": int(c == fixed)})
with open("results/partition_prediction/h1a_v2_predictions.csv", "w", newline="",
          encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def evaluate(pred_key, chooser):
    sel = {}
    for key in new:
        actual = {c: float(new[key][c]["online_ms"]) for c in candidates}
        oracle = min(actual, key=actual.get)
        chosen = chooser(key)
        sel[key] = (oracle, chosen, actual[chosen])
    acc = sum(1 for k, (o, c, _) in sel.items() if o == c) / len(sel)
    cost = float(np.mean([v[2] for v in sel.values()]))
    oracle_cost = float(np.mean([float(new[k][v[0]]["online_ms"]) for k, v in sel.items()]))
    regret = float(np.mean([(v[2] - float(new[k][v[0]]["online_ms"])) /
                            float(new[k][v[0]]["online_ms"]) for k, v in sel.items()]))
    return {"accuracy": acc, "mean_cost_ms": cost, "oracle_ms": oracle_cost,
            "regret": regret, "choices": {k[0]: v[1] for k, v in sel.items()}}


def chooser_model(name):
    def pick(key):
        return min(candidates, key=lambda c: predict(models[name], new[key][c], c))
    return pick


def chooser_rule(key):
    return rule(new[key][0])


def chooser_binary(key):
    return binary_rule(new[key][0])


def chooser_fixed(key):
    return fixed


def chooser_direct(key):
    return 0


scenarios_new = sorted({k[0] for k in new})
# group by scenario: one aggregated result per held-out scenario (median over seeds)
grouped = {}
for key in new:
    grouped.setdefault(key[0], []).append(key)
summary = {}
for name, pick in (("linear", chooser_model("linear")), ("log", chooser_model("log")),
                   ("rule", chooser_rule), ("binary", chooser_binary),
                   ("fixed", chooser_fixed), ("direct", chooser_direct)):
    per_scenario = []
    for s in scenarios_new:
        keys = grouped[s]
        actual = {c: float(np.median([float(new[k][c]["online_ms"]) for k in keys]))
                  for c in candidates}
        oracle = min(actual, key=actual.get)
        chosen = pick(keys[0])
        per_scenario.append({"scenario": s, "oracle": oracle, "chosen": chosen,
                             "cost": actual[chosen], "oracle_cost": actual[oracle]})
    acc = sum(1 for r in per_scenario if r["oracle"] == r["chosen"]) / len(per_scenario)
    cost = float(np.mean([r["cost"] for r in per_scenario]))
    oracle_cost = float(np.mean([r["oracle_cost"] for r in per_scenario]))
    regret = float(np.mean([(r["cost"] - r["oracle_cost"]) / r["oracle_cost"]
                            for r in per_scenario]))
    summary[name] = {"accuracy": acc, "mean_cost_ms": cost, "oracle_ms": oracle_cost,
                     "regret": regret, "per_scenario": per_scenario}
# prediction error of all candidates
for name in ("linear", "log"):
    err = np.array([abs(predict(models[name], new[k][c], c) - float(new[k][c]["online_ms"]))
                    for k in new for c in candidates])
    actual = np.array([float(new[k][c]["online_ms"]) for k in new for c in candidates])
    summary[f"error_{name}"] = {"mae_ms": float(err.mean()),
                                "rmse_ms": float(np.sqrt((err ** 2).mean())),
                                "mean_relative": float(np.mean(err / actual)),
                                "max_ms": float(err.max())}
print(json.dumps(summary, indent=1, default=str)[:3500])
Path("results/partition_prediction/h1a_v2_summary.json").write_text(
    json.dumps(summary, indent=1) + "\n", encoding="utf-8")
