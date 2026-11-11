"""H1-A: scenario-driven granularity selection.

Features come from the nominal scenario and history statistics available before
the run (never the actual future events of the evaluated trace). Labels are the
measured C++ full-online cost for each candidate granularity. Evaluation is
leave-one-scenario-out over the eight frozen scenarios, three seeds each.
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

rows = list(csv.DictReader(open("results/partition_prediction/dataset/dataset.csv")))
candidates = [0, 1, 4, 8, 16, 32]  # 0 = direct full-map Quad
N = 20000
WORLD = 4096.0
scenarios = sorted({r["scenario"] for r in rows})
seeds = sorted({int(r["seed"]) for r in rows})
index = {(r["scenario"], int(r["seed"]), int(r["cells"])): r for r in rows}


def features(r):
    cells = int(r["cells"])
    if cells == 0:
        return [1.0, float(r["qps"]), float(r["qps"]), float(r["qps"]) * math.sqrt(N),
                float(r["ups"]), float(r["ups"]) * math.sqrt(N), 0.0, 0.0,
                float(r["hot"]) * float(r["hp"]), float(r["migrate"]), float(r["disjoint"]),
                float(r["qbox"]) / 1000.0]
    side = math.sqrt(cells)
    cell = WORLD / side
    qbox = float(r["qbox"])
    qps = float(r["qps"])
    ups = float(r["ups"])
    move = float(r["move"])
    hot = float(r["hot"])
    hp = float(r["hp"])
    migrate = float(r["migrate"])
    disjoint = float(r["disjoint"])
    leaves_touched = (qbox / cell + 1.0) ** 2
    leaf_n = N / cells
    cross = ups * move / cell
    return [1.0, qps, qps * leaves_touched, qps * math.sqrt(leaf_n), ups,
            ups * math.sqrt(leaf_n), cells, cross, hot * hp, migrate, disjoint,
            qbox / 1000.0]


def cost(r):
    return float(r["online_ms"])


def fit(train_rows):
    X = np.array([features(r) for r in train_rows])
    y = np.array([cost(r) for r in train_rows])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def predict(beta, r):
    return float(np.dot(beta, features(r)))


results = []
for held in scenarios:
    train = [r for r in rows if r["scenario"] != held]
    beta = fit(train)
    # fixed best on the training scenarios (median cost per candidate)
    med = {c: float(np.median([cost(index[(s, sd, c)]) for s in scenarios if s != held
                               for sd in seeds])) for c in candidates}
    fixed = min(med, key=med.get)
    for sd in seeds:
        actual = {c: cost(index[(held, sd, c)]) for c in candidates}
        oracle = min(actual, key=actual.get)
        pred = {c: predict(beta, index[(held, sd, c)]) for c in candidates}
        chosen = min(pred, key=pred.get)
        # rule: large queries stay single; small concentrated/migrating hot spots go fine;
        # mobility prefers a middle granularity.
        r = index[(held, sd, 1)]
        qbox, qps, ups = float(r["qbox"]), float(r["qps"]), float(r["ups"])
        move, hot = float(r["move"]), float(r["hot"])
        if qbox >= 500:
            rule = 1
        elif hot > 0 and qbox <= 32:
            rule = 32
        elif ups >= 100:
            rule = 8
        elif float(r["hot"]) > 0:
            rule = 16
        else:
            rule = 1
        results.append({"scenario": held, "seed": sd, "oracle_cells": oracle,
                        "oracle_ms": actual[oracle], "chosen_cells": chosen,
                        "chosen_ms": actual[chosen], "fixed_cells": fixed,
                        "fixed_ms": actual[fixed], "rule_cells": rule,
                        "rule_ms": actual[rule],
                        "model_abs_error": abs(pred[chosen] - actual[chosen])})
with open("results/partition_prediction/h1a_results.csv", "w", newline="",
          encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)
n = len(results)
def acc(key):
    return sum(1 for r in results if r[key] == r["oracle_cells"]) / n
print("rows", n)
print("model accuracy (exact granularity)", round(acc("chosen_cells"), 3))
print("rule accuracy", round(acc("rule_cells"), 3))
print("fixed accuracy", round(acc("fixed_cells"), 3))
for key in ("chosen_ms", "rule_ms", "fixed_ms", "oracle_ms"):
    print(key, round(float(np.mean([r[key] for r in results])), 2))
regret = {k: float(np.mean([(r[k] - r["oracle_ms"]) / r["oracle_ms"] for r in results]))
          for k in ("chosen_ms", "rule_ms", "fixed_ms")}
print("mean relative regret", {k: round(v, 3) for k, v in regret.items()})
summary = {"rows": n, "candidates": candidates,
           "accuracy": {"model": acc("chosen_cells"), "rule": acc("rule_cells"),
                        "fixed": acc("fixed_cells")},
           "mean_ms": {k: float(np.mean([r[k] for r in results]))
                       for k in ("chosen_ms", "rule_ms", "fixed_ms", "oracle_ms")},
           "regret": regret,
           "per_scenario": {s: [r["chosen_cells"] for r in results if r["scenario"] == s]
                            for s in scenarios}}
Path("results/partition_prediction/h1a_summary.json").write_text(
    json.dumps(summary, indent=1) + "\n", encoding="utf-8")
