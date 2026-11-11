"""Risk-measure audit: quantile penalty vs CVaR and sample-size stability.

The cost of each index mode is evaluated on draws from the training-calibrated
residual model. For a fixed set of region-windows we repeat the draw R times at
M in {8, 32, 64, 128} and measure the relative spread of the expected cost and
of the CVaR estimate, the decision-flip rate and the draw time.
"""
import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np

from prepare import json_file

SIDE = 16
REGIONS = SIDE * SIDE
BINS = [(0.0, 0.5), (0.5, 5.0), (5.0, 20.0), (20.0, 100.0), (100.0, float("inf"))]


def read_nominal(path):
    table = {}
    with Path(path).open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            fields = line.split(",")
            table[(int(fields[0]), int(fields[1]))] = (float(fields[2]), float(fields[3]),
                                                       float(fields[4]), float(fields[5]))
    windows = max(w for w, _ in table) + 1
    array = np.zeros((windows, REGIONS, 4))
    for (window, region), value in table.items():
        array[window, region] = value
    return array


def bin_of(value):
    for index, (low, high) in enumerate(BINS):
        if low <= value < high:
            return index
    return len(BINS) - 1


def mode_cost(load, model):
    q, n, u, c = load
    query = np.array([q, q * n, n * c, q * math.sqrt(max(n, 0.0)), c])
    update = np.array([u, u * math.sqrt(max(n, 0.0))])
    return [float(query @ np.array(model["query"][i]) + update @ np.array(model["update"][i]))
            for i in range(3)]


def draw(residual, rng, q, u, c, samples):
    eps_q = rng.normal(0, 1, samples)
    eps_u = rng.normal(0, 1, samples)
    row_q = residual["queries"][bin_of(q)]
    row_u = residual["updates"][bin_of(u)]
    q_s = np.maximum(0.0, (q + 0.5) * np.exp(row_q["mean"] + row_q["std"] * eps_q) - 0.5)
    u_s = np.maximum(0.0, (u + 0.5) * np.exp(row_u["mean"] + row_u["std"] * eps_u) - 0.5)
    c_s = c * (q_s / max(q, 1e-6))
    return q_s, u_s, c_s


def cvar(values, alpha=0.8):
    values = np.sort(values)
    count = max(1, int(math.ceil((1.0 - alpha) * len(values))))
    return float(values[-count:].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scenario", default="test-0-brc100d")
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--sizes", nargs="+", type=int, default=[8, 32, 64, 128])
    args = parser.parse_args()
    model = json.loads((args.calibration / "cost_model.json").read_text(encoding="utf-8"))
    residual = json.loads((args.calibration.parent / "uncertainty" / "residual_model.json")
                          .read_text(encoding="utf-8"))["model"]
    nominal = read_nominal(args.directory / "scenarios" / f"{args.scenario}.nominal.csv")
    rng = np.random.default_rng(20261111)
    windows, regions = nominal.shape[0], nominal.shape[1]
    flat = [(w, r) for w in range(windows) for r in range(regions)]
    picks = rng.choice(len(flat), size=min(args.points, len(flat)), replace=False)
    points = [flat[int(i)] for i in picks]
    rows = []
    for samples in args.sizes:
        expected = np.zeros((args.repeats, len(points), 3))
        tail = np.zeros((args.repeats, len(points), 3))
        draw_time = 0.0
        for rep in range(args.repeats):
            start = time.perf_counter()
            for index, (w, r) in enumerate(points):
                q, n, u, c = nominal[w, r]
                q_s, u_s, c_s = draw(residual, rng, q, u, c, samples)
                costs = np.array([mode_cost((q_s[i], n, u_s[i], c_s[i]), model)
                                  for i in range(samples)])
                expected[rep, index, :] = costs.mean(axis=0)
                tail[rep, index, :] = [cvar(costs[:, mode]) for mode in range(3)] - \
                                      costs.mean(axis=0)
            draw_time += time.perf_counter() - start
        cost_mean = expected.mean(axis=0)
        rel_expected = float(np.mean(expected.std(axis=0) / np.maximum(cost_mean, 1e-9)))
        rel_tail = float(np.mean(tail.std(axis=0) / np.maximum(np.abs(tail).mean(axis=0), 1e-9)))
        chosen = expected.argmin(axis=2)
        flip = 0
        for index in range(len(points)):
            values = set(chosen[:, index].tolist())
            flip += len(values) > 1
        risk_scores = expected + 0.5 * tail
        risk_chosen = risk_scores.argmin(axis=2)
        risk_flip = sum(1 for index in range(len(points)) if len(set(risk_chosen[:, index].tolist())) > 1)
        rows.append({"samples": samples, "repeats": args.repeats, "points": len(points),
                     "relative_std_expected_cost": rel_expected,
                     "relative_std_tail_component": rel_tail,
                     "decision_flip_rate_expected": flip / len(points),
                     "decision_flip_rate_risk": risk_flip / len(points),
                     "draw_seconds_per_repeat": draw_time / args.repeats})
        print(f"M={samples}: rel_std(mean)={rel_expected:.4f} rel_std(tail)={rel_tail:.3f} "
              f"flip={flip / len(points):.3f} risk_flip={risk_flip / len(points):.3f} "
              f"draw={draw_time / args.repeats:.3f}s", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "sample_size_stability.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_file(args.output / "sample_size_stability.json",
              {"scenario": args.scenario, "rows": rows,
               "risk_definition": "CVaR 0.8 of per-mode cost minus its mean; "
                                  "score = mean + 0.5 * (CVaR - mean)"})


if __name__ == "__main__":
    main()
