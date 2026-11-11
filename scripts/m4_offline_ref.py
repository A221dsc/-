"""Offline decision references.

(1) Exact dynamic programming on a 2x2 regional problem under the calibrated
    cost model, verified against brute-force enumeration on a short horizon.
(2) Full-scale idealized lower bound: each window and region picks its cheapest
    mode with perfect knowledge, ignoring switch costs, cooldown and budgets.
(3) The constrained myopic reference is the existing perfect-forecast greedy
    measured through the real index (results/m4/oracle/oracle_greedy.csv).
"""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

SIDE = 16
MODES = 3
NAMES = ["SCAN", "FINE_GRID", "LOCAL_QUAD"]


def read_table(path):
    table = {}
    with Path(path).open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            fields = line.split(",")
            table[(int(fields[0]), int(fields[1]))] = (
                float(fields[2]), float(fields[3]), float(fields[4]), float(fields[5]))
    windows = max(w for w, _ in table) + 1
    array = np.zeros((windows, SIDE * SIDE, 4))
    for (window, region), value in table.items():
        array[window, region] = value
    return array


def query_update_cost(model, mode, load):
    q, n, u, c = load
    query = np.array([q, q * n, n * c, q * math.sqrt(max(n, 0.0)), c])
    update = np.array([u, u * math.sqrt(max(n, 0.0))])
    return float(query @ np.array(model["query"][mode]) + update @ np.array(model["update"][mode]))


def switch_matrix(model, entity):
    out = np.zeros((MODES, MODES))
    for s in range(MODES):
        for t in range(MODES):
            out[s, t] = float(np.array([1.0, entity]) @
                              np.array(model["switch"][NAMES[s]][NAMES[t]]))
    return out


def exact_dp(loads, regions, wcost, sw):
    windows = wcost.shape[0]
    states = {}
    for modes in itertools.product(range(MODES), repeat=4):
        states[(modes, (0, 0, 0, 0))] = sum(wcost[0, i, modes[i]] for i in range(4))
    for w in range(1, windows):
        nxt = {}
        for (modes, cooldown), value in states.items():
            for target in itertools.product(range(MODES), repeat=4):
                total = value
                legal = True
                for i in range(4):
                    if cooldown[i] and target[i] != modes[i]:
                        legal = False
                        break
                    total += sw[modes[i], target[i]] + wcost[w, i, target[i]]
                if not legal:
                    continue
                state = (target, tuple(1 if target[i] != modes[i] else 0 for i in range(4)))
                if total < nxt.get(state, float("inf")):
                    nxt[state] = total
        states = nxt
    return min(states.values())


def brute_force(loads, regions, wcost, sw, horizon):
    best = float("inf")
    for actions in itertools.product(range(MODES), repeat=4 * horizon):
        total = 0.0
        previous = (0, 0, 0, 0)
        last = [-10] * 4
        legal = True
        for w in range(horizon):
            modes = actions[w * 4:(w + 1) * 4]
            for i in range(4):
                if modes[i] != previous[i]:
                    if w - last[i] <= 1:
                        legal = False
                        break
                    total += sw[previous[i], modes[i]]
                    last[i] = w
                total += wcost[w, i, modes[i]]
            if not legal:
                break
            previous = modes
        if legal:
            best = min(best, total)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("scenarios", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scenario", default="test-2-den011d")
    args = parser.parse_args()
    model = json.loads((args.calibration / "cost_model.json").read_text(encoding="utf-8"))
    loads = read_table(args.scenarios / "loads" / f"{args.scenario}.actual_loads.csv")
    # Hottest 2x2 block over the whole replay, so the instance has real load.
    best_block, best_value = (0, 0), -1.0
    for by in range(SIDE - 1):
        for bx in range(SIDE - 1):
            value = float(sum(loads[w, SIDE * (by + dy) + bx + dx, 0]
                              for w in range(loads.shape[0]) for dy in (0, 1) for dx in (0, 1)))
            if value > best_value:
                best_block, best_value = (bx, by), value
    bx, by = best_block
    regions = [SIDE * by + bx, SIDE * by + bx + 1, SIDE * (by + 1) + bx, SIDE * (by + 1) + bx + 1]
    entity = float(loads[:, regions, 1].max())
    assert entity <= 6000, "entity rebuild budget would bind in the 2x2 instance"
    sw = switch_matrix(model, entity)
    wcost = np.zeros((loads.shape[0], 4, MODES))
    for w in range(loads.shape[0]):
        for i, region in enumerate(regions):
            for mode in range(MODES):
                wcost[w, i, mode] = query_update_cost(model, mode, loads[w, region])
    optimal = exact_dp(loads, regions, wcost, sw)
    short = brute_force(loads, regions, wcost[:2], sw, 2)
    dp_short = exact_dp(loads[:2], regions, wcost[:2], sw)
    assert abs(dp_short - short) < 1e-6, f"DP {dp_short} != brute force {short}"
    idealized = float(sum(min(query_update_cost(model, m, loads[w, r]) for m in range(MODES))
                          for w in range(loads.shape[0]) for r in range(SIDE * SIDE)))
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [
        {"instance": f"hottest 2x2 block at ({bx},{by}); 24 windows", "method": "exact DP",
         "cost": optimal},
        {"instance": "same, 2-window horizon", "method": "exact DP", "cost": dp_short},
        {"instance": "same, 2-window horizon", "method": "brute force", "cost": short},
        {"instance": "full 256 regions, 24 windows", "method": "idealized lower bound",
         "cost": idealized},
    ]
    with (args.output / "offline_reference.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["instance", "method", "cost"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"2x2 exact DP {optimal:.3f}")
    print(f"2-window DP {dp_short:.6f} == brute force {short:.6f}")
    print(f"idealized full-scale lower bound {idealized:.1f}")


if __name__ == "__main__":
    main()
