"""Analyse the M3 fair comparison: costs, decision quality, payback and bounds.

All inputs are the raw CSVs written by run_m3.py. Per-window profile costs are
rescaled so that their total matches the primary measured online time of the same
method and seed; this preserves the temporal shape without mixing two timing
paths. The offline oracle bounds are analysis-only references and are never used
by the online policies.
"""
import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prepare import json_file

METHODS = ["S0_ALL_SCAN", "S1_ALL_FINE", "S2_ALL_QUAD", "S3_THRESHOLD", "S4_GREEDY",
           "UNIFORM_GRID", "PR_QUADTREE", "RSTAR_BOOST"]
LABELS = {"S0_ALL_SCAN": "S0 SCAN", "S1_ALL_FINE": "S1 FINE_GRID", "S2_ALL_QUAD": "S2 LOCAL_QUAD",
          "S3_THRESHOLD": "S3 threshold", "S4_GREEDY": "S4 greedy",
          "UNIFORM_GRID": "Uniform Grid", "PR_QUADTREE": "PR Quadtree", "RSTAR_BOOST": "Boost R*"}
MODES = ["SCAN", "FINE_GRID", "LOCAL_QUAD"]
MODE_FIXED = {"S0_ALL_SCAN": "SCAN", "S1_ALL_FINE": "FINE_GRID", "S2_ALL_QUAD": "LOCAL_QUAD"}
PHASES = ["fixed", "migration", "alternation", "update_surge"]
PHASE_LABELS = {"fixed": "fixed hotspot", "migration": "hotspot migration",
                "alternation": "hot/cold alternation", "update_surge": "update surge"}
PALETTE = ["#22577A", "#278F85", "#B66233", "#7D5BA6", "#C8553D", "#5B8E7D", "#8D6E63", "#455A64"]


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def phase_of(window, windows_per_phase):
    return PHASES[min(window // windows_per_phase, len(PHASES) - 1)]


def rescale_profiles(regions, timeline, metrics):
    """Return per-window totals whose sum equals the measured online time."""
    measured = {(r["method"], int(r["seed"])): float(r["online_ms"]) for r in metrics}
    per_window = defaultdict(lambda: defaultdict(float))
    for row in regions:
        key = (row["method"], int(row["seed"]))
        window = int(row["window_index"])
        per_window[key][window] += (float(row["query_ms"]) + float(row["local_update_ms"])
                                    + float(row["switch_ms"]))
    policy = defaultdict(dict)
    for row in timeline:
        policy[(row["method"], int(row["seed"]))][int(row["window_index"])] = float(row["policy_ms"])
    scaled, scales = {}, {}
    for key, windows in per_window.items():
        raw = sum(windows.values()) + sum(policy[key].values())
        target = measured[key]
        scale = target / raw if raw > 0 else 1.0
        scales[key] = scale
        scaled[key] = {w: (windows.get(w, 0.0) + policy[key].get(w, 0.0)) * scale for w in range(24)}
    return scaled, scales


def hotspot_tracking(regions, schedules, methods, seeds, windows):
    """For every window, is the true hotspot region indexed by the policy?"""
    indexed = {}
    for row in regions:
        if row["method"] in methods:
            key = (row["method"], int(row["seed"]), int(row["window_index"]))
            indexed.setdefault(key, {})[int(row["region_id"])] = row["mode"] != "SCAN"
    tracking = []
    for method in methods:
        for seed in seeds:
            schedule = schedules[str(seed)]
            cells = []
            for window in range(windows):
                step = min(len(schedule) - 1, 50 * (window + 1) - 1)
                entry = schedule[step]
                cells.append((int(entry[2] + 0.5), int(entry[3] + 0.5)))
            for window, (x, y) in enumerate(cells):
                region = 16 * y + x
                modes = indexed.get((method, seed, window), {})
                tracking.append({"method": method, "seed": seed, "window": window,
                                 "phase": phase_of(window, 6), "hotspot_region": region,
                                 "indexed": bool(modes.get(region, False)),
                                 "indexed_regions": sum(1 for v in modes.values() if v),
                                 "changed": window > 0 and cells[window] != cells[window - 1]})
    return tracking


def detection_delays(tracking):
    by_key = defaultdict(list)
    for row in tracking:
        if row["changed"] and row["window"] > 0:
            by_key[(row["method"], row["seed"])].append(row)
    delays = []
    for (method, seed), changes in by_key.items():
        rows = {(r["window"]): r for r in tracking if r["method"] == method and r["seed"] == seed}
        for change in changes:
            start = change["window"]
            target = change["hotspot_region"]
            delay = None
            for window in range(start, 24):
                row = rows.get(window)
                if row and row["hotspot_region"] == target and row["indexed"]:
                    delay = window - start
                    break
            delays.append({"method": method, "seed": seed, "change_window": start,
                           "hotspot_region": target, "delay_windows": delay})
    return delays


def build_counterfactual(regions):
    """Per region-window cost of each fixed mode, from the three profile replays."""
    table = defaultdict(dict)
    for row in regions:
        if row["method"] not in MODE_FIXED:
            continue
        key = (int(row["seed"]), int(row["window_index"]), int(row["region_id"]))
        table[key][MODE_FIXED[row["method"]]] = (float(row["query_ms"])
                                                 + float(row["local_update_ms"]))
    return table


def payback(decisions, regions, seeds, windows_per_phase=6, windows=24):
    counterfactual = defaultdict(dict)
    for row in regions:
        if row["method"] not in MODE_FIXED:
            continue
        key = (int(row["seed"]), int(row["window_index"]), int(row["region_id"]))
        counterfactual[key][MODE_FIXED[row["method"]]] = (float(row["query_ms"])
                                                          + float(row["local_update_ms"]))
    governed = {}
    for row in regions:
        if row["method"] in ("S3_THRESHOLD", "S4_GREEDY"):
            key = (row["method"], int(row["seed"]), int(row["window_index"]), int(row["region_id"]))
            governed[key] = {"queries": int(row["query_count"]),
                             "cost": float(row["query_ms"]) + float(row["local_update_ms"])}
    rows = []
    for row in decisions:
        if row["status"] != "applied":
            continue
        seed, window = int(row["seed"]), int(row["window_index"])
        if window >= windows:
            continue
        key = (seed, window, int(row["region_id"]))
        costs = counterfactual.get(key, {})
        old, new = row["old_mode"], row["selected_mode"]
        switch_cost = float(row["switch_cost_ms"])
        saving = None
        if old in costs and new in costs:
            saving = costs[old] - costs[new]
        realised = governed.get((row["method"], seed, window, int(row["region_id"])), {}).get("queries")
        remaining = windows_per_phase - window % windows_per_phase
        kind = ("demote" if new == "SCAN" else
                "promote" if old == "SCAN" else "retune")
        rows.append({
            "method": row["method"], "seed": seed, "step": int(row["step"]), "window": window,
            "phase": phase_of(window, windows_per_phase), "region_id": int(row["region_id"]),
            "kind": kind,
            "entities": int(row["entity_count"]), "old_mode": old, "selected_mode": new,
            "switch_cost_ms": switch_cost,
            "counterfactual_saving_ms": saving,
            "payback_windows": (switch_cost / saving) if saving and saving > 0 else "",
            "windows_left_in_phase": remaining,
            "pays_back_in_phase": (switch_cost / saving <= remaining) if saving and saving > 0 else False,
            "governed_queries": realised,
            "wasted": kind != "demote" and realised is not None and realised < 5,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    metrics = read_csv(args.run / "metrics.csv")
    regions = read_csv(args.run / "regions.csv")
    timeline = read_csv(args.run / "timeline.csv")
    decisions = read_csv(args.run / "decisions.csv")
    schedules = json.loads((args.run / "trace_schedule.json").read_text(encoding="utf-8"))
    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    selection = json.loads((args.run / "selection.json").read_text(encoding="utf-8"))
    summary = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
    seeds = [int(s) for s in config["test_seeds"]]
    windows = config["steps"] // config["window_steps"]
    windows_per_phase = config["phase_steps"] // config["window_steps"]

    # ---- overall table ------------------------------------------------------
    by_method = defaultdict(dict)
    for row in metrics:
        by_method[row["method"]][int(row["seed"])] = row
    overall = []
    for method in METHODS:
        rows = by_method[method]
        median = lambda field: statistics.median(float(rows[s][field]) for s in seeds)
        overall.append({
            "method": method, "label": LABELS[method],
            "build_ms": median("build_ms"), "query_ms": median("query_ms"),
            "update_ms": median("update_ms"), "switch_ms": median("switch_ms"),
            "policy_ms": median("policy_ms"), "online_ms": median("online_ms"),
            "total_ms": median("total_ms"), "p50_us": median("p50_us"),
            "p95_us": median("p95_us"), "p99_us": median("p99_us"),
            "peak_rss_mb": median("peak_rss_mb"),
            "result_count": median("result_count"), "candidate_count": median("candidate_count"),
            "switches": median("switches"), "promotions": median("promotions"),
            "demotions": median("demotions"), "retunes": median("retunes"),
            "blocked": median("blocked"),
            "final_indexed_regions": statistics.median(
                int(rows[s]["final_fine"]) + int(rows[s]["final_quad"]) for s in seeds),
            "final_indexed_entities": median("final_indexed_entities"),
            "median_ratio_to_best": summary["paired_intervals"][method]["median_ratio_to_best_method"],
            "bootstrap95_ratio_to_best":
                ";".join(f"{v:.4f}" for v in
                         summary["paired_intervals"][method]["bootstrap95_ratio_to_best_method"]),
            "median_ratio_to_best_static_regional":
                summary["paired_intervals"][method]["median_ratio_to_best_static_regional"],
            "bootstrap95_ratio_to_best_static_regional":
                ";".join(f"{v:.4f}" for v in
                         summary["paired_intervals"][method]["bootstrap95_ratio_to_best_static_regional"]),
        })
    write_csv(args.output / "table_overall.csv", overall)

    # ---- profile-rescaled per-window costs ---------------------------------
    scaled, scales = rescale_profiles(regions, timeline, metrics)
    windows_index = list(range(windows))
    phase_rows = []
    for method in METHODS:
        if method not in ("S0_ALL_SCAN", "S1_ALL_FINE", "S2_ALL_QUAD", "S3_THRESHOLD", "S4_GREEDY"):
            continue
        for phase_index, phase in enumerate(PHASES):
            values = [statistics.median(scaled[(method, s)][w]
                                        for s in seeds)
                      for w in windows_index
                      if phase_of(w, windows_per_phase) == phase]
            phase_rows.append({"method": method, "phase": phase,
                               "windows": len(values),
                               "median_cost_ms": statistics.median(values),
                               "sum_cost_ms": sum(values)})
    write_csv(args.output / "table_phases.csv", phase_rows)

    # ---- mode distribution and switching per phase -------------------------
    mode_rows = []
    for method in ("S3_THRESHOLD", "S4_GREEDY"):
        for phase_index, phase in enumerate(PHASES):
            entries = [row for row in timeline if row["method"] == method
                       and phase_of(int(row["window_index"]), windows_per_phase) == phase]
            mode_rows.append({
                "method": method, "phase": phase,
                "windows": len(entries) // max(1, len(seeds)),
                "scan_regions": statistics.median(int(r["scan_regions"]) for r in entries),
                "fine_regions": statistics.median(int(r["fine_regions"]) for r in entries),
                "quad_regions": statistics.median(int(r["quad_regions"]) for r in entries),
                "indexed_regions": statistics.median(int(r["fine_regions"]) + int(r["quad_regions"])
                                                     for r in entries),
                "switches": sum(int(r["switches"]) for r in entries),
                "policy_ms": sum(float(r["policy_ms"]) for r in entries),
            })
    write_csv(args.output / "table_mode_distribution.csv", mode_rows)

    # ---- decision quality ---------------------------------------------------
    tracking = hotspot_tracking(regions, schedules,
                                ["S3_THRESHOLD", "S4_GREEDY"],
                                seeds, windows)
    delays = detection_delays(tracking)
    quality = []
    for method in ("S3_THRESHOLD", "S4_GREEDY"):
        for phase in PHASES:
            entries = [row for row in tracking if row["method"] == method and row["phase"] == phase]
            quality.append({
                "method": method, "phase": phase, "windows": len(entries),
                "hotspot_indexed_share": sum(1 for r in entries if r["indexed"]) / len(entries),
                "median_indexed_regions": statistics.median(r["indexed_regions"] for r in entries),
            })
    write_csv(args.output / "table_decision_quality.csv", quality)
    write_csv(args.output / "table_detection_delay.csv", delays)

    # ---- payback ------------------------------------------------------------
    payback_rows = payback(decisions, regions, seeds, windows_per_phase, windows)
    write_csv(args.output / "payback.csv", payback_rows)
    applied = [r for r in payback_rows if r["counterfactual_saving_ms"] is not None]
    payback_summary = {}
    for method in ("S3_THRESHOLD", "S4_GREEDY"):
        rows = [r for r in applied if r["method"] == method]
        entry = {}
        for kind in ("demote", "promote", "retune"):
            subset = [r for r in rows if r["kind"] == kind]
            with_saving = [r for r in subset if r["counterfactual_saving_ms"] > 0]
            values = [r["payback_windows"] for r in with_saving if r["payback_windows"] != ""]
            entry[kind] = {
                "switches": len(subset),
                "share_with_positive_saving": len(with_saving) / len(subset) if subset else None,
                "median_payback_windows": statistics.median(values) if values else None,
                "share_payback_within_three_windows": (
                    sum(1 for v in values if v <= 3) / len(values)) if values else None,
                "share_pays_back_in_phase": (sum(1 for r in with_saving if r["pays_back_in_phase"])
                                             / len(with_saving)) if with_saving else None,
                "total_switch_cost_ms": sum(r["switch_cost_ms"] for r in subset),
                "wasted_switches": sum(1 for r in subset if r["wasted"]),
            }
        entry["all"] = {"applied_switches": len(rows),
                        "total_switch_cost_ms": sum(r["switch_cost_ms"] for r in rows)}
        payback_summary[method] = entry
    json_file(args.output / "payback_summary.json", payback_summary)

    # ---- decision optimality against the fixed-mode counterfactual ----------
    counterfactual = build_counterfactual(regions)
    policy_modes = {}
    for row in regions:
        if row["method"] in ("S3_THRESHOLD", "S4_GREEDY"):
            policy_modes[(row["method"], int(row["seed"]), int(row["window_index"]),
                          int(row["region_id"]))] = row["mode"]
    optimality, confusion = [], []
    for method in ("S3_THRESHOLD", "S4_GREEDY"):
        for phase in PHASES:
            optimal = 0
            total = 0
            regret = 0.0
            for key, costs in counterfactual.items():
                if len(costs) != 3:
                    continue
                seed, window, region = key
                if phase_of(window, windows_per_phase) != phase:
                    continue
                chosen_mode = policy_modes.get((method, seed, window, region))
                if chosen_mode is None:
                    continue
                best = min(costs.values())
                chosen = costs[chosen_mode]
                total += 1
                regret += chosen - best
                if chosen - best <= max(0.05 * best, 1e-6):
                    optimal += 1
            optimality.append({"method": method, "phase": phase, "region_windows": total,
                               "optimal_share": optimal / total if total else None,
                               "total_regret_ms": regret,
                               "median_regret_us": (regret / total * 1000) if total else None})
        counts = Counter()
        for key, costs in counterfactual.items():
            if len(costs) != 3:
                continue
            seed, window, region = key
            chosen_mode = policy_modes.get((method, seed, window, region))
            if chosen_mode is None:
                continue
            best_mode = min(costs, key=costs.get)
            counts[(best_mode, chosen_mode)] += 1
        for (best_mode, chosen_mode), count in sorted(counts.items()):
            confusion.append({"method": method, "optimal_mode": best_mode,
                              "chosen_mode": chosen_mode, "region_windows": count})
    write_csv(args.output / "table_decision_optimality.csv", optimality)
    write_csv(args.output / "table_mode_confusion.csv", confusion)

    # ---- oscillation --------------------------------------------------------
    oscillation = {}
    for method in ("S3_THRESHOLD", "S4_GREEDY"):
        per_region = defaultdict(list)
        for row in decisions:
            if row["method"] != method or row["status"] != "applied":
                continue
            per_region[(int(row["seed"]), int(row["region_id"]))].append(
                (int(row["window_index"]), row["old_mode"], row["selected_mode"]))
        multi = {k: v for k, v in per_region.items() if len(v) >= 3}
        flip_back = 0
        for changes in per_region.values():
            for i in range(1, len(changes)):
                if changes[i][2] == changes[i - 1][1] and changes[i][0] - changes[i - 1][0] <= 3:
                    flip_back += 1
        oscillation[method] = {
            "regions_switched": len(per_region),
            "regions_with_three_or_more_switches": len(multi),
            "max_switches_one_region": max((len(v) for v in per_region.values()), default=0),
            "switches_back_to_previous_mode_within_three_windows": flip_back,
        }
    json_file(args.output / "oscillation.json", oscillation)

    # ---- offline oracle bounds ---------------------------------------------
    counterfactual = defaultdict(dict)
    for row in regions:
        if row["method"] not in MODE_FIXED:
            continue
        key = (int(row["seed"]), int(row["window_index"]), int(row["region_id"]))
        counterfactual[key][MODE_FIXED[row["method"]]] = (float(row["query_ms"])
                                                          + float(row["local_update_ms"]))
    static_totals = defaultdict(float)
    oracle_window = defaultdict(float)
    by_region = defaultdict(lambda: defaultdict(float))
    for key, costs in counterfactual.items():
        if len(costs) != 3:
            continue
        seed, window, region = key
        for mode, value in costs.items():
            static_totals[(seed, mode)] += value
            by_region[(seed, region)][mode] += value
        oracle_window[seed] += min(costs.values())
    oracle_static_region = defaultdict(float)
    for (seed, region), costs in by_region.items():
        oracle_static_region[seed] += min(costs.values())
    median = lambda data: statistics.median(data[s] for s in seeds)
    profile_sums = defaultdict(float)
    for row in regions:
        method = row["method"]
        if method in ("S0_ALL_SCAN", "S1_ALL_FINE", "S2_ALL_QUAD", "S3_THRESHOLD", "S4_GREEDY"):
            key = (method, int(row["seed"]))
            profile_sums[key] = profile_sums.get(key, 0.0) + (float(row["query_ms"])
                                                             + float(row["local_update_ms"])
                                                             + float(row["switch_ms"]))
    for row in timeline:
        if row["method"] in ("S0_ALL_SCAN", "S1_ALL_FINE", "S2_ALL_QUAD", "S3_THRESHOLD", "S4_GREEDY"):
            key = (row["method"], int(row["seed"]))
            profile_sums[key] = profile_sums.get(key, 0.0) + float(row["policy_ms"])
    profile_totals = {method: statistics.median(profile_sums[(method, s)] for s in seeds)
                      for method in ("S0_ALL_SCAN", "S1_ALL_FINE", "S2_ALL_QUAD",
                                     "S3_THRESHOLD", "S4_GREEDY")}
    best_static = min(profile_totals["S0_ALL_SCAN"], profile_totals["S1_ALL_FINE"],
                      profile_totals["S2_ALL_QUAD"])
    oracle_bound = median(oracle_window)
    static_region_bound = median(oracle_static_region)
    captured = {}
    for method in ("S3_THRESHOLD", "S4_GREEDY"):
        value = profile_totals[method]
        captured[method] = {
            "profile_total_ms": value,
            "gap_to_best_static_ms": best_static - value,
            "share_of_window_oracle_gap_closed": (
                (best_static - value) / (best_static - oracle_bound)) if best_static > oracle_bound else None,
            "share_of_region_static_oracle_gap_closed": (
                (best_static - value) / (best_static - static_region_bound))
            if best_static > static_region_bound else None,
        }
    json_file(args.output / "oracle.json", {
        "profile_totals_ms": profile_totals,
        "best_static_regional_profile_ms": best_static,
        "per_window_oracle_ms": oracle_bound,
        "per_region_static_oracle_ms": static_region_bound,
        "policy_capture": captured,
        "space": "all bounds use the profile per-region costs of the same three fixed-mode replays; "
                 "the coarse update term is absent from every value in this block",
        "note": "offline bounds from the three fixed-mode profile replays of the same test traces; "
                "analysis only, never used by S3/S4",
    })

    # ---- plots --------------------------------------------------------------
    plot_overall(overall, args.output / "fig_online_cost.png")
    plot_cumulative(scaled, seeds, windows, windows_per_phase, args.output / "fig_cumulative.png")
    plot_modes(timeline, seeds, windows, windows_per_phase, args.output / "fig_modes.png")
    plot_optimality(regions, counterfactual, windows, windows_per_phase,
                    args.output / "fig_decision_quality.png")
    plot_payback([r for r in payback_rows if r["kind"] != "demote"],
                 args.output / "fig_payback.png")
    plot_fine_bins(args.calibration / "fine_grid_bins.csv", args.output / "fig_fine_grid_bins.png")

    print("overall best:", min(overall, key=lambda r: r["online_ms"])["method"])
    print("capture:", json.dumps(captured, indent=2))
    print("optimality:", json.dumps(optimality, indent=1))
    print("payback:", json.dumps(payback_summary, indent=2))
    print("oscillation:", json.dumps(oscillation, indent=2))


def plot_optimality(regions, counterfactual, windows, windows_per_phase, path):
    policy_modes = {}
    for row in regions:
        if row["method"] in ("S3_THRESHOLD", "S4_GREEDY"):
            policy_modes[(row["method"], int(row["seed"]), int(row["window_index"]),
                          int(row["region_id"]))] = row["mode"]
    fig, (share, regret) = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, layout="constrained")
    for index, method in enumerate(("S3_THRESHOLD", "S4_GREEDY")):
        shares, regrets = [], []
        for window in range(windows):
            hits = total = 0
            lost = 0.0
            for key, costs in counterfactual.items():
                seed, w, region = key
                if w != window or len(costs) != 3:
                    continue
                chosen_mode = policy_modes.get((method, seed, window, region))
                if chosen_mode is None:
                    continue
                best = min(costs.values())
                total += 1
                lost += costs[chosen_mode] - best
                hits += costs[chosen_mode] - best <= max(0.05 * best, 1e-6)
            shares.append(hits / total if total else 0)
            regrets.append(lost)
        share.plot(range(1, windows + 1), shares, marker="o", ms=2.6, lw=1.8,
                   color=PALETTE[index], label=LABELS[method])
        regret.plot(range(1, windows + 1), regrets, marker="o", ms=2.6, lw=1.8,
                    color=PALETTE[index], label=LABELS[method])
    for ax in (share, regret):
        for boundary in range(windows_per_phase, windows, windows_per_phase):
            ax.axvline(boundary + .5, color="#999999", lw=.8, ls="--")
        ax.grid(alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["right", "top"]].set_visible(False)
    share.set_ylabel("share of region-windows within 5%\nof the offline mode optimum")
    share.set_ylim(-.02, 1.02)
    share.set_title("Decision quality against the fixed-mode counterfactual", loc="left")
    share.legend(frameon=False, ncol=2, loc="lower left", fontsize=8)
    regret.set_ylabel("regret per window (ms)")
    regret.set_xlabel("decision window")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def bar_stacked(ax, labels, series, palette):
    bottom = [0.0] * len(labels)
    for index, (name, values) in enumerate(series):
        ax.bar(labels, values, bottom=bottom, label=name, color=palette[index], width=.72)
        bottom = [b + v for b, v in zip(bottom, values)]
    return bottom


def plot_overall(overall, path):
    fig, (ax, ratio) = plt.subplots(2, 1, figsize=(11, 7.6), layout="constrained",
                                    gridspec_kw={"height_ratios": [3, 1.5]})
    labels = [row["label"] for row in overall]
    bar_stacked(ax, labels,
                [("query", [row["query_ms"] for row in overall]),
                 ("update", [row["update_ms"] for row in overall]),
                 ("switch + policy", [row["switch_ms"] + row["policy_ms"] for row in overall])],
                ["#22577A", "#B66233", "#7D5BA6"])
    ax.set_ylabel("online time (ms)")
    ax.set_ylim(0, 460)
    ax.set_title("M3 test traces: online cost decomposition, median of five paired seeds", loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.annotate("Boost R* 1,043 ms (clipped)", xy=(7, 455), xytext=(6.2, 400),
                fontsize=8, color="#666666", ha="center",
                arrowprops={"arrowstyle": "-", "color": "#999999", "lw": .8})
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[["right", "top"]].set_visible(False)

    for index, row in enumerate(overall):
        low, high = (float(v) for v in row["bootstrap95_ratio_to_best"].split(";"))
        ratio.plot([low, high], [index, index], color="#22577A", lw=1.6)
        ratio.plot(row["median_ratio_to_best"], index, "o", color="#B66233", ms=5)
        ratio.text(high * 1.05, index, f"{row['median_ratio_to_best']:.2f}", va="center", fontsize=7)
    ratio.axvline(1.0, color="#444444", lw=1, ls="--")
    ratio.set_xscale("log")
    ratio.set_yticks(range(len(overall)), labels)
    ratio.invert_yaxis()
    ratio.set_xlabel("paired online-time ratio to the best method (median, bootstrap 95% interval)")
    ratio.grid(axis="x", alpha=.2)
    ratio.set_axisbelow(True)
    ratio.spines[["right", "top"]].set_visible(False)
    plt.setp(ratio.get_xticklabels(), fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_cumulative(scaled, seeds, windows, windows_per_phase, path):
    fig, ax = plt.subplots(figsize=(11, 5.4), layout="constrained")
    methods = ["S0_ALL_SCAN", "S1_ALL_FINE", "S2_ALL_QUAD", "S3_THRESHOLD", "S4_GREEDY"]
    for index, method in enumerate(methods):
        cumulative = []
        total = 0.0
        for window in range(windows):
            total += statistics.median(scaled[(method, s)][window] for s in seeds)
            cumulative.append(total)
        ax.plot(range(1, windows + 1), cumulative, marker="o", ms=2.6, lw=1.8,
                color=PALETTE[index], label=LABELS[method])
    for boundary in range(windows_per_phase, windows, windows_per_phase):
        ax.axvline(boundary + .5, color="#999999", lw=.8, ls="--")
    for index, phase in enumerate(PHASES):
        ax.text(index * windows_per_phase + windows_per_phase / 2, ax.get_ylim()[1] * .98,
                PHASE_LABELS[phase], ha="center", va="top", fontsize=8, color="#555555")
    ax.set_xlabel("decision window (50 steps each)")
    ax.set_ylabel("cumulative online cost (ms)")
    ax.set_title("Cumulative cost along the 1,200-step trace; profile shape rescaled to the measured total",
                 loc="left")
    ax.legend(frameon=False, ncol=3)
    ax.grid(alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_modes(timeline, seeds, windows, windows_per_phase, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), sharey=True, layout="constrained")
    for ax, method in zip(axes, ("S3_THRESHOLD", "S4_GREEDY")):
        fine, quad = [], []
        for window in range(windows):
            entries = [row for row in timeline if row["method"] == method
                       and int(row["window_index"]) == window]
            fine.append(statistics.median(int(r["fine_regions"]) for r in entries) if entries else 0)
            quad.append(statistics.median(int(r["quad_regions"]) for r in entries) if entries else 0)
        positions = list(range(1, windows + 1))
        ax.bar(positions, fine, color="#278F85", alpha=.85, label="FINE_GRID")
        ax.bar(positions, quad, bottom=fine, color="#22577A", alpha=.85, label="LOCAL_QUAD")
        for boundary in range(windows_per_phase, windows, windows_per_phase):
            ax.axvline(boundary + .5, color="#999999", lw=.8, ls="--")
        median_indexed = statistics.median([f + q for f, q in zip(fine, quad)])
        ax.set_title(f"{LABELS[method]}: {256 - median_indexed:.0f} of 256 regions scan", loc="left")
        ax.set_xlabel("decision window")
        ax.set_ylim(0, 8)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["right", "top"]].set_visible(False)
    axes[0].set_ylabel("regions holding a local index (of 256)")
    axes[0].legend(frameon=False, loc="upper right", fontsize=8)
    fig.suptitle("Indexed-region timeline (median of five seeds); the rest of the grid stays in SCAN",
                 x=.01, ha="left")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_tracking(tracking, windows, windows_per_phase, path):
    fig, ax = plt.subplots(figsize=(11, 4.4), layout="constrained")
    for index, method in enumerate(("S3_THRESHOLD", "S4_GREEDY")):
        shares = []
        counts = []
        for window in range(windows):
            entries = [row for row in tracking if row["method"] == method and row["window"] == window]
            shares.append(sum(1 for r in entries if r["indexed"]) / len(entries) if entries else 0)
            counts.append(statistics.median(r["indexed_regions"] for r in entries) if entries else 0)
        ax.step(range(1, windows + 1), shares, where="mid", color=PALETTE[index], lw=1.8,
                label=f"{LABELS[method]} hotspot indexed")
        ax2 = ax.twinx()
        ax2.plot(range(1, windows + 1), counts, color=PALETTE[index], lw=1.0, alpha=.55, ls=":")
        ax2.set_ylabel("indexed regions (dotted)", color="#666666")
    for boundary in range(windows_per_phase, windows, windows_per_phase):
        ax.axvline(boundary + .5, color="#999999", lw=.8, ls="--")
    ax.set_ylim(-.02, 1.05)
    ax.set_xlabel("decision window")
    ax.set_ylabel("share of windows with the true hotspot indexed")
    ax.set_title("Does the policy index the region that actually receives the hotspot queries?", loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    ax.grid(alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_payback(payback_rows, path):
    values = [r["payback_windows"] for r in payback_rows
              if r["counterfactual_saving_ms"] is not None
              and r["counterfactual_saving_ms"] > 0 and r["payback_windows"] != ""]
    values = [min(float(v), 12.0) for v in values]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), layout="constrained")
    if values:
        ax.hist(values, bins=[0, .5, 1, 2, 3, 4, 6, 9, 12], color="#22577A", edgecolor="white")
    ax.axvline(1.0, color="#B66233", lw=1.2, ls="--")
    ax.text(1.05, ax.get_ylim()[1] * .9, "one-window greedy\nhorizon", color="#B66233", fontsize=8)
    ax.set_xlabel("payback time (windows), capped at 12")
    ax.set_ylabel("applied switches")
    ax.set_title("Switch payback against the fixed-mode counterfactual", loc="left")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_fine_bins(path_in, path_out):
    rows = read_csv(path_in)
    if not rows:
        return
    total = sum(int(r["paired_region_windows"]) for r in rows)
    wins = {mode: sum(int(r[f"{mode}_wins"]) for r in rows)
            for mode in ("SCAN", "FINE_GRID", "LOCAL_QUAD")}
    best_bin = max(rows, key=lambda r: int(r["FINE_GRID_wins"]) / max(1, int(r["paired_region_windows"])))
    best_share = int(best_bin["FINE_GRID_wins"]) / max(1, int(best_bin["paired_region_windows"]))
    fig, ax = plt.subplots(figsize=(7.4, 4.0), layout="constrained")
    labels = ["SCAN", "LOCAL_QUAD", "FINE_GRID"]
    values = [wins["SCAN"], wins["LOCAL_QUAD"], wins["FINE_GRID"]]
    bars = ax.bar(labels, values, color=["#7F8C8D", "#22577A", "#278F85"], width=.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + total * 0.01,
                f"{value}\n({value / total:.2%})", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("region-windows where the mode is cheapest")
    ax.set_ylim(0, total * 1.12)
    ax.set_title(f"FINE_GRID advantage over {total:,} calibration region-windows\n"
                 f"best single bin reaches only {best_share:.1%}", loc="left", fontsize=10)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(path_out, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
