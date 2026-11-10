"""Analyse the M4 comparison: costs, promotions, lead time and accuracy figures."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prepare import json_file

FULL = {
    "Fixed Regional Scan": "Fixed Regional Scan",
    "Fixed Local Fine Grid": "Fixed Local Fine Grid",
    "Fixed Local Quadtree": "Fixed Local Quadtree",
    "Threshold-Based Adaptive Regional Index Selection": "Threshold adaptive (history)",
    "Threshold-Based Adaptive Regional Index Selection (Scenario)": "Threshold adaptive (scenario)",
    "Cost-Aware Greedy Regional Index Selection": "Cost-aware greedy (history)",
    "Cost-Aware Greedy Regional Index Selection (Scenario)": "Cost-aware greedy (scenario)",
    "Uniform Grid": "Uniform Grid",
    "PR Quadtree": "PR Quadtree",
    "Boost R*-Tree": "Boost R*-Tree",
}
POLICIES = [m for m in FULL if "Adaptive" in m or "Greedy" in m]
PALETTE = ["#22577A", "#278F85", "#B66233", "#7D5BA6", "#C8553D", "#5B8E7D",
           "#8D6E63", "#455A64", "#9C6ADE", "#E08B2E"]


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def loads_path(directory, scenario, actual):
    name = (f"{scenario}.actual_loads.csv" if actual == 0
            else f"{scenario}.actual{actual}.loads.csv")
    return Path(directory) / name


def read_loads(path):
    windows, regions = 24, 256
    table = np.zeros((windows, regions, 4))
    for row in read_csv(path):
        table[int(row["window"]), int(row["region"])] = (
            float(row["query_count"]), float(row["entity_count"]),
            float(row["update_count"]), float(row["coverage"]))
    return table


def hotspot_onsets(loads):
    tops = [int(np.argmax(loads[w, :, 0])) for w in range(loads.shape[0])]
    return [w for w in range(1, len(tops)) if tops[w] != tops[w - 1]], tops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("accuracy", type=Path)
    parser.add_argument("scenarios", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = read_csv(args.run / "metrics.csv")
    decisions = read_csv(args.run / "decisions.csv")
    regions = read_csv(args.run / "regions.csv")
    summary = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
    paired = json.loads((args.run / "paired.json").read_text(encoding="utf-8"))

    overall = []
    for split in ("validation", "test"):
        rows = [r for r in metrics if r["split"] == split]
        by_method = defaultdict(list)
        for row in rows:
            by_method[row["method"]].append(row)
        for method, group in by_method.items():
            median = lambda field: statistics.median(float(r[field]) for r in group)
            overall.append({
                "split": split, "method": method, "label": FULL[method],
                "online_ms": median("online_ms"), "query_ms": median("query_ms"),
                "update_ms": median("update_ms"), "switch_ms": median("switch_ms"),
                "policy_ms": median("policy_ms"), "total_ms": median("total_ms"),
                "p95_us": median("p95_us"), "p99_us": median("p99_us"),
                "switches": median("switches"), "promotions": median("promotions"),
                "demotions": median("demotions"), "traces": len(group),
            })
    write_csv(args.output / "table_overall.csv", overall)

    feedforward = []
    for name, values in paired["pairs"].items():
        if "/ history" in name:
            feedforward.append({"comparison": name, "median_ratio": values["median_ratio"],
                                "bootstrap95_low": values["bootstrap95"][0],
                                "bootstrap95_high": values["bootstrap95"][1],
                                "wins": values["wins"], "n": values["n"]})
    write_csv(args.output / "table_feedforward.csv", feedforward)
    json_file(args.output / "paired.json", paired)

    # Invalid promotions: governed-window actual query count below five.
    promotion_rows = []
    loads_cache = {}
    for method in POLICIES:
        for split in ("validation", "test"):
            rows = [d for d in decisions if d["method"] == method and d["split"] == split
                    and d["status"] == "applied" and d["old_mode"] == "SCAN"
                    and d["selected_mode"] != "SCAN"]
            values = []
            for row in rows:
                key = (row["scenario"], int(row["actual"]))
                if key not in loads_cache:
                    loads_cache[key] = read_loads(loads_path(args.scenarios, *key))
                loads = loads_cache[key]
                window = int(row["window_index"])
                region = int(row["region_id"])
                if window < loads.shape[0]:
                    values.append(loads[window, region, 0])
            promotion_rows.append({
                "method": method, "label": FULL[method], "split": split,
                "promotions": len(values),
                "invalid_promotions": sum(1 for v in values if v < 5),
                "invalid_share": (sum(1 for v in values if v < 5) / len(values)) if values else None,
                "median_governed_queries": statistics.median(values) if values else None,
            })
    write_csv(args.output / "table_promotions.csv", promotion_rows)

    # Lead time and post-onset extra cost against the fixed local quadtree.
    baseline_windows = defaultdict(dict)
    policy_windows = defaultdict(dict)
    for row in regions:
        if row["split"] != "test":
            continue
        key = (row["scenario"], int(row["actual"]), int(row["window_index"]))
        cost = float(row["query_ms"]) + float(row["local_update_ms"]) + float(row["switch_ms"])
        if row["method"] == "Fixed Local Quadtree":
            baseline_windows[key]["cost"] = baseline_windows[key].get("cost", 0.0) + cost
        elif row["method"] in POLICIES:
            policy_windows[(row["method"], key)]["cost"] = \
                policy_windows[(row["method"], key)].get("cost", 0.0) + cost
    mode_of = {}
    for row in regions:
        if row["split"] == "test" and row["method"] in POLICIES:
            mode_of[(row["method"], row["scenario"], int(row["actual"]),
                     int(row["window_index"]), int(row["region_id"]))] = row["mode"]
    lead_rows = []
    for scenario, actual in sorted({(r["scenario"], int(r["actual"])) for r in metrics
                                    if r["split"] == "test"}):
        loads = read_loads(loads_path(args.scenarios, scenario, actual))
        onsets, tops = hotspot_onsets(loads)
        for method in POLICIES:
            leads, extra, base_extra = [], [], []
            for w in onsets:
                region = tops[w]
                first = None
                for k in range(0, w + 1):
                    mode = mode_of.get((method, scenario, actual, k, region))
                    if mode is not None and mode != "SCAN":
                        first = k
                        break
                leads.append(None if first is None else w - first)
                for offset in range(0, 2):
                    key = (scenario, actual, min(23, w + offset))
                    if key in baseline_windows and (method, key) in policy_windows:
                        extra.append(policy_windows[(method, key)]["cost"])
                        base_extra.append(baseline_windows[key]["cost"])
            found = [value for value in leads if value is not None]
            lead_rows.append({
                "method": method, "label": FULL[method], "scenario": scenario,
                "actual": actual, "onsets": len(onsets),
                "share_indexed_before_onset": sum(1 for v in found if v >= 1) / len(onsets) if onsets else None,
                "median_lead_windows": statistics.median(found) if found else None,
                "extra_cost_ms": (statistics.fmean(extra) - statistics.fmean(base_extra))
                if extra and base_extra else None,
            })
    write_csv(args.output / "table_lead.csv", lead_rows)

    # Figures.
    plot_accuracy(args.accuracy / "summary.json", args.output / "fig_accuracy.png")
    plot_online(overall, args.output / "fig_online_cost.png")
    plot_cumulative(regions, args.output / "fig_cumulative.png")
    print("wrote tables and figures to", args.output)
    for row in overall:
        if row["split"] == "test":
            print(f"{row['label']:38s} online={row['online_ms']:7.1f} q={row['query_ms']:7.1f} "
                  f"u={row['update_ms']:7.1f} sw={row['switch_ms']:6.2f}")


def plot_accuracy(path, output):
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    kinds = [("scenario", "Scenario prior", "#22577A"),
             ("persistence", "Persistence", "#B66233"),
             ("ewma", "EWMA (a=0.7)", "#7D5BA6")]
    metrics = [("query_mae_per_window", "query count MAE / region"),
               ("mean_tv_distance", "spatial TV distance"),
               ("hotspot_top1_accuracy", "next-window hotspot top-1")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    for ax, (field, title) in zip(axes, metrics):
        values = [summary["test"][kind][field] for kind, _, _ in kinds]
        ax.bar([label for _, label, _ in kinds], values,
               color=[color for _, _, color in kinds], width=.62)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["right", "top"]].set_visible(False)
        for index, value in enumerate(values):
            ax.text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Load derivation accuracy on the test actuals (lower is better except top-1)", x=.01, ha="left")
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_online(overall, output):
    rows = [r for r in overall if r["split"] == "test"]
    rows.sort(key=lambda r: r["online_ms"])
    labels = [r["label"].replace(" (", "\n(") for r in rows]
    fig, ax = plt.subplots(figsize=(11, 4.8), layout="constrained")
    bottom = [0.0] * len(rows)
    for name, field, color in (("query", "query_ms", "#22577A"),
                               ("update", "update_ms", "#B66233"),
                               ("switch + policy", None, "#7D5BA6")):
        values = ([r["switch_ms"] + r["policy_ms"] for r in rows] if field is None
                  else [r[field] for r in rows])
        ax.bar(labels, values, bottom=bottom, color=color, label=name, width=.7)
        bottom = [b + v for b, v in zip(bottom, values)]
    ax.set_ylabel("online time (ms)")
    ax.set_title("M4 public-map scenarios: online cost, median of six test replays", loc="left")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[["right", "top"]].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right", fontsize=8)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_cumulative(regions, output):
    windows = 24
    per = defaultdict(lambda: defaultdict(float))
    for row in regions:
        if row["split"] != "test":
            continue
        key = (row["method"], row["scenario"], int(row["actual"]))
        per[key][int(row["window_index"])] += (float(row["query_ms"]) +
                                               float(row["local_update_ms"]) +
                                               float(row["switch_ms"]))
    traces = sorted({(s, a) for _, s, a in per})
    fig, ax = plt.subplots(figsize=(10, 4.6), layout="constrained")
    for index, method in enumerate(["Fixed Local Quadtree", "PR Quadtree",
                                    "Cost-Aware Greedy Regional Index Selection",
                                    "Cost-Aware Greedy Regional Index Selection (Scenario)",
                                    "Threshold-Based Adaptive Regional Index Selection",
                                    "Threshold-Based Adaptive Regional Index Selection (Scenario)"]):
        cumulative = []
        running = 0.0
        for window in range(windows):
            values = [per.get((method, s, a), {}).get(window, 0.0) for s, a in traces]
            running += statistics.median(values) if values else 0.0
            cumulative.append(running)
        ax.plot(range(1, windows + 1), cumulative, marker="o", ms=2.6, lw=1.7,
                color=PALETTE[index], label=FULL[method])
    for boundary in (6, 12, 18):
        ax.axvline(boundary + .5, color="#999999", lw=.8, ls="--")
    ax.set_xlabel("decision window (50 steps)")
    ax.set_ylabel("cumulative profile cost (ms)")
    ax.set_title("Cumulative regional cost on the test scenarios", loc="left")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(output, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
