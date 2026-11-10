"""Load-derivation accuracy: scenario prior vs persistence vs EWMA.

All three predict the next region loads from information available before the
target window. The nominal prior uses planned routes and the planned query
schedule only; persistence and EWMA use the actual windows already observed.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np

from prepare import json_file

SIDE = 16
REGIONS = SIDE * SIDE


def read_table(path):
    table = {}
    with Path(path).open(encoding="utf-8") as handle:
        header = next(handle)
        assert header.startswith("window,region"), path
        for line in handle:
            window, region, queries, entities, updates, coverage = line.split(",")
            table[(int(window), int(region))] = (float(queries), float(entities),
                                                float(updates), float(coverage))
    windows = max(w for w, _ in table) + 1
    array = np.zeros((windows, REGIONS, 4))
    for (window, region), value in table.items():
        array[window, region] = value
    return array


def tv_distance(a, b):
    a = np.maximum(a, 0.0)
    b = np.maximum(b, 0.0)
    if a.sum() > 0:
        a = a / a.sum()
    if b.sum() > 0:
        b = b / b.sum()
    return 0.5 * float(np.abs(a - b).sum())


def ewma(actual, alpha=0.7):
    state = np.zeros_like(actual)
    current = np.zeros(REGIONS)
    for w in range(actual.shape[0]):
        current = alpha * actual[w] + (1 - alpha) * current
        state[w] = current
    return state


def hotspot(loads):
    return int(np.argmax(loads))


def topk(loads, k=3):
    return set(np.argsort(loads)[-k:].tolist())


def next_prediction(kind, window, nominal, actual, smooth):
    if kind == "scenario":
        return nominal[window, :, 0]
    if kind == "persistence":
        return actual[window - 1, :, 0]
    return smooth[window - 1]


def horizon_aggregate(kind, k, nominal, actual, smooth, horizon=4):
    windows = actual.shape[0]
    high = min(windows, k + 1 + horizon)
    if kind == "scenario":
        return nominal[k + 1:high, :, 0].sum(axis=0), k + 1, high
    if kind == "persistence":
        return actual[k, :, 0] * (high - k - 1), k + 1, high
    return smooth[k] * (high - k - 1), k + 1, high


def evaluate(split, scenarios, actuals, directory):
    rows, window_rows, onset_rows = [], [], []
    for scenario in scenarios:
        nominal = read_table(directory / "scenarios" / f"{scenario}.nominal.csv")
        for actual_index in actuals:
            name = (f"{scenario}.actual_loads.csv" if actual_index == 0
                    else f"{scenario}.actual{actual_index}.loads.csv")
            loads = read_table(directory / "loads" / name)
            windows = min(nominal.shape[0], loads.shape[0])
            nominal, actual = nominal[:windows], loads[:windows]
            smooth = ewma(actual[:, :, 0])
            for kind in ("scenario", "persistence", "ewma"):
                query_errors, entity_errors, update_errors, tv = [], [], [], []
                top1 = top3 = 0
                for w in range(1, windows):
                    predicted = next_prediction(kind, w, nominal, actual, smooth)
                    truth = actual[w, :, 0]
                    query_errors.append(float(np.abs(predicted - truth).sum()))
                    entity_errors.append(float(np.abs(
                        (nominal[w, :, 1] if kind == "scenario" else actual[w - 1, :, 1]) -
                        actual[w, :, 1]).sum()))
                    update_errors.append(float(np.abs(
                        (nominal[w, :, 2] if kind == "scenario" else actual[w - 1, :, 2]) -
                        actual[w, :, 2]).sum()))
                    tv.append(tv_distance(predicted, truth))
                    top1 += hotspot(predicted) == hotspot(truth)
                    top3 += hotspot(truth) in topk(predicted)
                    window_rows.append({"scenario": scenario, "actual": actual_index,
                                        "predictor": kind, "window": w,
                                        "top1_hit": int(hotspot(predicted) == hotspot(truth)),
                                        "tv_distance": tv[-1],
                                        "query_error": query_errors[-1]})
                horizon_hits = 0
                horizon_total = 0
                aggregates = {}
                for k in range(0, windows - 1):
                    aggregate, low, high = horizon_aggregate(kind, k, nominal, actual, smooth)
                    aggregates[k] = (aggregate, low, high)
                    horizon_hits += hotspot(aggregate) == hotspot(actual[low:high, :, 0].sum(axis=0))
                    horizon_total += 1
                onset, detected = [], []
                for w in range(2, windows):
                    if hotspot(actual[w, :, 0]) == hotspot(actual[w - 1, :, 0]):
                        continue
                    target = hotspot(actual[w, :, 0])
                    lead = None
                    for k in range(max(0, w - 4), w):
                        aggregate, low, high = aggregates[k]
                        if low < w <= high and hotspot(aggregate) == target:
                            lead = w - k
                            break
                    onset.append(lead is not None)
                    if lead is not None:
                        detected.append(lead)
                onset_rows.append({"split": split, "scenario": scenario, "actual": actual_index,
                                   "predictor": kind, "onsets": len(onset),
                                   "detected": sum(onset),
                                   "share_detected": (sum(onset) / len(onset)) if onset else None,
                                   "median_lead_windows": statistics.median(detected) if detected else None})
                rows.append({
                    "split": split, "scenario": scenario, "actual": actual_index,
                    "predictor": kind,
                    "query_mae_per_window": statistics.fmean(query_errors) / REGIONS,
                    "query_total_relative_error": statistics.fmean(
                        e / max(1.0, float(actual[w, :, 0].sum()))
                        for w, e in zip(range(1, windows), query_errors)),
                    "entity_mae_per_window": statistics.fmean(entity_errors) / REGIONS,
                    "update_mae_per_window": statistics.fmean(update_errors) / REGIONS,
                    "mean_tv_distance": statistics.fmean(tv),
                    "hotspot_top1_accuracy": top1 / max(1, windows - 1),
                    "hotspot_top3_accuracy": top3 / max(1, windows - 1),
                    "horizon4_hotspot_accuracy": horizon_hits / max(1, horizon_total),
                })
    return rows, window_rows, onset_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    all_rows, all_windows, all_onsets = [], [], []
    for split in ("validation", "test"):
        scenarios = [f"{split}-{i}-{Path(m).stem}" for i, m in enumerate(config["maps"][split])]
        actuals = [0, 1, 2] if split == "test" else [0, 1]
        rows, window_rows, onset_rows = evaluate(split, scenarios, actuals, args.directory)
        all_rows += rows
        all_windows += window_rows
        all_onsets += onset_rows
    for name, rows in (("metrics.csv", all_rows), ("by_window.csv", all_windows),
                       ("onsets.csv", all_onsets)):
        with (args.output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {}
    for split in ("validation", "test"):
        summary[split] = {}
        for kind in ("scenario", "persistence", "ewma"):
            subset = [r for r in all_rows if r["split"] == split and r["predictor"] == kind]
            onsets = [r for r in all_onsets if r["split"] == split and r["predictor"] == kind]
            lead = [r["median_lead_windows"] for r in onsets if r["median_lead_windows"] is not None]
            summary[split][kind] = {
                **{field: statistics.fmean(float(r[field]) for r in subset)
                   for field in ("query_mae_per_window", "query_total_relative_error",
                                 "entity_mae_per_window", "update_mae_per_window",
                                 "mean_tv_distance", "hotspot_top1_accuracy",
                                 "hotspot_top3_accuracy", "horizon4_hotspot_accuracy")},
                "onset_share_detected": statistics.fmean(
                    r["share_detected"] for r in onsets if r["share_detected"] is not None),
                "onset_median_lead_windows": statistics.median(lead) if lead else None,
            }
    json_file(args.output / "summary.json", summary)
    for split in ("validation", "test"):
        print(split)
        for kind, values in summary[split].items():
            print(f"  {kind:12s} mae={values['query_mae_per_window']:.2f} "
                  f"rel={values['query_total_relative_error']:.3f} tv={values['mean_tv_distance']:.4f} "
                  f"top1={values['hotspot_top1_accuracy']:.3f} h4={values['horizon4_hotspot_accuracy']:.3f} "
                  f"onset={values['onset_share_detected']:.3f} "
                  f"lead={values['onset_median_lead_windows']}")


if __name__ == "__main__":
    main()
