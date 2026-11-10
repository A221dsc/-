"""Calibrate the nominal-load residual model on training maps and draw samples.

The calibration uses only training scenarios and their actual replays. Samples
combine a global phase-timing shift with per-region log-normal residuals, so the
uncertainty-aware policy sees a distribution of plausible future loads rather
than a single expectation.
"""
import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from prepare import json_file

SIDE = 16
REGIONS = SIDE * SIDE
SAMPLES = 8
BINS = [(0.0, 0.5), (0.5, 5.0), (5.0, 20.0), (20.0, 100.0), (100.0, float("inf"))]
SHIFTS = [0, 0, 0, 0, 1, -1, 1, -1]


def read_nominal(path):
    table = {}
    with Path(path).open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            fields = line.split(",")
            window, region = int(fields[0]), int(fields[1])
            table[(window, region)] = (float(fields[2]), float(fields[3]),
                                       float(fields[4]), float(fields[5]),
                                       float(fields[6]) if len(fields) > 6 else 0.0)
    windows = max(w for w, _ in table) + 1
    array = np.zeros((windows, REGIONS, 5))
    for (window, region), value in table.items():
        array[window, region] = value
    return array


def read_loads(path):
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


def calibrate(directory, scenarios):
    residuals = {"queries": [[] for _ in BINS], "updates": [[] for _ in BINS]}
    for scenario in scenarios:
        nominal = read_nominal(directory / "scenarios" / f"{scenario}.nominal.csv")
        loads = read_loads(directory / "loads" / f"{scenario}.actual_loads.csv")
        windows = min(nominal.shape[0], loads.shape[0])
        for w in range(1, windows):
            for region in range(REGIONS):
                for field, column in (("queries", 0), ("updates", 2)):
                    predicted = nominal[w, region, column]
                    observed = loads[w, region, column]
                    if predicted <= 0 and observed <= 0:
                        continue
                    value = math.log((observed + 0.5) / (predicted + 0.5))
                    residuals[field][bin_of(predicted)].append(value)
    model = {}
    for field, bins in residuals.items():
        rows = []
        for index, (low, high) in enumerate(BINS):
            values = bins[index]
            if len(values) < 30:
                rows.append({"low": low, "high": high if high != float("inf") else None,
                             "count": len(values), "mean": 0.0, "std": 0.25})
                continue
            # Low-count bins are dominated by Poisson noise; clip the corrections so
            # the calibrated model does not amplify rounding noise into a bias.
            mean = statistics.fmean(values)
            std = statistics.pstdev(values)
            if low < 5.0:
                mean = 0.0
            rows.append({"low": low, "high": high if high != float("inf") else None,
                         "count": len(values),
                         "mean": max(-0.2, min(0.2, mean)),
                         "std": max(0.05, min(0.5, std))})
        model[field] = rows
    return model


def sample_table(nominal, model, rng, sample_index, use_noise=True, use_shift=True):
    windows, regions, _ = nominal.shape
    shift = SHIFTS[sample_index % len(SHIFTS)] if use_shift else 0
    table = np.zeros((windows, regions, 4))
    for w in range(windows):
        source = min(windows - 1, max(0, w + shift))
        for region in range(regions):
            queries = nominal[source, region, 0]
            entities = nominal[source, region, 1]
            updates = nominal[source, region, 2]
            coverage = nominal[source, region, 3]
            if sample_index == 0 or not use_noise:
                table[w, region] = (queries, entities, updates, coverage)
                continue
            for field, column in (("queries", 0), ("updates", 2)):
                predicted = queries if column == 0 else updates
                row = model[field][bin_of(predicted)]
                epsilon = rng.normal(row["mean"], max(row["std"], 0.05))
                scaled = max(0.0, (predicted + 0.5) * math.exp(epsilon) - 0.5)
                if column == 0:
                    queries = scaled
                else:
                    updates = scaled
            coverage = coverage * (max(1e-6, queries) / max(1e-6, nominal[source, region, 0]))
            table[w, region] = (queries, entities, updates, coverage)
    return table


def write_samples(path, tables):
    lines = ["sample,window,region,query_count,entity_count,update_count,coverage"]
    for index, table in enumerate(tables):
        windows = table.shape[0]
        for w in range(windows):
            for region in range(REGIONS):
                q, n, u, c = table[w, region][:4]
                lines.append(f"{index},{w},{region},{q:.6f},{n:.3f},{u:.6f},{c:.6f}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_point(path, table):
    lines = ["window,region,query_count,entity_count,update_count,coverage,query_centre"]
    windows = table.shape[0]
    for w in range(windows):
        for region in range(REGIONS):
            q, n, u, c = table[w, region][:4]
            lines.append(f"{w},{region},{q:.6f},{n:.3f},{u:.6f},{c:.6f},0")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    train = [f"train-{i}-{Path(m).stem}" for i, m in enumerate(config["maps"]["train"])]
    model = calibrate(args.directory, train)
    json_file(args.output / "residual_model.json", {"bins": BINS, "model": model,
                                                    "training_scenarios": train,
                                                    "samples": SAMPLES, "shifts": SHIFTS})
    rng = np.random.default_rng(20261111)
    for split in ("train", "validation", "test"):
        for index, map_name in enumerate(config["maps"][split]):
            ident = f"{split}-{index}-{Path(map_name).stem}"
            nominal = read_nominal(args.directory / "scenarios" / f"{ident}.nominal.csv")
            tables = [sample_table(nominal, model, rng, m) for m in range(SAMPLES)]
            write_samples(args.directory / "scenarios" / f"{ident}.nominal_samples.csv", tables)
            write_point(args.directory / "scenarios" / f"{ident}.nominal_calibrated.csv", tables[0])
            # Ablations: no spatial residual, no phase shift, no reporting lag and
            # the centre-only footprint baseline.
            write_samples(args.directory / "scenarios" / f"{ident}.samples_nospatial.csv",
                          [sample_table(nominal, model, rng, m, use_noise=False)
                           for m in range(SAMPLES)])
            write_samples(args.directory / "scenarios" / f"{ident}.samples_nophase.csv",
                          [sample_table(nominal, model, rng, m, use_shift=False)
                           for m in range(SAMPLES)])
            centres = nominal.copy()
            centres[:, :, 3] = nominal[:, :, 0]
            centres[:, :, 0] = nominal[:, :, 4]
            write_point(args.directory / "scenarios" / f"{ident}.nominal_centre_only.csv",
                        centres)
            nolag = tables[0].copy()
            nolag[:, :, 1] = nominal[:, :, 1]
            write_point(args.directory / "scenarios" / f"{ident}.nominal_nolag.csv", nolag)
            print(f"{ident}: samples + calibrated point + ablations", flush=True)


if __name__ == "__main__":
    main()
