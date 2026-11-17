"""Summarize paired independent runs and parse the instrumented phase traces."""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from run_h3_p0_v2 import digest_csv


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_interval(ratios):
    if len(ratios) < 2:
        return "", ""
    rng = random.Random(28411)
    reps = sorted(100 * (1 - math.exp(statistics.mean(rng.choices(ratios, k=len(ratios)))))
                  for _ in range(20000))
    return reps[500], reps[19499]


def summarize(root):
    completion = json.loads((root / "complete.json").read_text(encoding="utf-8"))
    environment = json.loads((root / "environment.json").read_text(encoding="utf-8"))
    if digest_csv(root / "timings.csv") != completion["timings_sha256"]:
        raise ValueError("timing file changed after replay completion")
    runs = read_csv(root / "timings.csv")
    if len(runs) != completion["runs"] or list(range(1, len(runs) + 1)) != [
            int(row["sequence"]) for row in runs]:
        raise ValueError("incomplete or duplicated run sequence")
    closed = defaultdict(dict)
    sentinels = defaultdict(dict)
    fixed_summary = defaultdict(list)
    open_runs = defaultdict(list)
    profiles = []
    for row in runs:
        kind = row["experiment"]
        block = int(row["block"])
        case = (row["scenario"], int(row["workers"]))
        method = row["method"]
        if kind == "closed":
            if (block, method) in closed[case] or int(row["seed"]) != environment["seeds"][block]:
                raise ValueError("invalid closed-loop pair")
            closed[case][block, method] = int(row["wall_us"])
            fixed_summary[case + (method,)].append(row)
        elif kind.startswith("sentinel_"):
            sentinels[block][kind] = int(row["wall_us"])
        elif kind == "open":
            open_runs[(row["scenario"], int(row["period_us"]), method)].append(row)
        elif kind == "profile":
            profiles.append(row)

    for scenario in ("W8-A", "W8-B", "W8-C"):
        for workers in environment["threads"]:
            methods = environment["methods"] + (["A+"] if workers == 1 else [])
            expected = {(block, method) for block in range(len(environment["seeds"])) for method in methods}
            if closed[(scenario, workers)].keys() != expected:
                raise ValueError("incomplete closed-loop block")

    comparable = set()
    sentinel_summary = []
    for block, pair in sorted(sentinels.items()):
        ratio = pair["sentinel_after"] / pair["sentinel_before"]
        passed = .95 <= ratio <= 1.05
        if passed:
            comparable.add(block)
        sentinel_summary.append(dict(block=block, before_us=pair["sentinel_before"],
                                     after_us=pair["sentinel_after"], ratio=ratio, passed=passed))
    save_csv(root / "sentinels.csv", sentinel_summary)

    pairs = []
    for (scenario, threads), measurements in sorted(closed.items()):
        for base in ("A+Lock", "A+Static", "A+StaticMatched"):
            blocks = sorted(block for block, method in measurements if method == base)
            for scope, selected in (("all", blocks),
                                    ("sentinel_pass", [b for b in blocks if b in comparable])):
                if not selected:
                    continue
                ratios = [math.log(measurements[b, "B+"] / measurements[b, base]) for b in selected]
                lower, upper = paired_interval(ratios)
                pairs.append(dict(scenario=scenario, workers=threads, baseline=base, scope=scope,
                                  pairs=len(ratios), b_wins=sum(r < 0 for r in ratios),
                                  savings_pct=100 * (1 - math.exp(statistics.mean(ratios))),
                                  ci95_low_pct=lower, ci95_high_pct=upper))
    save_csv(root / "paired.csv", pairs)

    def summary(groups, names, fields):
        rows = []
        for keys, group in sorted(groups.items()):
            row = dict(zip(names, keys), runs=len(group))
            for f in fields:
                row[f"median_{f}"] = statistics.median(float(r[f]) for r in group)
            rows.append(row)
        return rows

    latencies = ("q50_ns", "q95_ns", "q99_ns", "u50_ns", "u95_ns", "u99_ns")
    rates = ("wall_us", "query_per_s", "update_per_s")
    save_csv(root / "fixed_summary.csv", summary(
        fixed_summary, ("scenario", "workers", "method"),
        rates + latencies + ("construct_us", "mapping_ns", "index_bytes", "snapshot_bytes",
                             "mapping_bytes", "peak_rss_bytes")))
    save_csv(root / "open_summary.csv", summary(
        open_runs, ("scenario", "period_us", "method"),
        rates + latencies + ("late_ticks", "missed_ticks", "deadline_misses", "pending_at_horizon")))

    phases = []
    fields = ("prepare_ns", "cover_cpu_ns", "route_cpu_ns", "dispatch_ns", "execute_ns",
              "merge_ns", "barrier_tail_ns", "query_tasks", "update_tasks", "cells", "candidates",
              "read_locks", "write_locks", "read_misses", "write_misses", "read_wait_ns",
              "write_wait_ns", "migration_wait_ns", "queue_wait_ns", "query_ns", "update_ns",
              "migration_ns", "idle_window_ns", "same_cell", "cross_cell", "intrapartition",
              "cross_partition", "cross_worker", "insertions", "deletions", "snapshot_cells",
              "snapshot_entries")
    wall_fields = {"prepare_ns", "cover_cpu_ns", "route_cpu_ns", "dispatch_ns", "execute_ns",
                   "merge_ns", "barrier_tail_ns"}
    for profile in profiles:
        trace = read_csv(root / "profiles" / f"{int(profile['sequence']):04d}.csv")
        stages = ("all", "A", "transition", "B") if profile["scenario"] == "W8-C" else ("all",)
        for stage_index, stage in enumerate(stages):
            selected = [r for r in trace if not stage_index or
                        1 + int(r["tick"]) * 3 // int(profile["ticks"]) == stage_index]
            wall = [r for r in selected if r["worker"] == "0"]
            result = {f: sum(int(r[f]) for r in (wall if f in wall_fields else selected)) for f in fields}
            worker_time = Counter()
            worker_load = Counter()
            ticks = defaultdict(list)
            for r in selected:
                worker = int(r["worker"])
                busy = int(r["busy_ns"])
                worker_time[worker] += busy
                worker_load[worker] += int(r["candidates"]) + int(r["update_tasks"])
                ticks[int(r["tick"])].append(busy)
            workers = list(worker_time.values())
            load = list(worker_load.values())
            max_busy = sum(map(max, ticks.values()))
            # Busy is measured inside the execution phase, so this residual
            # measures phase overhead above the longest observed worker per tick.
            result.update(max_worker_busy_ns=max(workers), mean_worker_busy_ns=statistics.mean(workers),
                          var_worker_busy_ns2=statistics.pvariance(workers), max_worker_combined_work=max(load),
                          mean_worker_combined_work=statistics.mean(load), var_worker_combined_work=statistics.pvariance(load),
                          max_worker_busy_sum_ns=max_busy, execute_residual_ns=result["execute_ns"] - max_busy)
            phases.append(dict(scenario=profile["scenario"], method=profile["method"],
                               workers=profile["workers"], stage=stage, **result))
    save_csv(root / "phases.csv", phases)
    print(f"{len(runs)} runs, {len(comparable)}/{len(sentinels)} sentinel pairs within tolerance")
    for row in pairs:
        if row["scenario"] == "W8-A" and row["workers"] == 8 and row["scope"] == "all":
            print(row)


if __name__ == "__main__":
    summarize(Path(sys.argv[1]))
