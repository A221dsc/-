"""Summarize controlled M2 costs and plot online cost components."""
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
from run_m1 import write_csv

MODES = ["S0_ALL_SCAN","S1_ALL_FINE","S2_ALL_QUAD"]
LABELS = ["SCAN","FINE_GRID","LOCAL_QUAD"]


def read_csv(path):
    with path.open(encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))


def phase_analysis(directory):
    paired = defaultdict(dict)
    for row in read_csv(directory/"regions.csv"):
        if row["phase_index"] != "-1":
            key = tuple(row[k] for k in ("profile","seed","phase","region_id"))
            paired[key][row["baseline"]] = row
    bins = defaultdict(list)
    phase = defaultdict(Counter)
    cold = []
    for (profile,seed,label,_), methods in paired.items():
        assert set(methods) == set(MODES)
        for field in ("entities","query_count","object_updates"):
            assert len({r[field] for r in methods.values()}) == 1
        n,q,u = [int(methods[MODES[0]][f]) for f in ("entities","query_count","object_updates")]
        costs = tuple(sum(float(methods[m][f]) for f in ("query_ms","local_update_ms"))
                      for m in MODES)
        for i,cost in enumerate(costs):
            phase[profile,label,seed][LABELS[i]] += cost
        if n == 0 and u == 0:
            continue
        density = "0-16" if n <= 16 else "17-64" if n <= 64 else "65-256" if n <= 256 else "257+"
        ratio = ("0" if q == 0 else "0-0.1" if q <= u*.1 else
                 "0.1-1" if q <= u else ">1")
        bins[density,ratio].append(costs)
        if q == 0 and u > 0:
            cold.append(costs)
    result = []
    for (density,ratio), costs in sorted(bins.items()):
        winner_counts = Counter()
        for row in costs:
            winners = [mode for mode,v in zip(LABELS,row) if v == min(row)]
            winner_counts[winners[0] if len(winners) == 1 else "tie"] += 1
        result.append({
            "entities_at_window_end":density,"regional_query_to_update_ratio":ratio,
            "paired_region_windows":len(costs),
            **{f"{mode}_median_ms":statistics.median(row[i] for row in costs)
               for i,mode in enumerate(LABELS)},
            **{f"{mode}_wins":winner_counts[mode] for mode in LABELS},
            "ties":winner_counts["tie"],
        })
    phase_bins = defaultdict(list)
    for (profile,label,seed), costs in phase.items():
        phase_bins[profile,label].append(costs)
    phase_rows = [{"profile":profile,"phase":label,
                   **{f"{m}_median_query_plus_local_ms":statistics.median(v[m] for v in values)
                      for m in LABELS}}
                  for (profile,label),values in sorted(phase_bins.items())]
    return result,phase_rows,{
        "observations":len(cold),
        "median_ms":{m:statistics.median(row[i] for row in cold) for i,m in enumerate(LABELS)},
        "aggregate_ms":{m:sum(row[i] for row in cold) for i,m in enumerate(LABELS)},
    }


def plot(directory, output):
    profiles = json.loads((directory/"config.json").read_text())["profiles"]
    metrics = read_csv(directory/"metrics.csv")
    labels = ["70% hot\nclustered\n50% update","0% hot","95% hot","Uniform",
              "Strong\nclustering","10% update","100% update"]
    fig,(query,maintenance) = plt.subplots(2,1,figsize=(12,7),sharex=True,layout="constrained")
    width = .24
    palette = ["#22577A","#278F85","#B66233"]
    for i,mode in enumerate(MODES):
        x = [j+(i-1)*width for j in range(len(profiles))]
        samples = [[r for r in metrics if r["profile"] == profile and r["baseline"] == mode]
                   for profile in profiles]
        latency = [statistics.median(float(r["query_wall_ms"]) for r in group) for group in samples]
        coarse = [statistics.median(float(r["coarse_update_ms"]) for r in group) for group in samples]
        local = [statistics.median(float(r["local_update_ms"]) for r in group) for group in samples]
        query.bar(x,latency,width,color=palette[i],label=f"S{i} {LABELS[i]}")
        maintenance.bar(x,coarse,width,color=palette[i],alpha=.38)
        maintenance.bar(x,local,width,bottom=coarse,color=palette[i])
    maintenance.set_xticks(list(range(len(labels))),labels)
    query.set_ylabel("Query time (ms)")
    maintenance.set_ylabel("Update time (ms)")
    query.set_title("Query cost in the primary replay",loc="left")
    maintenance.set_title("Update cost in the separate profile: coarse (light) and local (solid)",loc="left")
    fig.suptitle("M2 controlled workload at 20,000 objects | medians of five paired seeds",fontsize=14)
    query.legend(ncol=3,frameon=False,loc="upper left")
    query.margins(y=.35)
    for ax in (query,maintenance):
        ax.grid(axis="y",alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["right","top"]].set_visible(False)
    fig.savefig(output/"regional_costs_20k.png",dpi=175)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("controlled_20k",type=Path)
    p.add_argument("output",type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    bins,phases,cold = phase_analysis(args.controlled_20k)
    write_csv(args.output/"regional_cost_by_density_and_load.csv",bins)
    write_csv(args.output/"regional_cost_by_phase.csv",phases)
    json_file(args.output/"cold_region_maintenance.json",cold)
    plot(args.controlled_20k,args.output)


if __name__ == "__main__":
    main()
