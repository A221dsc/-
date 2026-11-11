#!/usr/bin/env bash
# Reproduce H3-P0 management-partition correctness and benchmark experiments.
#
# Usage:
#   bash scripts/run_h3_partition.sh            # full 3-seed, two-map experiment
#   bash scripts/run_h3_partition.sh quick      # one seed, small entity count
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

bash scripts/build_h3_p0.sh
mkdir -p results/h3_partition/seeds_4km results/h3_partition/seeds_20km

./build_h3_p0/h3_management_tests.exe | tee results/h3_partition/tests.log

if [[ "${1:-full}" == "quick" ]]; then
  ./build_h3_p0/h3_partition_bench.exe 5000 5000 5000 9 \
    results/h3_partition/quick.csv clustered 4096 101 \
    | tee results/h3_partition/quick.log
  exit 0
fi

for seed in 101 202 303; do
  ./build_h3_p0/h3_partition_bench.exe 20000 20000 20000 9 \
    "results/h3_partition/seeds_4km/bench_${seed}.csv" clustered 4096 "$seed" \
    > "results/h3_partition/seeds_4km/log_${seed}.txt" 2>&1
  ./build_h3_p0/h3_partition_bench.exe 20000 10000 10000 10 \
    "results/h3_partition/seeds_20km/bench_${seed}.csv" uniform 20000 "$seed" \
    > "results/h3_partition/seeds_20km/log_${seed}.txt" 2>&1
  echo "seed ${seed} done"
done

python scripts/summarize_h3_partition.py \
  results/h3_partition/seeds_4km results/h3_partition/summary_4km.csv
python scripts/summarize_h3_partition.py \
  results/h3_partition/seeds_20km results/h3_partition/summary_20km.csv

echo "H3-P0 management-partition experiment complete"
