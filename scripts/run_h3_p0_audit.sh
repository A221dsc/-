#!/usr/bin/env bash
# H3-P0 audit: re-measure the A/B/C comparison against a fairly optimised
# global baseline (A+), and separate the shared costs (candidate cover,
# result sorting) from per-cell bucket work.
#
# Usage:
#   bash scripts/run_h3_p0_audit.sh          # 3 seeds, both maps
#   bash scripts/run_h3_p0_audit.sh quick    # 1 seed, both maps
#
# Requires the object files produced by scripts/build_h3_p0.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD="build_h3_p0"
OUT="results/h3_p0_audit"
H3_INC="third_party/h3-4.1.0/src/h3lib/include"

if [[ ! -f "$BUILD/obj/cpp/h3_management.o" ]]; then
  echo "missing $BUILD/obj/cpp/h3_management.o -- run scripts/build_h3_p0.sh first" >&2
  exit 1
fi
mkdir -p "$OUT"

g++ -std=c++20 -O2 -Wall -Wextra \
  -I. -Isrc -I"$H3_INC" -I"$BUILD/include" \
  tests/h3_p0_audit.cpp \
  "$BUILD/obj/cpp/geo.o" "$BUILD/obj/cpp/local_projection.o" \
  "$BUILD/obj/cpp/h3_util.o" "$BUILD/obj/cpp/h3_management.o" \
  "$BUILD/libh3.a" -lm -o "$BUILD/h3_p0_audit.exe"

seeds=(101 202 303)
if [[ "${1:-full}" == "quick" ]]; then
  seeds=(101)
fi

for seed in "${seeds[@]}"; do
  # 4 km x 4 km, res 9, clustered, 20k entities, 20k queries
  ./"$BUILD/h3_p0_audit.exe" 20000 20000 0 9 \
    "$OUT/audit_4km_${seed}.csv" clustered 4096 "$seed" \
    > "$OUT/log_4km_${seed}.txt" 2>&1
  # 20 km x 20 km, res 10, uniform, 20k entities, 10k queries
  ./"$BUILD/h3_p0_audit.exe" 20000 10000 0 10 \
    "$OUT/audit_20km_${seed}.csv" uniform 20000 "$seed" \
    > "$OUT/log_20km_${seed}.txt" 2>&1
  echo "seed ${seed} done"
done

echo "--- 4 km ---"
for seed in "${seeds[@]}"; do tail -n +2 "$OUT/log_4km_${seed}.txt"; done
echo "--- 20 km ---"
for seed in "${seeds[@]}"; do tail -n +2 "$OUT/log_20km_${seed}.txt"; done
