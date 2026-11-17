#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
out="${1:-results/h3_p0_v2}"
work="${H3_P0_V2_SAN_BUILD_ROOT:-build_h3_p0_v2}"
mkdir -p "$out"
for variant in tsan asan-ubsan; do
  build="$work/$variant"
  {
    uname -a
    H3_P0_V2_BUILD_DIR="$build" bash scripts/build_h3_p0_v2.sh "$variant"
    export TSAN_OPTIONS=halt_on_error=1:detect_deadlocks=1:second_deadlock_stack=1
    export ASAN_OPTIONS=halt_on_error=1:detect_leaks=1
    export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
    timeout 240s "$build/correctness" 180 16
    for scenario in A B C; do
      for method in A+Lock A+Static A+StaticMatched B+; do
        timeout 60s "$build/bench" "$scenario" "$method" 8 12011 4000 12 3 0 - 1
      done
    done
    echo "SANITIZER GATE PASS"
  } > "$out/$variant.log" 2>&1
done
