#!/usr/bin/env bash
# Native substitute gate used when TSan/ASan cannot run on the host.
#
# On this host: MinGW GCC 13.2 ships no libtsan/libasan/libubsan, and WSL is
# denied by the sandbox (Wsl/E_ACCESSDENIED). The historical TSan/ASan logs in
# results/h3_p0_v2 cover a different source fingerprint, so they do not certify
# the binary that produces the timings.
#
# This gate certifies the CURRENT source fingerprint instead:
#   - rebuilds the same translation units with -D_GLIBCXX_DEBUG
#     -D_GLIBCXX_ASSERTIONS (checked containers/iterators, a common victim of
#     concurrent mutation bugs) at -O1 -g;
#   - runs the frame-replay correctness suite at 8/16/32 workers, which
#     validates every query answer against exhaustive brute force and checks
#     slot uniqueness, bucket membership and active-point consistency;
#   - runs every method through the bench with VERIFY=1.
#
# It does NOT prove data-race freedom. That limitation is recorded in the gate
# log and in H3_P0_V2_DECISION.md.
set -euo pipefail
cd "$(dirname "$0")/.."

out="${1:-results/h3_p0_v2}"
build="${H3_P0_V2_NATIVE_BUILD_DIR:-build_h3_p0_v2/native}"
h3=third_party/h3-4.1.0
rust_dir="build_h3_p0_v2/release"
mkdir -p "$build/obj" "$build/include" "$out"

release_fp="$(cat "$rust_dir/source.sha256")"
{
  find src/core src/h3 "$h3/src/h3lib" -type f
  printf '%s\n' "$h3/VERSION" scripts/build_h3_p0_v2.sh \
    tests/h3_p0_v2_correctness.cpp tests/h3_p0_v2_bench.cpp
} | LC_ALL=C sort | while IFS= read -r source; do
  printf '%s ' "$source"
  sed 's/\r$//' "$source" | sha256sum | cut -d' ' -f1
done > "$build/source.manifest"
fp="$(sha256sum "$build/source.manifest" | cut -d' ' -f1)"

{
  echo "NATIVE DEBUG GATE (TSan/ASan unavailable on this host)"
  uname -a || true
  g++ --version | head -1
  if [[ "$fp" != "$release_fp" ]]; then
    echo "FAIL source fingerprint mismatch: gate=$fp release=$release_fp"
    exit 1
  fi
  echo "SOURCE_FINGERPRINT $fp"
  echo "FLAGS -O1 -g -D_GLIBCXX_DEBUG -D_GLIBCXX_ASSERTIONS"
  sed -e 's/@H3_VERSION_MAJOR@/4/g' -e 's/@H3_VERSION_MINOR@/1/g' \
      -e 's/@H3_VERSION_PATCH@/0/g' "$h3/src/h3lib/include/h3api.h.in" > "$build/include/h3api.h"
  includes=(-I"$build/include" -I"$h3/src/h3lib/include" -Isrc)
  flags=(-O1 -g -fno-omit-frame-pointer -D_GLIBCXX_DEBUG -D_GLIBCXX_ASSERTIONS)
  objs=()
  for source in "$h3/src/h3lib/lib/"*.c; do
    object="$build/obj/$(basename "$source" .c).o"
    gcc -std=c99 "${flags[@]}" "${includes[@]}" -c "$source" -o "$object"
    objs+=("$object")
  done
  for name in geo local_projection h3_util h3_management h3_p0_v2; do
    object="$build/obj/$name.o"
    g++ -std=c++20 -pthread -Wall -Wextra "${flags[@]}" "${includes[@]}" \
      -c "src/h3/$name.cpp" -o "$object"
    objs+=("$object")
  done
  for target in correctness bench; do
    g++ -std=c++20 -pthread -Wall -Wextra "${flags[@]}" "${includes[@]}" \
      "tests/h3_p0_v2_$target.cpp" "${objs[@]}" -lpsapi -lm \
      -o "$build/$target.exe"
  done
  echo "BUILD OK"
  frames="${H3_P0_V2_NATIVE_FRAMES:-600}"
  for workers in 8 16 32; do
    for repeat in 1 2; do
      echo "STRESS frames=$frames workers=$workers repeat=$repeat"
      "./$build/correctness.exe" "$frames" "$workers"
    done
  done
  for scenario in A B C; do
    for method in "A+Lock" "A+Static" "A+StaticMatched" "B+"; do
      "build_h3_p0_v2/release/bench.exe" "$scenario" "$method" 8 12011 4000 12 3 0 - 1 > /dev/null
      echo "VERIFY OK $scenario $method"
    done
  done
  echo "NATIVE GATE PASS"
} > "$out/native-gate.log" 2>&1

tail -3 "$out/native-gate.log"
