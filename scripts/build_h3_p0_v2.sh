#!/usr/bin/env bash
# Shared native and Linux build, using the tracked H3 4.1.0 sources.
set -euo pipefail
cd "$(dirname "$0")/.."
variant="${1:-release}"
build="${H3_P0_V2_BUILD_DIR:-build_h3_p0_v2/$variant}"
cc="${CC:-gcc}"
cxx="${CXX:-g++}"
h3=third_party/h3-4.1.0
case "$variant" in
  release) flags=(-O3 -DNDEBUG) ;;
  tsan) flags=(-O1 -g -fno-omit-frame-pointer -fsanitize=thread) ;;
  asan-ubsan) flags=(-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all) ;;
  *) echo "expected release, tsan or asan-ubsan" >&2; exit 2 ;;
esac
case "$(uname -s)" in
  MINGW*|MSYS*) extension=.exe; platform=(-lpsapi) ;;
  *) extension=; platform=() ;;
esac
mkdir -p "$build/obj" "$build/include"
{
  find src/core src/h3 "$h3/src/h3lib" -type f
  printf '%s\n' "$h3/VERSION" scripts/build_h3_p0_v2.sh \
    tests/h3_p0_v2_correctness.cpp tests/h3_p0_v2_bench.cpp
} | LC_ALL=C sort | while IFS= read -r source; do
  printf '%s ' "$source"
  sed 's/\r$//' "$source" | sha256sum | cut -d' ' -f1
done > "$build/source.manifest"
sed -e 's/@H3_VERSION_MAJOR@/4/g' -e 's/@H3_VERSION_MINOR@/1/g' \
    -e 's/@H3_VERSION_PATCH@/0/g' "$h3/src/h3lib/include/h3api.h.in" > "$build/include/h3api.h"
includes=(-I"$build/include" -I"$h3/src/h3lib/include" -Isrc)
objs=()
for source in "$h3/src/h3lib/lib/"*.c; do
  object="$build/obj/$(basename "$source" .c).o"
  "$cc" -std=c99 "${flags[@]}" "${includes[@]}" -c "$source" -o "$object"
  objs+=("$object")
done
for name in geo local_projection h3_util h3_management h3_p0_v2; do
  object="$build/obj/$name.o"
  "$cxx" -std=c++20 -pthread -Wall -Wextra "${flags[@]}" "${includes[@]}" \
    -c "src/h3/$name.cpp" -o "$object"
  objs+=("$object")
done
for target in correctness bench; do
  "$cxx" -std=c++20 -pthread -Wall -Wextra "${flags[@]}" "${includes[@]}" \
    "tests/h3_p0_v2_$target.cpp" "${objs[@]}" "${platform[@]}" -lm \
    -o "$build/$target$extension"
done
sha256sum "$build/source.manifest" | cut -d' ' -f1 > "$build/source.sha256"
sha256sum "$build/bench$extension" | cut -d' ' -f1 > "$build/bench.sha256"
"$cxx" --version
printf 'Built %s with flags %s\n' "$build" "${flags[*]}"
printf 'SOURCE_FINGERPRINT %s\n' "$(cat "$build/source.sha256")"
