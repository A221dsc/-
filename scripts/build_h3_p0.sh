#!/usr/bin/env bash
# Build the fixed-resolution H3 management-partition prototype without CMake.
#
# The repository can live under a non-ASCII Windows path. GNU Make (used by the
# existing MinGW CMake generator) cannot stat absolute prerequisites containing
# that path, so this script compiles the small vendored H3 C library and the
# prototype with relative paths. It is dependency-free apart from
# gcc/g++/ar/python.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD="build_h3_p0"
H3_DIR="third_party/h3-4.1.0"
H3_INC="$H3_DIR/src/h3lib/include"
H3_LIB_DIR="$H3_DIR/src/h3lib/lib"
H3_API_OUT="$BUILD/include/h3api.h"

mkdir -p "$BUILD/obj/h3" "$BUILD/obj/cpp" "$BUILD/include"

if [[ ! -f "$H3_LIB_DIR/h3Index.c" ]]; then
  echo "missing vendored H3 source: $H3_LIB_DIR" >&2
  exit 1
fi

python - "$H3_DIR/src/h3lib/include/h3api.h.in" "$H3_API_OUT" "$H3_DIR/VERSION" <<'PY'
import sys
from pathlib import Path

template = Path(sys.argv[1]).read_text(encoding="utf-8")
version = Path(sys.argv[3]).read_text(encoding="utf-8").strip().split("-")[0]
major, minor, patch = (version.split(".") + ["0", "0", "0"])[:3]
template = template.replace("@H3_VERSION_MAJOR@", major)
template = template.replace("@H3_VERSION_MINOR@", minor)
template = template.replace("@H3_VERSION_PATCH@", patch)
Path(sys.argv[2]).write_text(template, encoding="utf-8")
PY

CFLAGS=(-O2 -std=c99 -I"$H3_INC" -I"$BUILD/include")
for source in "$H3_LIB_DIR"/*.c; do
  name="$(basename "$source" .c)"
  gcc "${CFLAGS[@]}" -c "$source" -o "$BUILD/obj/h3/$name.o"
done
ar rcs "$BUILD/libh3.a" "$BUILD/obj/h3"/*.o

CXXFLAGS=(-std=c++20 -O2 -Wall -Wextra
          -I. -Isrc -I"$H3_INC" -I"$BUILD/include")

for source in src/h3/geo.cpp src/h3/local_projection.cpp src/h3/h3_util.cpp \
              src/h3/h3_management.cpp; do
  name="$(basename "$source" .cpp)"
  g++ "${CXXFLAGS[@]}" -c "$source" -o "$BUILD/obj/cpp/$name.o"
done

g++ "${CXXFLAGS[@]}" tests/h3_management_tests.cpp \
  "$BUILD/obj/cpp/geo.o" "$BUILD/obj/cpp/local_projection.o" \
  "$BUILD/obj/cpp/h3_util.o" "$BUILD/obj/cpp/h3_management.o" \
  src/validation/oracle.cpp "$BUILD/libh3.a" -lm \
  -o "$BUILD/h3_management_tests.exe"

g++ "${CXXFLAGS[@]}" tests/h3_partition_bench.cpp \
  "$BUILD/obj/cpp/geo.o" "$BUILD/obj/cpp/local_projection.o" \
  "$BUILD/obj/cpp/h3_util.o" "$BUILD/obj/cpp/h3_management.o" \
  src/validation/oracle.cpp "$BUILD/libh3.a" -lm \
  -o "$BUILD/h3_partition_bench.exe"

echo "built $BUILD/h3_management_tests.exe and $BUILD/h3_partition_bench.exe"
