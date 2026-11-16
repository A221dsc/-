#!/usr/bin/env bash
# Reproduce the H3-RTM execution-model pilot (B1).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

bash scripts/build_h3_p0.sh
mkdir -p results/h3_rtm

./build_h3_p0/h3_rtm_pilot.exe results/h3_rtm/pilot.csv 20000 5000 5000 \
  | tee results/h3_rtm/pilot.log

echo
echo "H3-RTM pilot artifacts:"
ls -lh results/h3_rtm/pilot.csv results/h3_rtm/pilot.log
