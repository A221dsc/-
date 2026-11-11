import sys
sys.path.insert(0, "scripts")
import csv
import json
from pathlib import Path

import numpy as np

from m4_scenario import actual_events, build_scenario, prepare, write_trace
from m4_maps import GameMap
from m4_actuals import actual_streams
from m4_loadstats import compare, compute, read_probe, read_trace

config = json.load(open("configs/m4.json"))
config = json.loads(json.dumps(config))
config["scenario"]["entity_count"] = 2000
config["steps"] = 300
config["phase_steps"] = 75
seeds = {"scenario": 4301, "objective": 4308, "route": 4314, "query": 4320, "actual": [5501, 5502, 5503, 5504, 5505]}
game, scenario, entities = build_scenario(config, "brc100d.map", "test", 0, seeds)
perturbation = dict(config["mismatch_profiles"]["medium"])
perturbation["exit_share"] = 0.2
perturbation["exit_step"] = 150
scenario["perturbation"] = perturbation
actual = actual_streams(scenario, seeds["actual"], perturbation)
prepared = prepare(scenario, game, entities, actual=actual)
execution = actual_events(scenario, game, prepared, actual)
trace = Path("results/m4_phase3/exit_smoke.m3bin")
trace.parent.mkdir(parents=True, exist_ok=True)
write_trace(trace, scenario["world"], execution["points"], execution["events"])
deletes = sum(1 for event in execution["events"] if event[0] == "D")
print("trace written, delete events:", deletes)

import subprocess
result = subprocess.run(["build_m3/spatial_m3.exe", "loadprobe", "160", "32", "0", str(trace),
                         "results/m4_phase3/exit_smoke_probe"], capture_output=True, text=True)
if result.returncode:
    raise SystemExit(result.stderr)
world, points, events = read_trace(trace)
offline = compute(world, points, events)
probe = read_probe("results/m4_phase3/exit_smoke_probe_loads_window.csv")
report = compare(offline, probe)
print("offline vs probe:", report)
assert report["match"], report
result = subprocess.run(["build_m3/spatial_m3.exe", "validate", "Fixed Regional Scan", "160", "32",
                         "1", "0", str(trace), "NONE"], capture_output=True, text=True)
print("validate scan:", result.stdout.strip(), result.stderr.strip())
print("deletion smoke test passed")
