"""20% entity-exit disturbance experiment on the two test maps."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from m4_actuals import actual_streams
from m4_maps import GameMap
from m4_scenario import actual_events, build_scenario, prepare, sha, write_trace
from prepare import json_file

config = json.load(open("configs/m4.json"))
selection = json.load(open("results/m4/calibration/selected.json"))
fine, quad = selection["fine_cell_m"], selection["quad_capacity"]
base_cfg = Path("results/m4/calibration/selector_final.txt")
cvar_cfg = Path("results/m4/phase3_m32/tmp/selector_3_0.50_1.txt")
quantile_cfg = Path("results/m4/phase3_m32/tmp/selector_2_0.50_1.txt")
scen = Path("data/processed/m4/scenarios")
traces = Path("data/processed/m4/traces")
out = Path("results/m4/phase3/exits")
out.mkdir(parents=True, exist_ok=True)
methods = [
    ("Fixed Regional Scan", None, None, 1),
    ("Fixed Local Quadtree", None, None, quad),
    ("PR Quadtree", None, None, selection["classic"]["PR Quadtree"]),
    ("Uniform Grid", None, None, selection["classic"]["Uniform Grid"]),
    ("Cost-Aware Greedy Regional Index Selection", base_cfg, None, 1),
    ("Nominal Point Feed-Forward Regional Index Selection", base_cfg, "nominal.csv", 1),
    ("Uncertainty-Aware Feed-Forward Regional Index Selection", base_cfg,
     "nominal_samples_m32.csv", 1),
    ("Quantile-Cost-Penalized Feed-Forward Regional Index Selection", quantile_cfg,
     "nominal_samples_m32.csv", 1),
    ("CVaR-Risk-Constrained Feed-Forward Regional Index Selection", cvar_cfg,
     "nominal_samples_m32.csv", 1),
]
if not cvar_cfg.exists():
    cvar_cfg = base_cfg
if not quantile_cfg.exists():
    quantile_cfg = base_cfg

rows = []
for index, map_name in enumerate(config["maps"]["test"]):
    ident = f"test-{index}-{Path(map_name).stem}"
    scenario = json.load(open(scen / f"{ident}.json"))
    game = GameMap(Path("data/maps") / map_name)
    _, rebuilt, entities = build_scenario(config, map_name, "test", index, scenario["seeds"])
    assert rebuilt["scenario_sha256"] == scenario["scenario_sha256"]
    perturbation = dict(config["mismatch_profiles"]["medium"])
    perturbation["exit_share"] = 0.2
    perturbation["exit_step"] = 600
    scenario["perturbation"] = perturbation
    seeds = [seed + 900000 for seed in scenario["seeds"]["actual"]]
    actual = actual_streams(scenario, seeds, perturbation)
    prepared = prepare(scenario, game, entities, actual=actual)
    execution = actual_events(scenario, game, prepared, actual)
    trace = traces / f"{ident}.exit20.m3bin"
    write_trace(trace, scenario["world"], execution["points"], execution["events"])
    deletes = sum(1 for event in execution["events"] if event[0] == "D")
    print(f"{ident}: {deletes} delete events", flush=True)
    for method, cfg, prior_name, param in methods:
        args = ["build_m3/spatial_m3.exe", "validate", method, str(fine), str(quad), str(param),
                "0", str(trace), str(cfg) if cfg else "NONE"]
        if prior_name:
            args.append(str(scen / f"{ident}.{prior_name}"))
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode:
            raise SystemExit(result.stderr)
        truth = dict(part.split("=") for part in result.stdout.strip().split())
        if int(truth["mismatches"]) != 0:
            raise SystemExit(f"mismatch for {method}")
        args[1] = "measure"
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode:
            raise SystemExit(result.stderr)
        head = dict(token.split("=", 1) for token in result.stdout.strip().split())
        rows.append({"scenario": ident, "method": method,
                     "online_ms": head["online_ms"], "query_ms": head["query_ms"],
                     "update_ms": head["update_ms"], "switch_ms": head["switch_ms"],
                     "policy_ms": head["policy_ms"], "p95_us": head["p95_us"],
                     "p99_us": head["p99_us"], "switches": head["switches"],
                     "promotions": head["promotions"],
                     "delete_count": head.get("delete_count", ""),
                     "correctness_pass": True})
        print(f"  {method}: {head['online_ms']} ms", flush=True)
with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = __import__("csv").DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
json_file(out / "summary.json", {"traces": [r["scenario"] for r in rows][:0] or
                                 sorted({r["scenario"] for r in rows}),
                                 "methods": sorted({r["method"] for r in rows}),
                                 "exit_share": 0.2, "exit_step": 600})
print("done")
