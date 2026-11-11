"""Held-out comparison on the new, never-used test map (den011d)."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from prepare import json_file

config = json.load(open("configs/m4.json"))
selection = json.load(open("results/m4/calibration/selected.json"))
fine, quad = selection["fine_cell_m"], selection["quad_capacity"]
base = Path("results/m4/calibration/selector_final.txt")
cvar = Path("results/m4/phase3_m32/tmp/selector_3_1.00_1.txt")
quantile = Path("results/m4/phase3_m32/tmp/selector_2_0.50_1.txt")
for path in (cvar, quantile):
    if not path.exists():
        path.write_text(base.read_text(encoding="utf-8"), encoding="utf-8")
scen = Path("data/processed/m4/scenarios")
traces = Path("data/processed/m4/traces")
out = Path("results/m4/phase3/holdout")
out.mkdir(parents=True, exist_ok=True)
ident = "test-2-den011d"
methods = [
    ("Uniform Grid", 1, None, selection["classic"]["Uniform Grid"], None),
    ("PR Quadtree", 1, None, selection["classic"]["PR Quadtree"], None),
    ("Boost R*-Tree", 1, None, selection["classic"]["Boost R*-Tree"], None),
    ("Fixed Regional Scan", 1, None, 1, None),
    ("Fixed Local Fine Grid", 1, None, 1, None),
    ("Fixed Local Quadtree", quad, None, 1, None),
    ("Threshold-Based Adaptive Regional Index Selection", 1, base, 1, None),
    ("Cost-Aware Greedy Regional Index Selection", 1, base, 1, None),
    ("Nominal Point Feed-Forward Regional Index Selection", 1, base, 1, "nominal.csv"),
    ("Uncertainty-Aware Feed-Forward Regional Index Selection", 1, base, 1,
     "nominal_samples_m32.csv"),
    ("Quantile-Cost-Penalized Feed-Forward Regional Index Selection", 1, quantile, 1,
     "nominal_samples_m32.csv"),
    ("CVaR-Risk-Constrained Feed-Forward Regional Index Selection", 1, cvar, 1,
     "nominal_samples_m32.csv"),
]
rows = []
for actual in range(3):
    name = f"{ident}.actual.m3bin" if actual == 0 else f"{ident}.actual{actual}.m3bin"
    trace = traces / name
    for method, param, cfg, classic_param, prior_name in methods:
        param_value = classic_param if method in ("Uniform Grid", "PR Quadtree", "Boost R*-Tree",
                                                  "Fixed Local Quadtree") else param
        args = ["build_m3/spatial_m3.exe", "validate", method, str(fine), str(quad),
                str(param_value), "0", str(trace), str(cfg) if cfg else "NONE"]
        if prior_name:
            args.append(str(scen / f"{ident}.{prior_name}"))
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode:
            raise SystemExit(result.stderr)
        truth = dict(part.split("=") for part in result.stdout.strip().split())
        if int(truth["mismatches"]) != 0:
            raise SystemExit(f"mismatch {method}")
        args[1] = "measure"
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode:
            raise SystemExit(result.stderr)
        head = dict(token.split("=", 1) for token in result.stdout.strip().split())
        rows.append({"scenario": ident, "actual": actual, "method": method,
                     "online_ms": head["online_ms"], "build_ms": head["build_ms"],
                     "query_ms": head["query_ms"], "update_ms": head["update_ms"],
                     "policy_ms": head["policy_ms"], "p95_us": head["p95_us"],
                     "p99_us": head["p99_us"], "peak_rss_mb": head["peak_rss_mb"],
                     "switches": head["switches"], "promotions": head["promotions"],
                     "correctness_pass": True})
        print(f"a{actual} {method}: {head['online_ms']} ms", flush=True)
with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = __import__("csv").DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
json_file(out / "manifest.json", {"map": "den011d.map", "split": "test", "index": 2,
                                  "scenario": ident, "actuals": 3,
                                  "seeds": {"scenario": config["scenario_seeds"]["test"][2],
                                            "actual": config["actual_seeds"]["test"][2]},
                                  "frozen": True,
                                  "note": "new map never used in M1-M4 phase 1/2 studies"})
print("done")
