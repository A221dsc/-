"""Single-session full comparison: all methods on the frozen test traces."""
import csv
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
tmp = Path("results/m4/phase3_v2/tmp")
quantile = tmp / "selector_2_1.00_1.txt"
cvar = tmp / "selector_3_0.25_1.txt"
fusion = Path("results/m4/phase3/tmp/fusion_0.7.txt")
for path, fallback in ((quantile, base), (cvar, base)):
    if not path.exists():
        path.write_text(fallback.read_text(encoding="utf-8"), encoding="utf-8")
scen = Path("data/processed/m4/scenarios")
traces = Path("data/processed/m4/traces")
out = Path("results/m4/phase3/final")
out.mkdir(parents=True, exist_ok=True)
methods = [
    ("Uniform Grid", selection["classic"]["Uniform Grid"], None, None, 1),
    ("PR Quadtree", selection["classic"]["PR Quadtree"], None, None, 1),
    ("Boost R*-Tree", selection["classic"]["Boost R*-Tree"], None, None, 1),
    ("Fixed Regional Scan", 1, None, None, 1),
    ("Fixed Local Fine Grid", 1, None, None, fine),
    ("Fixed Local Quadtree", 1, None, None, quad),
    ("Threshold-Based Adaptive Regional Index Selection", 1, base, None, 1),
    ("Cost-Aware Greedy Regional Index Selection", 1, base, None, 1),
    ("Nominal Point Feed-Forward Regional Index Selection", 1, base, "nominal.csv", 1),
    ("Uncertainty-Aware Feed-Forward Regional Index Selection", 1, base,
     "nominal_samples_m32.csv", 1),
    ("Quantile-Cost-Penalized Feed-Forward Regional Index Selection", 1, quantile,
     "nominal_samples_m32.csv", 1),
    ("CVaR-Risk-Constrained Feed-Forward Regional Index Selection", 1, cvar,
     "nominal_samples_m32.csv", 1),
    ("Historical-Error-Feedback Robust Feed-Forward Regional Index Selection", 1, fusion,
     "nominal_samples_m32.csv", 1),
]
rows = []
for index in range(3):
    ident = f"test-{index}-{Path(config['maps']['test'][index]).stem}"
    for actual in range(3):
        trace = traces / (f"{ident}.actual.m3bin" if actual == 0 else f"{ident}.actual{actual}.m3bin")
        for method, param, cfg, prior_name, fixed_param in methods:
            args = ["build_m3/spatial_m3.exe", "measure", method, str(fine), str(quad),
                    str(fixed_param if method.startswith("Fixed Local") else param), "0",
                    str(trace), str(cfg) if cfg else "NONE"]
            if prior_name:
                args.append(str(scen / f"{ident}.{prior_name}"))
            result = subprocess.run(args, capture_output=True, text=True)
            if result.returncode:
                raise SystemExit(result.stderr)
            head = dict(token.split("=", 1) for token in result.stdout.strip().split())
            rows.append({"scenario": ident, "map": config["maps"]["test"][index],
                         "actual": actual, "method": method,
                         "online_ms": head["online_ms"], "build_ms": head["build_ms"],
                         "query_ms": head["query_ms"], "update_ms": head["update_ms"],
                         "switch_ms": head["switch_ms"], "policy_ms": head["policy_ms"],
                         "p95_us": head["p95_us"], "p99_us": head["p99_us"],
                         "peak_rss_mb": head["peak_rss_mb"], "switches": head["switches"],
                         "promotions": head["promotions"]})
            print(f"{ident} a{actual} {method}: {head['online_ms']} ms", flush=True)
with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
json_file(out / "manifest.json", {
    "maps": config["maps"]["test"], "actuals_per_map": 3,
    "configs": {"quantile": str(quantile), "cvar": str(cvar), "fusion": str(fusion)},
    "note": "single session; all methods measured on the same machine state",
    "validation_selection": "quantile lambda=1.0, cvar lambda=0.25, horizon=1 (frozen phase3_v2 tuning)"})
print("done")
