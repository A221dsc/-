"""Export an offline replay package for the wargame visualisation front end.

Each package is self-contained: world and region geometry, per-step phases and
actual query geometry, a sampled entity roster, per-window region loads with the
index mode each region used, applied switches and cumulative cost shares. The
export runs in its own process and never touches the timing path.
"""
import argparse
import csv
import gzip
import json
import struct
import subprocess
from pathlib import Path

HEADER = struct.Struct("<8s4d2Q")
POINT = struct.Struct("<2d")
EVENT = struct.Struct("<c7xq4d")
SIDE = 16
CATEGORIES = ["infantry", "vehicle", "supply"]
METHODS = {
    "scan": "Fixed Regional Scan",
    "quad": "Fixed Local Quadtree",
    "threshold": "Threshold-Based Adaptive Regional Index Selection",
    "threshold-scenario": "Threshold-Based Adaptive Regional Index Selection (Scenario)",
    "greedy": "Cost-Aware Greedy Regional Index Selection",
    "greedy-scenario": "Cost-Aware Greedy Regional Index Selection (Scenario)",
}


def read_trace(path):
    raw = Path(path).read_bytes()
    magic, *rest = HEADER.unpack_from(raw, 0)
    world = rest[:4]
    entities, count = rest[4:]
    offset = HEADER.size
    points = []
    for _ in range(entities):
        points.append(POINT.unpack_from(raw, offset))
        offset += POINT.size
    events = []
    for _ in range(count):
        kind, ident, a, b, c, d = EVENT.unpack_from(raw, offset)
        events.append((kind.decode(), ident, a, b, c, d))
        offset += EVENT.size
    return {"world": world, "points": points, "events": events, "entities": entities}


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("binary", type=Path)
    parser.add_argument("selector", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scenario", default="test-0-brc100d")
    parser.add_argument("--actual", type=int, default=0)
    parser.add_argument("--method", default="greedy-scenario")
    parser.add_argument("--entity-stride", type=int, default=25)
    parser.add_argument("--cpu", type=int, default=0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    ident = f"{args.scenario}.actual" + ("" if args.actual == 0 else str(args.actual))
    trace = read_trace(args.directory / "traces" / f"{ident}.m3bin")
    scenario = json.loads((args.directory / "scenarios" / f"{args.scenario}.json").read_text(encoding="utf-8"))
    selection = json.loads((Path("results/m4/calibration/selected.json")).read_text(encoding="utf-8"))
    method = METHODS[args.method]
    prior = args.directory / "scenarios" / f"{args.scenario}.nominal.csv"
    prefix = args.output / "profile" / f"{ident}-{args.method}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    use_prior = str(prior) if "Scenario" in method else "NONE"
    result = subprocess.run([str(args.binary), "profile", method,
                             str(selection["fine_cell_m"]), str(selection["quad_capacity"]), "1",
                             str(args.cpu), str(args.directory / "traces" / f"{ident}.m3bin"),
                             str(args.selector), str(prefix), use_prior],
                            capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(result.stderr.strip())
    regions = read_csv(prefix.parent / f"{prefix.name}_regions.csv")
    decisions = read_csv(prefix.parent / f"{prefix.name}_decisions.csv")
    timeline = {int(row["window_index"]): row
                for row in read_csv(prefix.parent / f"{prefix.name}_timeline.csv")}
    nominal = read_csv(args.directory / "scenarios" / f"{args.scenario}.nominal_queries.csv")

    world = trace["world"]
    queries, updates = {}, {}
    step = 0
    for kind, ident_value, a, b, c, d in trace["events"]:
        if kind == "P":
            step = ident_value
        elif kind == "U":
            updates.setdefault(step, []).append(int(ident_value))
        elif kind == "C":
            queries.setdefault(step, []).append({"kind": "C", "x": round(a, 2), "y": round(b, 2),
                                                "radius": round(c, 2)})
        else:
            queries.setdefault(step, []).append({"kind": "R", "x": round((a + c) / 2, 2),
                                                "y": round((b + d) / 2, 2),
                                                "half": round((c - a) / 2, 2)})
    phase_names = [p["name"] for p in scenario["phases"]]
    steps = []
    for step in range(scenario["steps"]):
        steps.append({"step": step,
                      "phase": phase_names[min(len(phase_names) - 1,
                                               step // scenario["phase_steps"])],
                      "queries": queries.get(step, [])})
    roster = [{"id": i + 1, "category": CATEGORIES[i % len(CATEGORIES)],
               "x": round(trace["points"][i][0], 2), "y": round(trace["points"][i][1], 2)}
              for i in range(0, len(trace["points"]), args.entity_stride)]
    windows = []
    cumulative = {"query": 0.0, "local_update": 0.0, "switch": 0.0, "policy": 0.0}
    for w in range(scenario["steps"] // scenario["decision_window"]):
        rows = [r for r in regions if int(r["window_index"]) == w]
        if not rows:
            continue
        cumulative["query"] += sum(float(r["query_ms"]) for r in rows)
        cumulative["local_update"] += sum(float(r["local_update_ms"]) for r in rows)
        cumulative["switch"] += sum(float(r["switch_ms"]) for r in rows)
        cumulative["policy"] += float(timeline.get(w, {}).get("policy_ms", 0.0))
        windows.append({
            "window": w, "step_end": (w + 1) * scenario["decision_window"],
            "regions": [{"id": int(r["region_id"]), "mode": r["mode"],
                         "entities": int(r["entities"]), "queries": int(r["query_count"]),
                         "updates": int(r["update_count"]),
                         "coverage": round(float(r["coverage"]), 4)} for r in rows],
            "switches": [{"region": int(d["region_id"]), "from": d["old_mode"],
                          "to": d["selected_mode"], "step": int(d["step"]),
                          "switch_ms": float(d["switch_cost_ms"])}
                         for d in decisions
                         if int(d["window_index"]) == w and d["status"] == "applied"],
            "cumulative_cost_ms": {key: round(value, 3) for key, value in cumulative.items()},
        })
    package = {
        "schema": "m4-replay-v1",
        "scenario": args.scenario, "actual": args.actual, "method": method,
        "world": list(world), "regions_side": SIDE,
        "region_width": (world[2] - world[0]) / SIDE,
        "region_height": (world[3] - world[1]) / SIDE,
        "phases": [{"name": p["name"], "steps": scenario["phase_steps"]} for p in scenario["phases"]],
        "map": scenario["map"],
        "entity_count": len(trace["points"]), "entity_stride": args.entity_stride,
        "entities_sample": roster,
        "steps": steps,
        "windows": windows,
        "nominal_queries_sample": nominal[:2000],
        "timing": "exported in a separate process; never part of an index timing run",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{args.scenario}.actual{args.actual}.{args.method}.replay.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(package, handle, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB), {len(steps)} steps, "
          f"{len(windows)} windows")


if __name__ == "__main__":
    main()
