"""Index-observed regional load semantics, reimplemented offline and verified.

The index counts
  * one query touch per region whose box intersects the query bounding box,
  * one update touch for the new region and, on a cross-region move, one more
    for the old region,
  * the region of the last reported position for entity counts,
  * coverage as the covered fraction of the region box.
This module reproduces those definitions independently and checks them against
the C++ load probe on every actual replay.
"""
import argparse
import csv
import json
import struct
import subprocess
from pathlib import Path

import numpy as np

HEADER = struct.Struct("<8s4d2Q")
POINT = struct.Struct("<2d")
EVENT = struct.Struct("<c7xq4d")
SIDE = 16
REGIONS = SIDE * SIDE


def read_trace(path):
    raw = Path(path).read_bytes()
    magic, *rest = HEADER.unpack_from(raw, 0)
    world = np.array(rest[:4], dtype=float)
    entities, count = rest[4:]
    offset = HEADER.size
    points = np.zeros((entities, 2))
    for i in range(entities):
        points[i] = POINT.unpack_from(raw, offset)
        offset += POINT.size
    events = []
    for _ in range(count):
        kind, ident, a, b, c, d = EVENT.unpack_from(raw, offset)
        events.append((kind.decode(), ident, a, b, c, d))
        offset += EVENT.size
    return world, points, events


def regions_of(points, world):
    x0, y0, x1, y1 = world
    bx = np.clip(((points[:, 0] - x0) / ((x1 - x0) / SIDE)).astype(np.int64), 0, SIDE - 1)
    by = np.clip(((points[:, 1] - y0) / ((y1 - y0) / SIDE)).astype(np.int64), 0, SIDE - 1)
    return SIDE * by + bx


def region_span(low, high, origin, span):
    a = int(np.clip((low - origin) / span, 0, SIDE - 1))
    b = int(np.clip((high - origin) / span, 0, SIDE - 1))
    return a, b


def compute(world, points, events, window=50):
    x0, y0, x1, y1 = world
    rw, rh = (x1 - x0) / SIDE, (y1 - y0) / SIDE
    area = rw * rh
    last = points.copy()
    last_region = regions_of(last, world)
    windows = sum(1 for event in events if event[0] == "P") + 1
    data = {"queries": np.zeros((windows, REGIONS)),
            "updates": np.zeros((windows, REGIONS)),
            "coverage": np.zeros((windows, REGIONS)),
            "entities": np.zeros((windows, REGIONS))}
    window = 0
    for kind, ident, a, b, c, d in events:
        if kind == "P":
            data["entities"][window] = np.bincount(last_region, minlength=REGIONS)
            window += 1
        elif kind == "U":
            new = np.array([[a, b]])
            new_region = int(regions_of(new, world)[0])
            old_region = int(last_region[ident - 1])
            data["updates"][window, new_region] += 1
            if new_region != old_region:
                data["updates"][window, old_region] += 1
            last[ident - 1] = (a, b)
            last_region[ident - 1] = new_region
        else:
            if kind == "C":
                left, right, bottom, top = a - c, a + c, b - c, b + c
            else:
                left, right, bottom, top = a, c, b, d
            bx0, bx1 = region_span(left, right, x0, rw)
            by0, by1 = region_span(bottom, top, y0, rh)
            for by in range(by0, by1 + 1):
                oy = max(0.0, min(y0 + (by + 1) * rh, top) - max(y0 + by * rh, bottom))
                for bx in range(bx0, bx1 + 1):
                    ox = max(0.0, min(x0 + (bx + 1) * rw, right) - max(x0 + bx * rw, left))
                    region = SIDE * by + bx
                    data["queries"][window, region] += 1
                    data["coverage"][window, region] += min(1.0, ox * oy / area)
    data["entities"][window] = np.bincount(last_region, minlength=REGIONS)
    return data


def read_probe(path):
    data = {"queries": np.zeros((0, REGIONS)), "updates": np.zeros((0, REGIONS)),
            "coverage": np.zeros((0, REGIONS)), "entities": np.zeros((0, REGIONS))}
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8")))
    windows = max(int(row["window"]) for row in rows) + 1
    out = {key: np.zeros((windows, REGIONS)) for key in data}
    for row in rows:
        w, r = int(row["window"]), int(row["region"])
        out["queries"][w, r] = float(row["query_count"])
        out["updates"][w, r] = float(row["update_count"])
        out["coverage"][w, r] = float(row["coverage"])
        out["entities"][w, r] = float(row["entity_count"])
    return out


def compare(offline, probe, tolerance=1e-4):
    report = {}
    for key in ("queries", "updates", "entities"):
        diff = float(np.abs(offline[key] - probe[key]).sum())
        report[f"{key}_abs_diff"] = diff
        report[f"{key}_match"] = diff == 0
    coverage_diff = float(np.abs(offline["coverage"] - probe["coverage"]).max())
    report["coverage_max_abs_diff"] = coverage_diff
    report["coverage_match"] = coverage_diff <= tolerance
    report["match"] = (report["queries_match"] and report["updates_match"] and
                       report["entities_match"] and report["coverage_match"])
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--traces", nargs="*")
    parser.add_argument("--cpu", type=int, default=0)
    args = parser.parse_args()
    traces = sorted((args.directory / "traces").glob("*.actual*.m3bin"))
    if args.traces:
        traces = [path for path in traces if any(name in path.name for name in args.traces)]
    audit_dir = args.directory / "audit"
    loads_dir = args.directory / "loads"
    legacy_dir = args.directory / "loads_legacy_python_semantics"
    audit_dir.mkdir(exist_ok=True)
    legacy_dir.mkdir(exist_ok=True)
    records = []
    for trace in traces:
        ident = trace.name[:-len(".m3bin")]
        probe_prefix = audit_dir / f"{ident}.probe"
        result = subprocess.run([str(args.binary), "loadprobe", "160", "32", str(args.cpu),
                                 str(trace), str(probe_prefix)], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        world, points, events = read_trace(trace)
        offline = compute(world, points, events)
        probe = read_probe(probe_prefix.parent / f"{probe_prefix.name}_loads_window.csv")
        report = compare(offline, probe)
        report["trace"] = trace.name
        records.append(report)
        print(f"{ident}: match={report['match']} qdiff={report['queries_abs_diff']:.0f} "
              f"udiff={report['updates_abs_diff']:.0f} ndiff={report['entities_abs_diff']:.0f} "
              f"cov={report['coverage_max_abs_diff']:.2e}", flush=True)
        # Preserve the previous Python labels once, then replace with index-observed labels.
        old_load = loads_dir / f"{ident}.loads.csv"
        if old_load.exists():
            legacy = legacy_dir / f"{ident}.loads.csv"
            if not legacy.exists():
                legacy.write_bytes(old_load.read_bytes())
        with (loads_dir / f"{ident}.loads.csv").open("w", encoding="utf-8", newline="") as handle:
            handle.write("window,region,query_count,entity_count,update_count,coverage\n")
            windows = probe["queries"].shape[0]
            for w in range(windows):
                for region in range(REGIONS):
                    handle.write(f"{w},{region},{int(probe['queries'][w, region])},"
                                 f"{int(probe['entities'][w, region])},"
                                 f"{int(probe['updates'][w, region])},"
                                 f"{probe['coverage'][w, region]:.6f}\n")
    summary = {"semantics": "index-observed (C++ load probe)",
               "acceptance": "offline statistician equals the online index exactly",
               "traces": len(records),
               "all_match": all(record["match"] for record in records),
               "records": records,
               "legacy_note": "loads_legacy_python_semantics keeps the previous labels: "
                              "query centres instead of touches, circles with the box half "
                              "instead of the radius, no old-region update touch"}
    (audit_dir / "load_semantics_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("all_match:", summary["all_match"])


if __name__ == "__main__":
    main()
