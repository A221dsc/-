"""Normalize original GPS samples into deterministic ID-disjoint CSV files."""
import argparse
import csv
import hashlib
import io
import json
import math
import tempfile
import zipfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path

FIELDS = ("entity_id", "timestamp", "x", "y", "event_type")
FORMAT = "%Y-%m-%d %H:%M:%S"
TDRIVE = "https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/"
PORTO = "https://doi.org/10.24432/C55W25"


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def json_file(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ecef(lon, lat):
    a, e2 = 6378137.0, 6.6943799901413165e-3
    longitude, latitude = math.radians(lon), math.radians(lat)
    s, c = math.sin(latitude), math.cos(latitude)
    n = a / math.sqrt(1 - e2 * s * s)
    return n*c*math.cos(longitude), n*c*math.sin(longitude), n*(1-e2)*s


def local_plane(origin):
    x0, y0, z0 = ecef(*origin)
    lon, lat = map(math.radians, origin)
    slon, clon, slat, clat = math.sin(lon), math.cos(lon), math.sin(lat), math.cos(lat)

    def project(longitude, latitude):
        x, y, z = ecef(longitude, latitude)
        dx, dy, dz = x-x0, y-y0, z-z0
        return -slon*dx+clon*dy, -slat*(clon*dx+slon*dy)+clat*dz

    return project


def split_id(taxi):
    digest = hashlib.sha256(f"v2-id-split-20261110:{taxi}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return "train" if bucket < 60 else "validation" if bucket < 80 else "test"


def coordinates(lon, lat):
    lon, lat = float(lon), float(lat)
    if not (-180 <= lon <= 180 and -90 <= lat <= 90) or (lon, lat) == (0, 0):
        raise ValueError("invalid or missing coordinates")
    return lon, lat


def write_normalized(groups, archive, output, dataset, origin, counts, notes):
    output.mkdir(parents=True, exist_ok=True)
    project = local_plane(origin)
    splits = ("train", "validation", "test")
    sizes = {s: Counter() for s in splits}
    extents = {s: [math.inf, math.inf, -math.inf, -math.inf] for s in splits}
    with ExitStack() as stack:
        writers = {}
        for split in splits:
            stream = stack.enter_context((output / f"{split}.csv").open("w", newline="", encoding="utf-8"))
            writers[split] = csv.writer(stream)
            writers[split].writerow(FIELDS)
        for taxi, rows in groups:
            if not rows:
                continue
            split = split_id(taxi)
            sizes[split]["entities"] += 1
            previous = None
            for stamp, point in sorted(rows.items()):
                x, y = project(*point)
                extent = extents[split]
                extent[:] = min(extent[0],x), min(extent[1],y), max(extent[2],x), max(extent[3],y)
                if previous is not None:
                    gap = int((datetime.fromisoformat(stamp)-datetime.fromisoformat(previous)).total_seconds())
                    if gap > 900:
                        counts["gaps_over_15min"] += 1
                writers[split].writerow((taxi,stamp,x,y,"INSERT" if previous is None else "UPDATE"))
                sizes[split]["records"] += 1
                previous = stamp
    json_file(output / "quality.json", {
        "counts": dict(counts), "splits": {s: dict(sizes[s]) for s in splits},
        "xy_bounds_m": {s: extents[s] if sizes[s] else None for s in splits},
    })
    json_file(output / "metadata.json", {
        "dataset": dataset, "official_source": TDRIVE if dataset == "TDRIVE" else PORTO,
        "archive": archive.name, "archive_sha256": sha(archive),
        "crs": "WGS84 ECEF projected onto a fixed local east/north tangent plane, height=0",
        "origin_lon_lat": origin, "xy_unit": "m", "duplicate_policy": "last valid input row per taxi/timestamp",
        "missing_policy": "drop invalid samples; no gap filling or interpolation",
        "split_policy": "stable salted SHA256 of taxi ID, buckets 60/20/20",
        "output_sha256": {s: sha(output / f"{s}.csv") for s in splits},
        **notes,
    })


def add_sample(rows, stamp, point, counts):
    if stamp in rows:
        counts["duplicate_exact" if rows[stamp] == point else "duplicate_conflict"] += 1
    rows[stamp] = point


def prepare_tdrive(archive, output):
    counts = Counter()
    with zipfile.ZipFile(archive) as z:
        by_taxi = defaultdict(list)
        for name in sorted(z.namelist()):
            p = Path(name)
            if p.suffix.lower() in {".txt", ".csv"} and p.stem.isdecimal():
                by_taxi[int(p.stem)].append(name)
        if not by_taxi:
            raise ValueError("T-Drive archive lacks numeric taxi files")

        def groups():
            for taxi, names in sorted(by_taxi.items()):
                rows = {}
                for name in names:
                    with z.open(name) as raw:
                        for fields in csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig")):
                            counts["input_rows"] += 1
                            try:
                                ident, stamp, lon, lat = fields
                                if int(ident) != taxi:
                                    raise ValueError("taxi ID does not match filename")
                                if datetime.strptime(stamp, FORMAT).strftime(FORMAT) != stamp:
                                    raise ValueError("noncanonical timestamp")
                                point = coordinates(lon, lat)
                            except ValueError:
                                counts["invalid"] += 1
                                continue
                            add_sample(rows, stamp, point, counts)
                yield taxi, rows

        write_normalized(groups(), archive, output, "TDRIVE", (116.4,39.9), counts, {
            "source_label": "REAL_TRAJECTORY", "time_reference": "source local wall-clock, interpreted as UTC+08 for epoch conversion",
            "timestamp_policy": "retain verbatim source timestamp",
        })


def prepare_porto(archive, output, start, end):
    """Select a fixed UTC window from the official complete-trip training archive."""
    counts = Counter()
    groups = defaultdict(dict)
    lower = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp())
    upper = int(datetime.fromisoformat(end).replace(tzinfo=timezone.utc).timestamp())
    if lower >= upper:
        raise ValueError("invalid Porto time window")
    with zipfile.ZipFile(archive) as outer, tempfile.TemporaryFile() as nested:
        from shutil import copyfileobj
        with outer.open("train.csv.zip") as zipped:
            copyfileobj(zipped, nested)
        with zipfile.ZipFile(nested) as train, train.open("train.csv") as file:
            reader = csv.DictReader(io.TextIOWrapper(file, encoding="utf-8-sig", newline=""))
            for row in reader:
                counts["trips_read"] += 1
                start_s = int(row["TIMESTAMP"])
                if start_s >= upper:
                    continue
                if row["MISSING_DATA"] != "False":
                    counts["trips_flagged_missing_before_end"] += 1
                    continue
                # A compact Porto polyline contains at least five characters per sample.
                # Skip trips too old to overlap even under this conservative upper bound.
                if start_s + 15*(len(row["POLYLINE"])//5) < lower:
                    continue
                positions = json.loads(row["POLYLINE"])
                taxi = int(row["TAXI_ID"])
                for i, pair in enumerate(positions):
                    stamp_s = start_s + 15*i
                    if not lower <= stamp_s < upper:
                        continue
                    counts["input_rows"] += 1
                    try:
                        point = coordinates(*pair)
                    except (ValueError, TypeError):
                        counts["invalid"] += 1
                        continue
                    stamp = datetime.fromtimestamp(stamp_s, timezone.utc).strftime(FORMAT)
                    add_sample(groups[taxi], stamp, point, counts)
    write_normalized(sorted(groups.items()), archive, output, "PORTO", (-8.61,41.15), counts, {
        "source_label": "REAL_TRAJECTORY", "time_reference": "UTC",
        "timestamp_policy": "published trip start epoch + 15*i seconds for original sampled points, no interpolation",
        "selection_window_utc": [start,end], "missing_trip_policy": "exclude all MISSING_DATA trips",
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["tdrive","porto"])
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", default="2013-07-03 06:00:00")
    parser.add_argument("--end", default="2013-07-03 09:00:00")
    args = parser.parse_args()
    if args.dataset == "tdrive":
        prepare_tdrive(args.archive,args.output)
    else:
        prepare_porto(args.archive,args.output,args.start,args.end)


if __name__ == "__main__":
    main()
