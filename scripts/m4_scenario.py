"""Scenario-driven nominal derivation and perturbed actual execution.

The scenario JSON plus the entity table contain only planned data: start
positions, categories, planned objective sequence, speeds, phase schedule and
periodic query configuration. The nominal deriver reads nothing else and is
therefore independent of the actual disturbance seed. The actual executor uses a
separate seed stream to inject departure delays, speed noise, position noise,
route detours and phase shifts, and writes the replay that every index replays.
"""
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np

from m4_maps import GameMap

HEADER = struct.Struct("<8s4d2Q")
POINT = struct.Struct("<2d")
EVENT = struct.Struct("<c7xq4d")
SIDE = 16
CATEGORY_SPEED = {"infantry": 6.0, "vehicle": 14.0, "supply": 3.0}
CATEGORY_SHARE = {"infantry": 0.60, "vehicle": 0.30, "supply": 0.10}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def region_array(points, bounds):
    x0, y0, x1, y1 = bounds
    w, h = (x1 - x0) / SIDE, (y1 - y0) / SIDE
    bx = np.clip(((points[:, 0] - x0) / w).astype(np.int64), 0, SIDE - 1)
    by = np.clip(((points[:, 1] - y0) / h).astype(np.int64), 0, SIDE - 1)
    return SIDE * by + bx


def region_of(point, bounds):
    return int(region_array(np.asarray([point], dtype=float), bounds)[0])


def region_box(region, bounds):
    x0, y0, x1, y1 = bounds
    w, h = (x1 - x0) / SIDE, (y1 - y0) / SIDE
    bx, by = region % SIDE, region // SIDE
    return (x0 + bx * w, y0 + by * h, x0 + (bx + 1) * w, y0 + (by + 1) * h)


def build_polyline(game, start_cell, targets):
    points = []
    holds = []
    cursor = start_cell
    for target in targets:
        cells = game.route_cells(cursor, target)
        segment = np.array([game.cell_center(c) for c in cells], dtype=float)
        if points:
            segment = segment[1:]
        points.append(segment)
        holds.append(sum(len(part) for part in points) - 1)
        cursor = target
    pts = np.vstack(points)
    deltas = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(deltas)])
    hold_distance = [float(cum[min(h, len(cum) - 1)]) for h in holds]
    return pts, cum, hold_distance


def interpolate(points, cum, distance):
    distance = np.clip(distance, 0.0, cum[-1])
    segment = np.searchsorted(cum, distance, side="right") - 1
    segment = np.clip(segment, 0, len(cum) - 2)
    span = cum[segment + 1] - cum[segment]
    frac = np.where(span > 0, (distance - cum[segment]) / np.maximum(span, 1e-9), 0.0)
    return points[segment] + frac[:, None] * (points[segment + 1] - points[segment])


def build_scenario(config, map_name, split, index, seeds):
    game = GameMap(Path("data/maps") / map_name)
    spec = config["scenario"]
    objectives = game.objective_cells(np.random.default_rng([seeds["objective"], index]),
                                      len(spec["objectives"]))
    rng = np.random.default_rng([seeds["scenario"], index])
    assemblies = []
    for obj in objectives:
        placed = 0
        for _attempt in range(400):
            if placed >= 2:
                break
            cx = obj[0] + int(rng.integers(-6, 7))
            cy = obj[1] + int(rng.integers(-6, 7))
            if 0 <= cx < game.width and 0 <= cy < game.height and game.walkable[cy, cx]:
                assemblies.append((cx, cy))
                placed += 1
        while placed < 2:
            assemblies.append(obj)
            placed += 1
    count = spec["entity_count"]
    categories = list(CATEGORY_SHARE)
    shares = np.array([CATEGORY_SHARE[c] for c in categories], dtype=float)
    categories_pick = rng.choice(len(categories), size=count, p=shares / shares.sum())
    assembly_pick = rng.integers(0, len(assemblies), size=count)
    main_group = rng.random(count) < spec["main_group_share"]
    entities = []
    for i in range(count):
        category = categories[int(categories_pick[i])]
        assembly = int(assembly_pick[i])
        home = f"obj{assembly % len(objectives)}"
        plan = ([spec["phase_targets"][k] if main_group[i] else home
                 for k in range(len(spec["phase_targets"]))])
        entities.append({"entity_id": i + 1, "category": category,
                         "start_cell": [int(assemblies[assembly][0]), int(assemblies[assembly][1])],
                         "assembly": assembly,
                         "group": "main" if main_group[i] else "reserve",
                         "objectives": plan,
                         "speed_mps": round(float(CATEGORY_SPEED[category] * rng.uniform(0.9, 1.1)), 3),
                         "offset_cells": [float(rng.uniform(-1.6, 1.6)), float(rng.uniform(-1.6, 1.6))]})
    scenario = {
        "scenario_id": f"{Path(map_name).stem}-{split}-{index}",
        "dataset_variant": "PUBLIC_MAP_SCENARIO",
        "map": game.manifest(),
        "world": list(game.bounds),
        "steps": config["steps"], "phase_steps": config["phase_steps"],
        "decision_window": config["decision_window"],
        "forecast_horizon": config["forecast_horizon"],
        "entity_count": count,
        "objectives": [{"id": f"obj{i}", "cell": [int(c[0]), int(c[1])],
                        "world": list(game.cell_center(c))} for i, c in enumerate(objectives)],
        "assemblies": [{"cell": [int(c[0]), int(c[1])], "world": list(game.cell_center(c))}
                       for c in assemblies],
        "phases": config["phases"],
        "queries": config["queries"],
        "main_group_share": spec["main_group_share"],
        "seeds": seeds,
        "entity_table": "entities.csv",
    }
    scenario["scenario_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in scenario.items() if k != "seeds"}, sort_keys=True).encode()).hexdigest()
    return game, scenario, entities


def prepare(scenario, game, entities, actual=None):
    """Precompute route polylines and per-entity state; actual adds disturbances."""
    phases = scenario["phases"]
    objective_cell = {o["id"]: tuple(o["cell"]) for o in scenario["objectives"]}
    focus = [objective_cell[p.get("focus", "obj0")] for p in phases]
    polylines = {}
    for assembly_id, assembly in enumerate(scenario["assemblies"]):
        start = tuple(assembly["cell"])
        home = objective_cell[f"obj{assembly_id % len(scenario['objectives'])}"]
        for group, targets in (("main", focus), ("reserve", [home] * len(phases))):
            polylines[(assembly_id, group, "std")] = build_polyline(game, start, targets)
        polylines[(assembly_id, "main", "detour")] = build_polyline(
            game, start, [focus[0], focus[0], focus[2] if len(focus) > 2 else focus[0],
                          focus[-1]])
    poly_keys = list(polylines)
    poly_index = {key: i for i, key in enumerate(poly_keys)}
    count = len(entities)
    entity_poly = np.zeros(count, dtype=np.int32)
    for i, entity in enumerate(entities):
        variant = "std"
        if actual is not None and entity["group"] == "main":
            if actual["detour_rng"].random() < scenario["perturbation"]["detour_share"]:
                variant = "detour"
        entity_poly[i] = poly_index[(entity["assembly"], entity["group"], variant)]
    speed = np.array([e["speed_mps"] for e in entities], dtype=float)
    offset = np.array([e["offset_cells"] for e in entities], dtype=float) * game.cell_m
    delay = np.zeros(count, dtype=np.int32)
    if actual is not None:
        speed = speed * (1.0 + actual["speed_rng"].normal(0.0, scenario["perturbation"]["speed_sigma"], count))
        delay = actual["delay_rng"].integers(0, scenario["perturbation"]["delay_steps_max"] + 1, count)
        speed = np.maximum(speed, 0.5)
    groups = {}
    hold = np.zeros((count, len(phases)), dtype=float)
    for poly_id in np.unique(entity_poly):
        points, cum, holds = polylines[poly_keys[int(poly_id)]]
        members = np.flatnonzero(entity_poly == poly_id)
        groups[int(poly_id)] = (points, cum, np.array(holds), members)
        hold[members] = np.array(holds)[None, :]
    return {"groups": groups, "speed": speed, "offset": offset, "delay": delay,
            "hold": hold, "count": count}


def simulate(scenario, game, prepared, actual=None, reporting=False):
    """Advance every entity; returns positions and window-end load counters.

    With ``reporting`` the nominal run also simulates the periodic position
    reports with an independent nominal seed, so entity counts refer to the last
    reported state and update touches include the old region on cross-region
    moves, exactly like the index observes them.
    """
    steps = scenario["steps"]
    window = scenario["decision_window"]
    windows = steps // window
    phases = scenario["phases"]
    speed_factor = [p.get("speed_factor", 1.0) for p in phases]
    if actual is None:
        boundaries = [scenario["phase_steps"] * k for k in range(1, len(phases))]
    else:
        shift = np.cumsum([0] + [int(s) for s in actual["phase_shift"]])
        boundaries = [max(1, min(steps - 1, scenario["phase_steps"] * k + int(shift[k])))
                      for k in range(1, len(phases))]
        boundaries = sorted(boundaries)
    phase_at = np.searchsorted(boundaries, np.arange(steps), side="right").astype(np.int32)
    speed, offset, delay = prepared["speed"], prepared["offset"], prepared["delay"]
    groups = prepared["groups"]
    hold = prepared["hold"]
    distance = np.zeros(prepared["count"], dtype=float)
    positions = np.zeros((prepared["count"], 2), dtype=float)
    for poly_id, (route, cum, holds, members) in groups.items():
        positions[members] = interpolate(route, cum, np.zeros(len(members))) + offset[members]
    initial = positions.copy()
    counts = np.zeros((windows, SIDE * SIDE), dtype=np.int64)
    updates = np.zeros((windows, SIDE * SIDE), dtype=np.int64)
    entities = np.zeros((windows, SIDE * SIDE), dtype=np.int64)
    bounds = scenario["world"]
    report_rng = np.random.default_rng(scenario["seeds"]["query"] + 104729)
    reported_region_ids = region_array(initial, bounds)
    for step in range(steps):
        phase = int(phase_at[step])
        moving = delay <= step
        for poly_id, (route, cum, holds, members) in groups.items():
            target = np.minimum(distance[members] +
                                np.where(moving[members], speed[members] * speed_factor[phase], 0.0),
                                hold[members, phase])
            distance[members] = target
            positions[members] = interpolate(route, cum, distance[members]) + offset[members]
        w = min(windows - 1, step // window)
        if reporting:
            rate = min(phases[phase]["update_rate"], prepared["count"])
            chosen = report_rng.choice(prepared["count"], size=rate, replace=False)
            new_region = region_array(positions[chosen], bounds)
            old_region = reported_region_ids[chosen]
            updates[w] += np.bincount(new_region, minlength=SIDE * SIDE)
            moved = new_region != old_region
            if moved.any():
                updates[w] += np.bincount(old_region[moved], minlength=SIDE * SIDE)
            reported_region_ids[chosen] = new_region
        if (step + 1) % window == 0:
            counts[(step + 1) // window - 1] = np.bincount(
                region_array(positions, bounds), minlength=SIDE * SIDE)
            if reporting:
                entities[(step + 1) // window - 1] = np.bincount(
                    reported_region_ids, minlength=SIDE * SIDE)
    return {"initial": initial, "final": positions.copy(), "window_counts": counts,
            "window_updates": updates, "window_entities": entities, "phase_at": phase_at}


def planned_query_schedule(scenario, game):
    steps = scenario["steps"]
    phase_steps = scenario["phase_steps"]
    phases = scenario["phases"]
    queries = scenario["queries"]
    rng = np.random.default_rng(scenario["seeds"]["query"])
    objective_world = {o["id"]: np.array(o["world"]) for o in scenario["objectives"]}
    walkable = np.argwhere(game.walkable)
    cells = np.array([game.cell_center((int(c[1]), int(c[0]))) for c in walkable])
    schedule = []
    for step in range(steps):
        phase = min(len(phases) - 1, step // phase_steps)
        spec = phases[phase]
        count = spec["queries_per_step"]
        hot = round(count * spec["hot_share"])
        focus_id = spec["focus"]
        if "focus_cycle" in spec:
            cycle = spec.get("cycle_steps", 100)
            options = spec["focus_cycle"]
            focus_id = options[(step % cycle) * len(options) // cycle]
        centre = objective_world[focus_id]
        hot_indices = set(rng.choice(count, size=hot, replace=False).tolist())
        for j in range(count):
            half = float(rng.choice(queries["half_sizes_m"]))
            kind = "R" if rng.random() >= queries["circle_share"] else "C"
            if j in hot_indices:
                point = (float(centre[0] + rng.normal(0, queries["sigma_m"])),
                         float(centre[1] + rng.normal(0, queries["sigma_m"])))
                tag = focus_id
            else:
                point = tuple(float(v) for v in cells[int(rng.integers(len(cells)))])
                tag = "background"
            box_half = half if kind == "R" else 2.0 * half / math.sqrt(math.pi)
            schedule.append((step, kind, point, box_half, tag))
    return schedule


def _normal_mass(low, high, mean, sigma):
    return 0.5 * (math.erf((high - mean) / (sigma * math.sqrt(2))) -
                  math.erf((low - mean) / (sigma * math.sqrt(2))))


def _normal_integral(low, high, a, sigma):
    """Integral over [low, high] of Phi((x - a) / sigma) dx."""
    def primitive(x):
        u = (x - a) / sigma
        return sigma * (u * 0.5 * (1.0 + math.erf(u / math.sqrt(2))) +
                        math.exp(-0.5 * u * u) / math.sqrt(2 * math.pi))
    return primitive(high) - primitive(low)


def _expected_overlap(low, high, centre, sigma, half):
    """E[ length of the intersection of [centre-half, centre+half] with [low, high] ]."""
    return (_normal_integral(low, high, centre - half, sigma) -
            _normal_integral(low, high, centre + half, sigma))


def query_distributions(schedule, game, scenario, windows):
    """Expected query touches, query centres and covered fraction per region.

    A query touches every region whose box intersects the query bounding box, so
    the touch expectation integrates the centre distribution over the region
    expanded by the box half width. Coverage is the expected covered fraction of
    the region box, which is separable and therefore analytic.
    """
    bounds = scenario["world"]
    x0, y0, x1, y1 = bounds
    sigma = scenario["queries"]["sigma_m"]
    rw, rh = (x1 - x0) / SIDE, (y1 - y0) / SIDE
    region_area = rw * rh
    region_walkable = np.zeros(SIDE * SIDE)
    for by in range(SIDE):
        for bx in range(SIDE):
            rx0, ry0 = x0 + bx * rw, y0 + by * rh
            c0, r0 = game.world_to_cell((rx0, ry0))
            c1, r1 = game.world_to_cell((rx0 + rw, ry0 + rh))
            region_walkable[SIDE * by + bx] = game.walkable[r0:r1 + 1, c0:c1 + 1].sum()
    region_walkable = np.maximum(region_walkable, 1.0)
    background = region_walkable / max(1.0, float(game.walkable_count))
    touch = np.zeros((windows, SIDE * SIDE))
    centre_counts = np.zeros((windows, SIDE * SIDE))
    coverage = np.zeros((windows, SIDE * SIDE))
    window = scenario["decision_window"]
    by_step = {}
    for item in schedule:
        by_step.setdefault(item[0] // window, []).append(item)
    for w in range(windows):
        for (step, kind, point, half, tag) in by_step.get(w, []):
            region = region_of(point, bounds)
            if tag == "background":
                touch[w] += background
                coverage[w, region] += min(1.0, (2 * half) ** 2 / region_area)
                centre_counts[w] += background
                continue
            centre_counts[w] += _centre_distribution(point, sigma, bounds)
            bx_lo = max(0, int((point[0] - half - 4 * sigma - x0) / rw))
            bx_hi = min(SIDE - 1, int((point[0] + half + 4 * sigma - x0) / rw))
            by_lo = max(0, int((point[1] - half - 4 * sigma - y0) / rh))
            by_hi = min(SIDE - 1, int((point[1] + half + 4 * sigma - y0) / rh))
            for by in range(by_lo, by_hi + 1):
                ry0, ry1 = y0 + by * rh, y0 + (by + 1) * rh
                touch_y = _normal_mass(ry0 - half, ry1 + half, point[1], sigma)
                if touch_y <= 0:
                    continue
                overlap_y = max(0.0, _expected_overlap(ry0, ry1, point[1], sigma, half))
                for bx in range(bx_lo, bx_hi + 1):
                    rx0, rx1 = x0 + bx * rw, x0 + (bx + 1) * rw
                    region = SIDE * by + bx
                    touch[w, region] += touch_y * _normal_mass(rx0 - half, rx1 + half,
                                                               point[0], sigma)
                    if overlap_y > 0:
                        overlap_x = max(0.0, _expected_overlap(rx0, rx1, point[0], sigma, half))
                        coverage[w, region] += min(1.0, overlap_x * overlap_y / region_area)
    return touch, centre_counts, coverage


def _centre_distribution(point, sigma, bounds):
    x0, y0, x1, y1 = bounds
    rw, rh = (x1 - x0) / SIDE, (y1 - y0) / SIDE
    out = np.zeros(SIDE * SIDE)
    bx_lo = max(0, int((point[0] - 4 * sigma - x0) / rw))
    bx_hi = min(SIDE - 1, int((point[0] + 4 * sigma - x0) / rw))
    by_lo = max(0, int((point[1] - 4 * sigma - y0) / rh))
    by_hi = min(SIDE - 1, int((point[1] + 4 * sigma - y0) / rh))
    for by in range(by_lo, by_hi + 1):
        py = _normal_mass(y0 + by * rh, y0 + (by + 1) * rh, point[1], sigma)
        for bx in range(bx_lo, bx_hi + 1):
            px = _normal_mass(x0 + bx * rw, x0 + (bx + 1) * rw, point[0], sigma)
            out[SIDE * by + bx] += px * py
    return out


def nominal_summary(scenario, game, prepared):
    windows = scenario["steps"] // scenario["decision_window"]
    schedule = planned_query_schedule(scenario, game)
    touch, centres, coverage = query_distributions(schedule, game, scenario, windows)
    state = simulate(scenario, game, prepared, reporting=True)
    return {"queries": touch, "centres": centres,
            "entities": state["window_entities"].astype(float),
            "entities_true": state["window_counts"].astype(float),
            "updates": state["window_updates"].astype(float),
            "coverage": coverage, "phase_at": state["phase_at"],
            "initial": state["initial"], "schedule": schedule}


def write_nominal_table(path, summary):
    windows = summary["queries"].shape[0]
    lines = ["window,region,query_count,entity_count,update_count,coverage,query_centre"]
    for w in range(windows):
        for region in range(SIDE * SIDE):
            lines.append(f"{w},{region},{summary['queries'][w, region]:.6f},"
                         f"{int(summary['entities'][w, region])},"
                         f"{summary['updates'][w, region]:.6f},"
                         f"{summary['coverage'][w, region]:.6f},"
                         f"{summary['centres'][w, region]:.6f}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def actual_events(scenario, game, prepared, actual):
    """Perturbed execution: reported positions, queries, phase shifts and loads."""
    steps = scenario["steps"]
    window = scenario["decision_window"]
    windows = steps // window
    phases = scenario["phases"]
    queries = scenario["queries"]
    perturbation = scenario["perturbation"]
    bounds = scenario["world"]
    x0, y0, x1, y1 = bounds
    rw, rh = (x1 - x0) / SIDE, (y1 - y0) / SIDE
    region_area = rw * rh
    state = simulate(scenario, game, prepared, actual=actual)
    phase_at = state["phase_at"]
    objective_world = {o["id"]: np.array(o["world"]) for o in scenario["objectives"]}
    walkable = np.argwhere(game.walkable)
    cells = np.array([game.cell_center((int(c[1]), int(c[0]))) for c in walkable])
    rng = actual["query_rng"]
    offset = actual["focus_offset"]
    initial_points = state["initial"] + actual["noise_rng"].normal(
        0.0, perturbation["report_noise_m"], state["initial"].shape)
    reported = initial_points.copy()
    events = []
    query_counts = np.zeros((windows, SIDE * SIDE))
    update_counts = np.zeros((windows, SIDE * SIDE))
    entity_counts = np.zeros((windows, SIDE * SIDE))
    coverage = np.zeros((windows, SIDE * SIDE))
    speed, delay = prepared["speed"], prepared["delay"]
    groups = prepared["groups"]
    hold = prepared["hold"]
    speed_factor = [p.get("speed_factor", 1.0) for p in phases]
    distance = np.zeros(prepared["count"], dtype=float)
    positions = state["initial"].copy()
    for step in range(steps):
        phase = int(phase_at[step])
        w = min(windows - 1, step // window)
        moving = delay <= step
        for poly_id, (route, cum, holds, members) in groups.items():
            target = np.minimum(distance[members] +
                                np.where(moving[members], speed[members] * speed_factor[phase], 0.0),
                                hold[members, phase])
            distance[members] = target
            positions[members] = interpolate(route, cum, distance[members]) + prepared["offset"][members]
        if step and step % window == 0:
            events.append(("P", step, 0.0, 0.0, 0.0, 0.0))
        spec = phases[phase]
        rate = min(spec["update_rate"], prepared["count"])
        chosen = rng.choice(prepared["count"], size=rate, replace=False)
        reported[chosen] = positions[chosen] + rng.normal(0.0, perturbation["report_noise_m"],
                                                          (rate, 2))
        update_counts[w] += np.bincount(region_array(reported[chosen], bounds),
                                        minlength=SIDE * SIDE)
        for index in chosen:
            events.append(("U", int(index) + 1, float(reported[index, 0]),
                           float(reported[index, 1]), 0.0, 0.0))
        count = spec["queries_per_step"]
        hot = round(count * spec["hot_share"])
        focus_id = spec["focus"]
        if "focus_cycle" in spec:
            cycle = spec.get("cycle_steps", 100)
            options = spec["focus_cycle"]
            focus_id = options[(step % cycle) * len(options) // cycle]
        centre = objective_world[focus_id] + offset[phase]
        hot_indices = set(rng.choice(count, size=hot, replace=False).tolist())
        for j in range(count):
            half = float(rng.choice(queries["half_sizes_m"]))
            kind = "R" if rng.random() >= queries["circle_share"] else "C"
            if j in hot_indices:
                point = (float(centre[0] + rng.normal(0, queries["sigma_m"] * perturbation["query_sigma_factor"]) +
                               rng.normal(0, perturbation["query_jitter_m"])),
                         float(centre[1] + rng.normal(0, queries["sigma_m"] * perturbation["query_sigma_factor"]) +
                               rng.normal(0, perturbation["query_jitter_m"])))
            else:
                point = tuple(float(v) for v in cells[int(rng.integers(len(cells)))])
            if kind == "C":
                radius = 2 * half / math.sqrt(math.pi)
                events.append(("C", 0, point[0], point[1], radius, 0.0))
            else:
                events.append(("R", 0, point[0] - half, point[1] - half,
                               point[0] + half, point[1] + half))
            bx_lo = max(0, int((point[0] - half - x0) / rw))
            bx_hi = min(SIDE - 1, int((point[0] + half - x0) / rw))
            by_lo = max(0, int((point[1] - half - y0) / rh))
            by_hi = min(SIDE - 1, int((point[1] + half - y0) / rh))
            for by in range(by_lo, by_hi + 1):
                oy = max(0.0, min(y0 + (by + 1) * rh, point[1] + half) -
                         max(y0 + by * rh, point[1] - half))
                if oy <= 0:
                    continue
                for bx in range(bx_lo, bx_hi + 1):
                    ox = max(0.0, min(x0 + (bx + 1) * rw, point[0] + half) -
                             max(x0 + bx * rw, point[0] - half))
                    if ox <= 0:
                        continue
                    region = SIDE * by + bx
                    query_counts[w, region] += 1
                    coverage[w, region] += min(1.0, ox * oy / region_area)
        if (step + 1) % window == 0:
            entity_counts[(step + 1) // window - 1] = np.bincount(
                region_array(positions, bounds), minlength=SIDE * SIDE)
    loads = {"queries": query_counts, "entities": entity_counts,
             "updates": update_counts, "coverage": coverage}
    return {"points": initial_points, "events": events, "actual_loads": loads,
            "phase_at": phase_at}


def write_trace(path, bounds, points, events):
    with Path(path).open("wb", buffering=1 << 20) as handle:
        handle.write(HEADER.pack(b"M3BIN001", *bounds, len(points), len(events)))
        for point in points:
            handle.write(POINT.pack(float(point[0]), float(point[1])))
        for event in events:
            kind = event[0]
            handle.write(EVENT.pack(kind.encode(), int(event[1]), float(event[2]), float(event[3]),
                                    float(event[4]), float(event[5])))
