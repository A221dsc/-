"""Generate a byte-stable, complete synthetic M3 point-movement replay.

The M3 scenario deliberately separates where the entities are concentrated from
where queries look. Force clusters are dense but only some of them are watched;
several query anchors sit in sparse areas. That separation is what makes a
regional index choice non-trivial: a dense, never-queried cluster is pure
maintenance, while a hot sparse area is cheaper to scan than to index.
"""
import math
import random
import struct
from pathlib import Path

from prepare import json_file, sha

HEADER = struct.Struct("<8s4d2Q")
POINT = struct.Struct("<2d")
EVENT = struct.Struct("<c7xq4d")


def anchors_for(cfg, phase, offset, duration):
    """Query anchor cell for one step of a phase."""
    plan = cfg["query_anchor_phases"][phase]
    if phase == "migration":
        x0, y0 = plan[0]
        x1, y1 = plan[1]
        frac = (offset + 1) / duration
        return x0 + frac * (x1 - x0), y0 + frac * (y1 - y0)
    if phase in ("alternation", "update_surge"):
        first, second = plan
        return first if (offset // cfg["alternation_steps"]) % 2 == 0 else second
    return plan[0]


def generate(config, seed, split, directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{split}_{seed}.m3bin"
    cfg = config
    n = cfg["entities"]
    lo, bot, hi, top = cfg["world"]
    phases = cfg["phases"]
    duration = cfg["phase_steps"]
    assert cfg["steps"] == len(phases) * duration
    assert cfg["window_steps"] == 50 and cfg["cooldown_windows"] == 2
    assert (cfg["switch_limit"], cfg["rebuild_entity_limit"]) == (4, 6000)
    assert duration % 50 == 0 and cfg["alternation_steps"] > 0
    updates = sum(cfg["updates_per_step"]) * duration
    queries = sum(cfg["queries_per_step"]) * duration
    boundaries = cfg["steps"] // 50 - 1
    rng = random.Random(seed ^ 0x519AF)
    motion = random.Random(seed ^ 0x160B)
    query_rng = random.Random(seed ^ 0xA72B)
    # Shift anchor cells by one seeded column to vary scenario layout across splits.
    dx = rng.choice((-1, 1)) * (seed % 2)
    entity_anchors = [(x + dx, y) for x, y in cfg["entity_clusters_xy"]]
    query_anchors = {phase: [(x + dx, y) for x, y in plan]
                     for phase, plan in cfg["query_anchor_phases"].items()}
    cellw, cellh = (hi - lo) / 16, (top - bot) / 16

    def center(pair):
        x, y = pair
        return lo + (x + 0.5) * cellw, bot + (y + 0.5) * cellh

    def clamp(x, y):
        return min(hi, max(lo, x)), min(top, max(bot, y))

    points = []
    masses = cfg["entity_cluster_mass"]
    for cluster, mass in enumerate(masses):
        count = round(n * mass)
        anchor = entity_anchors[cluster]
        x, y = center(anchor)
        sigma = cfg["cluster_sigma_m"][cluster]
        for _ in range(count):
            points.append(clamp(rng.gauss(x, sigma), rng.gauss(y, sigma)))
    while len(points) > n:
        points.pop()
    while len(points) < n:
        points.append((rng.uniform(lo, hi), rng.uniform(bot, top)))
    rng.shuffle(points)

    schedule = []
    with path.open("wb", buffering=1024 * 1024) as f:
        f.write(HEADER.pack(b"M3BIN001", *cfg["world"], n, updates + queries + boundaries))
        for p in points:
            f.write(POINT.pack(*p))
        for phase_index, label in enumerate(phases):
            u, q = cfg["updates_per_step"][phase_index], cfg["queries_per_step"][phase_index]
            hot = round(q * cfg["forced_hot_share"][phase_index])
            plan = query_anchors[label]
            for offset in range(duration):
                step = phase_index * duration + offset
                if step and step % 50 == 0:
                    f.write(EVENT.pack(b"P", step, 0, 0, 0, 0))
                for idx in motion.sample(range(n), u):
                    x, y = points[idx]
                    sd = cfg["movement_sigma_m"]
                    points[idx] = clamp(motion.gauss(x, sd), motion.gauss(y, sd))
                    f.write(EVENT.pack(b"U", idx + 1, *points[idx], 0, 0))
                anchor = anchors_for(cfg, label, offset, duration)
                cx, cy = center(anchor)
                schedule.append([step, label, round(anchor[0], 5), round(anchor[1], 5), hot])
                selected = set(query_rng.sample(range(q), hot))
                for j in range(q):
                    if j in selected:
                        x, y = clamp(query_rng.gauss(cx, 270), query_rng.gauss(cy, 270))
                    else:
                        x, y = query_rng.uniform(lo, hi), query_rng.uniform(bot, top)
                    half = query_rng.choice(cfg["query_half_sizes_m"])
                    if j % 2:
                        f.write(EVENT.pack(b"C", 0, x, y, 2 * half / math.sqrt(math.pi), 0))
                    else:
                        f.write(EVENT.pack(b"R", 0, x - half, y - half, x + half, y + half))
    info = {
        "trace": path.name, "split": split, "seed": seed, "sha256": sha(path),
        "dataset_variant": cfg["dataset_variant"], "steps": cfg["steps"], "entities": n,
        "updates": updates, "queries": queries, "boundaries": boundaries,
        "entity_clusters": [[float(x), float(y)] for x, y in entity_anchors],
        "entity_cluster_mass": masses,
        "query_anchor_phases": {phase: [[float(x), float(y)] for x, y in plan]
                                for phase, plan in query_anchors.items()},
        "schedule_fields": ["step", "phase", "hot_x_cell", "hot_y_cell", "forced_hot_queries"],
        "hotspot_schedule": schedule,
    }
    json_file(path.with_suffix(".json"), info)
    return path, info


if __name__ == "__main__":
    import argparse
    import json
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for split in ("train", "validation", "test"):
        for seed in config[f"{split}_seeds"]:
            path, info = generate(config, seed, split, args.directory)
            print(path, info["sha256"], flush=True)
