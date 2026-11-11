"""Evaluate partition Double DQN policies with the same QPC replay path as the
greedy selectors. The learned policy is replayed through the interactive server
to collect its executed mode changes, then the plan is timed by `partition`."""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from m4_dqn_partition import Agent, FEATURES, Env, features, leaf_geometry, read_plan  # noqa: E402
from prepare import json_file  # noqa: E402

PARTITIONS = ["uniform-16", "adaptive-query"]


def load_agent(path, has_scenario):
    agent = Agent(0, has_scenario=has_scenario)
    data = np.load(path)
    for name, param in (("W1", agent.online.W1), ("b1", agent.online.b1),
                        ("W2", agent.online.W2), ("b2", agent.online.b2),
                        ("W3", agent.online.W3), ("b3", agent.online.b3)):
        param[...] = data[name]
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("scenarios", type=Path)
    parser.add_argument("binary", type=Path)
    parser.add_argument("plans", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    plan_dir = args.output / "plans"
    plan_dir.mkdir(exist_ok=True)
    models = []
    for variant in ("historical", "scenario"):
        for path in sorted(args.weights.glob(f"weights-{variant}-*.npz")):
            models.append((variant, int(path.stem.split("-")[-1]), path))
    rows = []
    for variant, seed, path in models:
        agent = load_agent(path, variant == "scenario")
        for partition in PARTITIONS:
            for index in range(3):
                ident = f"test-{index}-{Path(config['maps']['test'][index]).stem}"
                source = args.plans / f"{ident}.{partition}.plan"
                rects, levels, prior = read_plan(source)
                areas, aspect = leaf_geometry(rects)
                for actual in range(3):
                    trace = args.scenarios / "traces" / (
                        f"{ident}.actual.m3bin" if actual == 0 else f"{ident}.actual{actual}.m3bin")
                    env = Env(args.binary, trace, source)
                    changes = []
                    inference = 0.0
                    previous = np.asarray(env.state["modes"])
                    while not env.state["done"]:
                        window = env.state["window"]
                        feats = features(env.state, min(23, window + 1), areas, aspect, levels,
                                         prior, agent.has_scenario)
                        start = time.perf_counter()
                        actions, benefits = agent.act(feats, env.state, 0.0)
                        inference += time.perf_counter() - start
                        result = env.step(actions, benefits)
                        applied = np.asarray(result["applied"])
                        for leaf in range(len(previous)):
                            if previous[leaf] != applied[leaf]:
                                changes.append((window, leaf, int(applied[leaf])))
                        previous = applied
                    env.close()
                    plan_path = plan_dir / f"{variant}-{seed}-{ident}-{partition}-a{actual}.plan"
                    lines = [f"L {x0} {y0} {x1} {y1} {level}"
                             for (x0, y0, x1, y1), level in zip(rects, levels)]
                    lines += [f"M {w} {leaf} {mode} 1.0" for (w, leaf, mode) in changes]
                    plan_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    result = subprocess.run([str(args.binary), "partition", "160", "32", "0",
                                             str(trace), str(plan_path), "0"],
                                            capture_output=True, text=True)
                    if result.returncode:
                        raise SystemExit(result.stderr)
                    head = dict(token.split("=", 1) for token in result.stdout.strip().split())
                    rows.append({"variant": variant, "seed": seed, "partition": partition,
                                 "scenario": ident, "actual": actual,
                                 "online_ms": float(head["online_ms"]) + inference * 1000.0,
                                 "index_ms": head["online_ms"],
                                 "inference_ms": inference * 1000.0,
                                 "changes": len(changes)})
                    print(f"{variant}-{seed} {partition} {ident} a{actual}: "
                          f"{float(head['online_ms']) + inference*1000:.1f} ms", flush=True)
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_file(args.output / "manifest.json", {"models": [f"{v}-{s}" for v, s, _ in models],
                                              "partitions": PARTITIONS})


if __name__ == "__main__":
    main()
