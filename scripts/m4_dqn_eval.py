"""Evaluate trained Double DQN policies on the real replay and time them fairly.

The learned policy is first replayed through the interactive server to produce
a per-window action plan, then the plan is timed by `replayplan` on the same
QPC path as the built-in selectors. Inference time is measured separately.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from m4_dqn import FEATURES, MODES, REGIONS, Agent, Env, build_features, prior_features  # noqa: E402
from prepare import json_file  # noqa: E402


def load_agent(path, scenario):
    agent = Agent(0, has_scenario=scenario)
    data = np.load(path)
    agent.online.W1[...] = data["W1"]
    agent.online.b1[...] = data["b1"]
    agent.online.W2[...] = data["W2"]
    agent.online.b2[...] = data["b2"]
    agent.online.W3[...] = data["W3"]
    agent.online.b3[...] = data["b3"]
    return agent


def plan_episode(agent, binary, trace, prior_path):
    env = Env(binary, trace)
    plan = []
    inference = 0.0
    while not env.state["done"]:
        window = env.state["window"]
        prior = prior_features(prior_path, min(23, window + 1))
        feats = build_features(env.state, window, prior, agent.has_scenario)
        start = time.perf_counter()
        actions, q = agent.act(feats, env.state, 0.0)
        inference += time.perf_counter() - start
        result = env.step(actions, agent.benefits(q, env.state))
        plan.append([int(a) for a in result["applied"]])
    env.close()
    return plan, inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("scenarios", type=Path)
    parser.add_argument("binary", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    plans_dir = args.output / "plans"
    plans_dir.mkdir(exist_ok=True)
    models = []
    for variant in ("historical", "scenario"):
        for seed in sorted(args.weights.glob(f"weights-{variant}-*.npz")):
            models.append((variant, int(seed.stem.split("-")[-1]), seed))
    traces = []
    for split in ("validation", "test"):
        for index, map_name in enumerate(config["maps"][split]):
            ident = f"{split}-{index}-{Path(map_name).stem}"
            count = 0
            while True:
                name = (f"{ident}.actual.m3bin" if count == 0
                        else f"{ident}.actual{count}.m3bin")
                path = args.scenarios / "traces" / name
                if not path.exists():
                    break
                traces.append({"split": split, "scenario": ident, "actual": count,
                               "trace": path,
                               "prior": args.scenarios / "scenarios" / f"{ident}.nominal_samples.csv"})
                count += 1
    rows = []
    for variant, seed, path in models:
        agent = load_agent(path, scenario=(variant == "scenario"))
        for entry in traces:
            plan, inference = plan_episode(agent, args.binary, entry["trace"], entry["prior"])
            plan_file = plans_dir / f"{variant}-{seed}-{entry['scenario']}-a{entry['actual']}.txt"
            plan_file.write_text("\n".join(",".join(str(a) for a in row) for row in plan) + "\n",
                                 encoding="utf-8")
            result = subprocess.run([str(args.binary), "replayplan", "160", "32", "0",
                                     str(entry["trace"]), str(plan_file)],
                                    capture_output=True, text=True)
            if result.returncode:
                raise SystemExit(result.stderr)
            head = dict(token.split("=", 1) for token in result.stdout.strip().split())
            rows.append({"variant": variant, "seed": seed, "split": entry["split"],
                         "scenario": entry["scenario"], "actual": entry["actual"],
                         "online_ms": head["online_ms"], "query_ms": head["query_ms"],
                         "update_ms": head["update_ms"], "switch_ms": head["switch_ms"],
                         "promotions": head["promotions"], "switches": head["switches"],
                         "inference_ms": inference * 1000.0,
                         "decisions": len(plan)})
            print(f"{variant}-{seed} {entry['scenario']} a{entry['actual']}: "
                  f"{head['online_ms']} ms", flush=True)
    import csv
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_file(args.output / "manifest.json", {"models": [f"{v}-{s}" for v, s, _ in models],
                                              "traces": len(traces)})
    print("done")


if __name__ == "__main__":
    main()
