"""Double DQN on the PartitionIndex (fixed leaf set, variable leaf count)."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from m4_dqn import MLP, MODES  # noqa: E402
from prepare import json_file  # noqa: E402

FEATURES = 20


def read_plan(path):
    rects, levels, prior = [], [], {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line or line[0] == "#":
            continue
        f = line.split()
        if f[0] == "L":
            rects.append(tuple(int(v) for v in f[1:5]))
            levels.append(int(f[5]))
        elif f[0] == "P":
            prior[(int(f[1]), int(f[2]))] = (float(f[3]), float(f[4]), float(f[5]), float(f[6]))
    return rects, levels, prior


def leaf_geometry(rects):
    area = np.array([(r[2] - r[0]) * (r[3] - r[1]) / 16.0 for r in rects], dtype=np.float32)
    aspect = np.array([(r[2] - r[0]) / max(1.0, r[3] - r[1]) for r in rects], dtype=np.float32)
    return area, aspect


class Env:
    def __init__(self, binary, trace, plan, verify=False):
        self.proc = subprocess.Popen(
            [str(binary), "partition_rlserve", "160", "32", "0", str(trace), str(plan),
             "1" if verify else "0"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1)
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(self.proc.stderr.read())
        self.state = json.loads(line)

    def step(self, actions, benefits):
        text = "step " + ",".join(str(int(a)) for a in actions) + ";" + \
               ",".join(f"{float(b):.6f}" for b in benefits)
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()
        result = json.loads(self.proc.stdout.readline())
        self.state = result["state"]
        return result

    def close(self):
        try:
            self.proc.stdin.write("close\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def features(state, window, areas, aspect, levels, prior, has_scenario):
    n = len(state["modes"])
    f = np.zeros((n, FEATURES), dtype=np.float32)
    f[:, 0] = np.asarray(state["entities"]) / 20000.0
    f[:, 1] = np.asarray(state["queries"]) / 50.0
    f[:, 2] = np.asarray(state["updates"]) / 400.0
    f[:, 3] = np.asarray(state["coverage"])
    modes = np.asarray(state["modes"], dtype=np.float32)
    f[:, 4] = modes / 2.0
    f[:, 5] = (modes == 0)
    f[:, 6] = (modes == 1)
    f[:, 7] = (modes == 2)
    f[:, 8] = np.asarray(state["cooldown"]) / 100.0
    f[:, 9] = state["rebuilds_left"] / 4.0
    f[:, 10] = state["entities_left"] / 6000.0
    f[:, 11] = window / 24.0
    f[:, 12] = areas
    f[:, 13] = np.asarray(levels, dtype=np.float32) / 2.0
    f[:, 14] = aspect
    if has_scenario:
        for leaf in range(n):
            cell = prior.get((window, leaf))
            if cell:
                f[leaf, 15] = cell[0] / 50.0
                f[leaf, 16] = cell[1] / 8000.0
                f[leaf, 17] = cell[2] / 400.0
                f[leaf, 18] = cell[3]
        f[:, 19] = 1.0
    return f


class Agent:
    def __init__(self, seed, has_scenario):
        self.online = MLP(FEATURES, seed=seed)
        self.target = MLP(FEATURES, seed=seed + 1)
        self.target.copy_from(self.online)
        self.has_scenario = has_scenario
        self.buffer = []
        self.rng = np.random.default_rng(seed + 5)
        self.updates = 0

    @staticmethod
    def mask(state):
        mode = np.asarray(state["modes"])
        cool = np.asarray(state["cooldown"])
        mask = np.ones((len(mode), MODES), dtype=bool)
        mask[cool > 0] = False
        mask[cool > 0, mode[cool > 0]] = True
        return mask

    def act(self, feats, state, epsilon):
        mask = self.mask(state)
        q = np.where(mask, self.online.forward(feats), -1e9)
        actions = q.argmax(axis=1)
        if epsilon > 0:
            explore = self.rng.random(len(actions)) < epsilon
            for i in np.flatnonzero(explore):
                actions[i] = self.rng.choice(np.flatnonzero(mask[i]))
        mode = np.asarray(state["modes"])
        benefit = (q - q[np.arange(len(mode)), mode][:, None]).max(axis=1)
        return actions.astype(int), benefit

    def remember(self, feats, actions, rewards, next_feats, next_mask, done):
        if len(self.buffer) > 150000:
            self.buffer = self.buffer[50000:]
        for i in range(len(actions)):
            self.buffer.append((feats[i], actions[i], rewards[i], next_feats[i],
                                next_mask[i], done))

    def train(self, batch=256, gamma=0.9):
        if len(self.buffer) < batch:
            return 0.0
        idx = self.rng.integers(0, len(self.buffer), batch)
        s = np.stack([self.buffer[i][0] for i in idx])
        a = np.array([self.buffer[i][1] for i in idx])
        r = np.array([self.buffer[i][2] for i in idx], dtype=np.float32)
        s2 = np.stack([self.buffer[i][3] for i in idx])
        m2 = np.stack([self.buffer[i][4] for i in idx])
        done = np.array([self.buffer[i][5] for i in idx], dtype=np.float32)
        q2 = np.where(m2, self.online.forward(s2), -1e9)
        best = q2.argmax(axis=1)
        t2 = self.target.forward(s2)
        target = r + gamma * (1 - done) * t2[np.arange(batch), best]
        q = self.online.forward(s, keep=True)
        diff = q[np.arange(batch), a] - target
        dq = np.zeros_like(q)
        dq[np.arange(batch), a] = 2 * diff / batch
        self.online.update(self.online.backward(dq))
        self.updates += 1
        if self.updates % 200 == 0:
            self.target.copy_from(self.online)
        return float((diff * diff).mean())


def episode(agent, binary, trace, plan, epsilon, train=True, collect=None):
    rects, levels, prior = read_plan(plan)
    areas, aspect = leaf_geometry(rects)
    env = Env(binary, trace, plan)
    total = 0.0
    inference = 0.0
    previous = np.asarray(env.state["modes"])
    while not env.state["done"]:
        window = env.state["window"]
        feats = features(env.state, min(23, window + 1), areas, aspect, levels, prior,
                         agent.has_scenario)
        start = time.perf_counter()
        actions, benefits = agent.act(feats, env.state, epsilon)
        inference += time.perf_counter() - start
        result = env.step(actions, benefits)
        total += result["cost_ms"]
        if train:
            nxt = features(env.state, min(23, window + 2), areas, aspect, levels, prior,
                           agent.has_scenario)
            rewards = -np.asarray(result["region_us"], dtype=np.float32) / (1000.0 * 5.0)
            agent.remember(feats, np.asarray(result["applied"], dtype=int),
                           np.clip(rewards, -5.0, 0.0), nxt, agent.mask(env.state),
                           float(result["done"]))
            for _ in range(3):
                agent.train()
        if collect is not None and not np.array_equal(previous, result["applied"]):
            for leaf in range(len(previous)):
                if previous[leaf] != result["applied"][leaf]:
                    collect.append((window, leaf, int(result["applied"][leaf])))
            previous = np.asarray(result["applied"])
    env.close()
    return total, inference


def run(args):
    config = json.loads(args.config.read_text(encoding="utf-8"))
    train = []
    for index, map_name in enumerate(config["maps"]["train"]):
        ident = f"train-{index}-{Path(map_name).stem}"
        trace = args.scenarios / "traces" / f"{ident}.actual.m3bin"
        plan = args.plans / f"{ident}.{args.train_partition}.plan"
        train.append((trace, plan))
    results = {}
    for variant in ("historical", "scenario"):
        for seed in args.seeds:
            agent = Agent(seed, has_scenario=(variant == "scenario"))
            best = float("inf")
            stale = 0
            start = time.time()
            for episode_index in range(args.episodes):
                trace, plan = train[episode_index % len(train)]
                epsilon = max(0.05, 1.0 - episode_index / max(1, args.episodes * 0.7))
                total, _ = episode(agent, args.binary, trace, plan, epsilon)
                if (episode_index + 1) % 5 == 0:
                    value, _ = episode(agent, args.binary, train[0][0], train[0][1], 0.0,
                                       train=False)
                    print(f"{variant} seed {seed} ep {episode_index + 1}: val {value:.1f} ms "
                          f"train {time.time() - start:.0f}s", flush=True)
                    if value < best - 1.0:
                        best, stale = value, 0
                    else:
                        stale += 1
                        if stale >= 3:
                            print(f"{variant} seed {seed}: early stop", flush=True)
                            break
            value, _ = episode(agent, args.binary, train[0][0], train[0][1], 0.0, train=False)
            results[f"{variant}-{seed}"] = {"validation_ms": value, "episodes": episode_index + 1,
                                            "train_seconds": time.time() - start}
            np.savez(args.output / f"weights-{variant}-{seed}.npz",
                     W1=agent.online.W1, b1=agent.online.b1, W2=agent.online.W2,
                     b2=agent.online.b2, W3=agent.online.W3, b3=agent.online.b3)
            print(f"{variant} seed {seed}: final {value:.1f} ms", flush=True)
    json_file(args.output / "training.json", results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("scenarios", type=Path)
    parser.add_argument("binary", type=Path)
    parser.add_argument("plans", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--train-partition", default="uniform-16")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
