"""Two Double DQN selectors trained inside the real C++ replay environment.

Both variants share one two-layer 64-unit MLP that scores the three regional
modes. The historical variant sees only observed state; the scenario variant
receives the same state plus frozen scenario-prior features and a flag. The
scheduler is the C++ server, which enforces cooldown and rebuild budgets.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from prepare import json_file, sha  # noqa: E402

REGIONS = 256
MODES = 3
FEATURES = 16  # 12 history + 3 scenario + 1 flag


class Env:
    def __init__(self, binary, trace, verify=False):
        self.proc = subprocess.Popen(
            [str(binary), "rlserve", "160", "32", "0", str(trace), "1" if verify else "0"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1)
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(self.proc.stderr.read())
        self.state = json.loads(line)

    def command(self, text):
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def step(self, actions, benefits=None):
        text = "step " + ",".join(str(int(a)) for a in actions)
        if benefits is not None:
            text += ";" + ",".join(f"{float(b):.6f}" for b in benefits)
        result = self.command(text)
        self.state = result["state"]
        return result

    def close(self):
        try:
            self.command("close")
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def prior_features(prior_path, window):
    """Mean and standard deviation of the scenario prior for one window."""
    if prior_path is None or not Path(prior_path).exists():
        return np.zeros((REGIONS, 3), dtype=np.float32)
    path = Path(prior_path)
    grouped = {}
    with path.open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            fields = line.split(",")
            if int(fields[1]) != window:
                continue
            region = int(fields[2])
            grouped.setdefault(region, []).append(
                [float(fields[3]), float(fields[4]), float(fields[5])])
    out = np.zeros((REGIONS, 3), dtype=np.float32)
    for region, values in grouped.items():
        out[region] = np.asarray(values, dtype=np.float32).mean(axis=0)
    return out


def build_features(state, window, prior, has_scenario):
    n = np.asarray(state["entities"], dtype=np.float32)
    q = np.asarray(state["queries"], dtype=np.float32)
    u = np.asarray(state["updates"], dtype=np.float32)
    c = np.asarray(state["coverage"], dtype=np.float32)
    mode = np.asarray(state["modes"], dtype=np.float32)
    cool = np.asarray(state["cooldown"], dtype=np.float32)
    phase = min(3.0, (state["step"] // 300))
    feats = np.zeros((REGIONS, FEATURES), dtype=np.float32)
    feats[:, 0] = n / 20000.0
    feats[:, 1] = q / 50.0
    feats[:, 2] = u / 400.0
    feats[:, 3] = c
    feats[:, 4] = mode / 2.0
    feats[:, 5] = (mode == 0).astype(np.float32)
    feats[:, 6] = (mode == 1).astype(np.float32)
    feats[:, 7] = (mode == 2).astype(np.float32)
    feats[:, 8] = cool / 100.0
    feats[:, 9] = state["rebuilds_left"] / 4.0
    feats[:, 10] = state["entities_left"] / 6000.0
    feats[:, 11] = window / 24.0
    feats[:, 12] = phase / 3.0
    if has_scenario:
        feats[:, 13] = prior[:, 0] / 50.0
        feats[:, 14] = prior[:, 2] / 400.0
        feats[:, 15] = 1.0
    return feats


class MLP:
    def __init__(self, din, hidden=64, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / din), (din, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, hidden)).astype(np.float32)
        self.b2 = np.zeros(hidden, dtype=np.float32)
        self.W3 = rng.normal(0, np.sqrt(1.0 / hidden), (hidden, MODES)).astype(np.float32)
        self.b3 = np.zeros(MODES, dtype=np.float32)
        self.m = [np.zeros_like(p) for p in self.params()]
        self.v = [np.zeros_like(p) for p in self.params()]
        self.t = 0

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, x, keep=False):
        z1 = x @ self.W1 + self.b1
        h1 = np.maximum(z1, 0)
        z2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(z2, 0)
        q = h2 @ self.W3 + self.b3
        if keep:
            self.cache = (x, z1, h1, z2, h2)
        return q

    def backward(self, dq):
        x, z1, h1, z2, h2 = self.cache
        dz2 = (dq @ self.W3.T) * (z2 > 0)
        dh1 = (dz2 @ self.W2.T) * (z1 > 0)
        grads = [x.T @ dh1, dh1.sum(axis=0), h1.T @ dz2, dz2.sum(axis=0),
                 h2.T @ dq, dq.sum(axis=0)]
        return grads

    def update(self, grads, lr=1e-3, clip=5.0):
        self.t += 1
        norm = np.sqrt(sum(float((g * g).sum()) for g in grads))
        scale = min(1.0, clip / (norm + 1e-9))
        b1, b2 = 0.9, 0.999
        for p, g, m, v in zip(self.params(), grads, self.m, self.v):
            g = g * scale
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * (g * g)
            mh = m / (1 - b1 ** self.t)
            vh = v / (1 - b2 ** self.t)
            p -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)

    def copy_from(self, other):
        for p, q in zip(self.params(), other.params()):
            p[...] = q


class Agent:
    def __init__(self, seed, has_scenario):
        self.online = MLP(FEATURES, seed=seed)
        self.target = MLP(FEATURES, seed=seed + 1)
        self.target.copy_from(self.online)
        self.has_scenario = has_scenario
        self.buffer = []
        self.rng = np.random.default_rng(seed + 7)
        self.soft_target = False
        self.tau = 0.005

    def mask(self, state):
        mode = np.asarray(state["modes"])
        cool = np.asarray(state["cooldown"])
        mask = np.ones((REGIONS, MODES), dtype=bool)
        mask[cool > 0] = False
        mask[cool > 0, mode[cool > 0]] = True
        return mask

    @staticmethod
    def benefits(q, state):
        """Expected advantage of each requested mode over keeping the current one."""
        mode = np.asarray(state["modes"])
        current = q[np.arange(REGIONS), mode]
        return (q - current[:, None]).max(axis=1)

    def act(self, feats, state, epsilon):
        mask = self.mask(state)
        q = self.online.forward(feats)
        q = np.where(mask, q, -1e9)
        actions = q.argmax(axis=1)
        if epsilon > 0:
            explore = self.rng.random(REGIONS) < epsilon
            for region in np.flatnonzero(explore):
                choices = np.flatnonzero(mask[region])
                actions[region] = self.rng.choice(choices)
        return actions.astype(int), q

    def remember(self, feats, actions, rewards, next_feats, next_mask, done):
        if len(self.buffer) > 150000:
            self.buffer = self.buffer[50000:]
        for region in range(REGIONS):
            self.buffer.append((feats[region], actions[region], rewards[region],
                                next_feats[region], next_mask[region], done))

    def train(self, batch=256, gamma=0.9):
        if len(self.buffer) < batch:
            return 0.0
        idx = self.rng.integers(0, len(self.buffer), batch)
        states = np.stack([self.buffer[i][0] for i in idx])
        actions = np.array([self.buffer[i][1] for i in idx])
        rewards = np.array([self.buffer[i][2] for i in idx], dtype=np.float32)
        next_states = np.stack([self.buffer[i][3] for i in idx])
        next_masks = np.stack([self.buffer[i][4] for i in idx])
        dones = np.array([self.buffer[i][5] for i in idx], dtype=np.float32)
        q_next_online = self.online.forward(next_states)
        q_next_online = np.where(next_masks, q_next_online, -1e9)
        best = q_next_online.argmax(axis=1)
        q_next_target = self.target.forward(next_states)
        targets = rewards + gamma * (1 - dones) * q_next_target[np.arange(batch), best]
        q = self.online.forward(states, keep=True)
        diff = q[np.arange(batch), actions] - targets
        loss = float((diff * diff).mean())
        dq = np.zeros_like(q)
        dq[np.arange(batch), actions] = 2 * diff / batch
        grads = self.online.backward(dq)
        self.online.update(grads)
        self.updates = getattr(self, "updates", 0) + 1
        if self.soft_target:
            for p, q in zip(self.target.params(), self.online.params()):
                p *= 1.0 - self.tau
                p += self.tau * q
        elif self.updates % 200 == 0:
            self.target.copy_from(self.online)
        return loss


def evaluate(agent, binary, traces, scenario_paths, epsilon=0.0, export=None):
    total = 0.0
    windows = 0
    inference = 0.0
    plans = {}
    for trace, prior_path in zip(traces, scenario_paths):
        env = Env(binary, trace)
        plan = []
        while not env.state["done"]:
            prior = prior_features(prior_path, min(23, env.state["window"] + 1))
            feats = build_features(env.state, env.state["window"], prior, agent.has_scenario)
            start = time.perf_counter()
            actions, q = agent.act(feats, env.state, epsilon)
            inference += time.perf_counter() - start
            result = env.step(actions, agent.benefits(q, env.state))
            plan.append([int(a) for a in result["applied"]])
            total += result["cost_ms"]
            windows += 1
        env.close()
        plans[str(trace)] = plan
    if export:
        Path(export).parent.mkdir(parents=True, exist_ok=True)
        Path(export).write_text(json.dumps(plans), encoding="utf-8")
    return total, windows, inference


def run(args):
    config = json.loads((args.config).read_text(encoding="utf-8"))
    seeds = args.seeds
    train = []
    for index, map_name in enumerate(config["maps"]["train"]):
        ident = f"train-{index}-{Path(map_name).stem}"
        trace = args.scenarios / "traces" / f"{ident}.actual.m3bin"
        prior = args.scenarios / "scenarios" / f"{ident}.nominal_samples.csv"
        train.append((trace, prior))
    validation = []
    for index, map_name in enumerate(config["maps"]["validation"]):
        ident = f"validation-{index}-{Path(map_name).stem}"
        validation.append((args.scenarios / "traces" / f"{ident}.actual.m3bin",
                           args.scenarios / "scenarios" / f"{ident}.nominal_samples.csv"))
    results = {}
    for variant in ("historical", "scenario"):
        for seed in seeds:
            agent = Agent(seed, has_scenario=(variant == "scenario"))
            losses = []
            start = time.time()
            best_validation = float("inf")
            stale = 0
            for episode in range(args.episodes):
                trace, prior_path = train[episode % len(train)]
                env = Env(args.binary, trace)
                while not env.state["done"]:
                    window = env.state["window"]
                    prior = prior_features(prior_path, min(23, window + 1))
                    feats = build_features(env.state, window, prior, agent.has_scenario)
                    epsilon = max(0.05, 1.0 - episode / max(1, args.episodes * 0.7))
                    actions, q = agent.act(feats, env.state, epsilon)
                    benefits = agent.benefits(q, env.state)
                    result = env.step(actions, benefits)
                    next_prior = prior_features(prior_path, min(23, window + 2))
                    next_feats = build_features(env.state, window + 1, next_prior,
                                                agent.has_scenario)
                    next_mask = agent.mask(env.state)
                    rewards = -np.asarray(result["region_us"], dtype=np.float32) / (1000.0 * 5.0)
                    rewards = np.clip(rewards, -5.0, 0.0)
                    # The scheduler may reject illegal requests; learn from the
                    # executed action and keep the next-state legal mask.
                    agent.remember(feats, np.asarray(result["applied"], dtype=int),
                                   rewards, next_feats, next_mask, float(result["done"]))
                    for _ in range(4):
                        losses.append(agent.train())
                env.close()
                if (episode + 1) % 5 == 0:
                    value, windows, _ = evaluate(agent, args.binary, [v[0] for v in validation],
                                                 [v[1] for v in validation])
                    print(f"{variant} seed {seed} episode {episode + 1}: "
                          f"validation {value:.1f} ms over {windows} windows, "
                          f"train {time.time() - start:.0f}s", flush=True)
                    if value < best_validation - 1.0:
                        best_validation = value
                        stale = 0
                    else:
                        stale += 1
                        if stale >= 3:
                            print(f"{variant} seed {seed}: early stop at episode {episode + 1}",
                                  flush=True)
                            break
            value, windows, inference = evaluate(agent, args.binary, [v[0] for v in validation],
                                                 [v[1] for v in validation])
            results[f"{variant}-{seed}"] = {"validation_ms": value, "windows": windows,
                                            "inference_s": inference,
                                            "episodes": args.episodes,
                                            "train_seconds": time.time() - start,
                                            "buffer": len(agent.buffer),
                                            "loss_last": float(np.mean(losses[-50:])) if losses else 0.0}
            np.savez(args.output / f"weights-{variant}-{seed}.npz",
                     W1=agent.online.W1, b1=agent.online.b1, W2=agent.online.W2,
                     b2=agent.online.b2, W3=agent.online.W3, b3=agent.online.b3)
            print(f"{variant} seed {seed}: validation {value:.1f} ms, "
                  f"train {time.time() - start:.0f}s", flush=True)
    json_file(args.output / "training.json", results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("scenarios", type=Path)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22])
    parser.add_argument("--episodes", type=int, default=60)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
