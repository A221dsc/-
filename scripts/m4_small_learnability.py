"""2x2 learnability check on the exact cost model used by the DP reference.

The tabular Q-learner and the shared-network Double DQN see the same 24-window
loads, switch costs and cooldown as the exact DP. This validates the learning
mechanism only; it does not replace real-index performance tests.
"""
import argparse
import itertools
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from m4_dqn import MLP, MODES  # noqa: E402
from m4_offline_ref import query_update_cost, read_table, switch_matrix  # noqa: E402

REGIONS = 4
NAMES = ["SCAN", "FINE_GRID", "LOCAL_QUAD"]


class SmallEnv:
    def __init__(self, loads, regions, model, horizon=24):
        self.model = model
        self.windows = min(horizon, loads.shape[0])
        self.loads = loads[: self.windows][:, regions, :]
        self.sw = switch_matrix(model, float(loads[:, regions, 1].max()))
        self.mode_cost = np.zeros((self.windows, REGIONS, MODES))
        for w in range(self.windows):
            for i in range(REGIONS):
                for m in range(MODES):
                    self.mode_cost[w, i, m] = query_update_cost(model, m, self.loads[w, i])
        self.reset()

    def reset(self):
        self.window = 0
        self.modes = (0, 0, 0, 0)
        self.cooldown = (0, 0, 0, 0)
        return self.state()

    def state(self):
        return (self.window, self.modes, self.cooldown)

    def legal(self):
        return [[m for m in range(MODES) if not (self.cooldown[i] and m != self.modes[i])]
                for i in range(REGIONS)]

    def step(self, target):
        total = 0.0
        rewards = []
        for i in range(REGIONS):
            cost = self.mode_cost[self.window, i, target[i]]
            if target[i] != self.modes[i]:
                cost += self.sw[self.modes[i], target[i]]
            rewards.append(-cost)
            total += cost
        new_cooldown = tuple(1 if target[i] != self.modes[i] else 0 for i in range(REGIONS))
        self.window += 1
        self.modes = tuple(target)
        self.cooldown = new_cooldown
        done = self.window >= self.windows
        return self.state(), rewards, total, done


def enumerate_policy(env):
    """Optimal value via the same DP as the offline reference (4 regions)."""
    states = {((0, 0, 0, 0), (0, 0, 0, 0)): 0.0}  # start in SCAN like the environment
    for w in range(env.windows):
        nxt = {}
        for (modes, cooldown), value in states.items():
            for target in itertools.product(range(MODES), repeat=REGIONS):
                if any(cooldown[i] and target[i] != modes[i] for i in range(REGIONS)):
                    continue
                cost = value
                for i in range(REGIONS):
                    cost += env.sw[modes[i], target[i]] if target[i] != modes[i] else 0.0
                    cost += env.mode_cost[w, i, target[i]]
                state = (target, tuple(1 if target[i] != modes[i] else 0 for i in range(REGIONS)))
                if cost < nxt.get(state, float("inf")):
                    nxt[state] = cost
        states = nxt
    return min(states.values())


def greedy_baseline(env):
    """Myopic cheapest mode per region, no switch lookahead."""
    env.reset()
    total = 0.0
    while True:
        target = tuple(int(np.argmin([env.mode_cost[env.window, i, m] for m in range(MODES)]))
                       for i in range(REGIONS))
        _, _, cost, done = env.step(target)
        total += cost
        if done:
            return total


def keep_baseline(env):
    env.reset()
    total = 0.0
    while True:
        _, _, cost, done = env.step((0, 0, 0, 0))
        total += cost
        if done:
            return total


def random_baseline(env, rng, episodes=200):
    values = []
    for _ in range(episodes):
        env.reset()
        total = 0.0
        while True:
            target = tuple(rng.choice(env.legal()[i]) for i in range(REGIONS))
            _, _, cost, done = env.step(target)
            total += cost
            if done:
                values.append(total)
                break
    return sum(values) / len(values)


def legal_indices(state, action_index):
    _, modes, cooldown = state
    choices = [[m for m in range(MODES) if not (cooldown[i] and m != modes[i])]
               for i in range(REGIONS)]
    return [action_index[tuple(a)] for a in itertools.product(*choices)]


def tabular_q(env, episodes=4000, gamma=0.95, alpha=0.2, seed=0):
    rng = random.Random(seed)
    actions = list(itertools.product(range(MODES), repeat=REGIONS))
    action_index = {a: i for i, a in enumerate(actions)}
    q = {}
    def get(state):
        return q.setdefault(state, np.zeros(len(actions)))
    for episode in range(episodes):
        env.reset()
        epsilon = max(0.02, 1.0 - episode / (episodes * 0.8))
        while True:
            state = env.state()
            values = get(state)
            legal = [action_index[tuple(a)] for a in
                     itertools.product(*[env.legal()[i] for i in range(REGIONS)])]
            if rng.random() < epsilon:
                index = rng.choice(legal)
            else:
                index = max(legal, key=lambda i: values[i])
            target = actions[index]
            next_state, _, cost, done = env.step(target)
            reward = -cost
            nxt = get(next_state)
            if done:
                update = reward
            else:
                legal_next = legal_indices(next_state, action_index)
                update = reward + gamma * max(nxt[i] for i in legal_next)
            values[index] += alpha * (update - values[index])
            if done:
                break
    # greedy evaluation
    env.reset()
    total = 0.0
    while True:
        state = env.state()
        values = get(state)
        legal = [action_index[tuple(a)] for a in
                 itertools.product(*[env.legal()[i] for i in range(REGIONS)])]
        index = max(legal, key=lambda i: values[i])
        _, _, cost, done = env.step(actions[index])
        total += cost
        if done:
            break
    return total


class DQN4:
    """Shared-network Double DQN over the four regions."""

    def __init__(self, seed, hidden=64):
        self.net = MLP(12, hidden=hidden, seed=seed)
        self.target = MLP(12, hidden=hidden, seed=seed + 1)
        self.target.copy_from(self.net)
        self.rng = np.random.default_rng(seed + 3)
        self.buffer = []
        self.updates = 0

    def features(self, env):
        out = np.zeros((REGIONS, 12), dtype=np.float32)
        for i in range(REGIONS):
            q, n, u, c = env.loads[min(env.window, env.windows - 1), i]
            out[i, 0] = q / 50.0
            out[i, 1] = n / 8000.0
            out[i, 2] = u / 400.0
            out[i, 3] = c
            out[i, 4] = env.modes[i] / 2.0
            out[i, 5] = (env.modes[i] == 0)
            out[i, 6] = (env.modes[i] == 1)
            out[i, 7] = (env.modes[i] == 2)
            out[i, 8] = env.cooldown[i]
            out[i, 9] = env.window / 24.0
            out[i, 10] = 1.0
            out[i, 11] = 0.0
        return out

    @staticmethod
    def mask(env):
        mask = np.ones((REGIONS, MODES), dtype=bool)
        for i in range(REGIONS):
            if env.cooldown[i]:
                mask[i] = False
                mask[i, env.modes[i]] = True
        return mask

    def act(self, feats, env, epsilon=0.0):
        q = self.net.forward(feats)
        mask = self.mask(env)
        q = np.where(mask, q, -1e9)
        actions = q.argmax(axis=1)
        if epsilon > 0:
            for i in range(REGIONS):
                if self.rng.random() < epsilon:
                    choices = np.flatnonzero(mask[i])
                    actions[i] = self.rng.choice(choices)
        return actions.astype(int)

    def train(self, batch=128, gamma=0.95):
        if len(self.buffer) < batch:
            return
        idx = self.rng.integers(0, len(self.buffer), batch)
        s = np.stack([self.buffer[i][0] for i in idx])
        a = np.array([self.buffer[i][1] for i in idx])
        r = np.array([self.buffer[i][2] for i in idx], dtype=np.float32)
        s2 = np.stack([self.buffer[i][3] for i in idx])
        m2 = np.stack([self.buffer[i][4] for i in idx])
        done = np.array([self.buffer[i][5] for i in idx], dtype=np.float32)
        q2 = self.net.forward(s2)
        q2 = np.where(m2, q2, -1e9)
        best = q2.argmax(axis=1)
        t2 = self.target.forward(s2)
        target = r + gamma * (1 - done) * t2[np.arange(batch), best]
        q = self.net.forward(s, keep=True)
        diff = q[np.arange(batch), a] - target
        dq = np.zeros_like(q)
        dq[np.arange(batch), a] = 2 * diff / batch
        self.net.update(self.net.backward(dq))
        self.updates += 1
        if self.updates % 100 == 0:
            self.target.copy_from(self.net)


def train_dqn(env, episodes=600, seed=0, epsilon_end=0.05):
    agent = DQN4(seed)
    rng = np.random.default_rng(seed)
    for episode in range(episodes):
        env.reset()
        epsilon = max(epsilon_end, 1.0 - episode / (episodes * 0.8))
        while True:
            feats = agent.features(env)
            actions = agent.act(feats, env, epsilon)
            next_state, rewards, cost, done = env.step(tuple(int(a) for a in actions))
            next_feats = agent.features(env)
            next_mask = agent.mask(env)
            for i in range(REGIONS):
                agent.buffer.append((feats[i], int(actions[i]), rewards[i] / 5.0,
                                     next_feats[i], next_mask[i], float(done)))
            if len(agent.buffer) > 40000:
                agent.buffer = agent.buffer[10000:]
            for _ in range(2):
                agent.train()
            if done:
                break
    env.reset()
    total = 0.0
    while True:
        feats = agent.features(env)
        actions = agent.act(feats, env, 0.0)
        _, _, cost, done = env.step(tuple(int(a) for a in actions))
        total += cost
        if done:
            break
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration", type=Path)
    parser.add_argument("scenarios", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    model = json.loads((args.calibration / "cost_model.json").read_text(encoding="utf-8"))
    loads = read_table(args.scenarios / "loads" / "test-2-den011d.actual_loads.csv")
    best_block, best_value = (0, 0), -1.0
    for by in range(15):
        for bx in range(15):
            value = float(sum(loads[w, 16 * (by + dy) + bx + dx, 0]
                              for w in range(loads.shape[0]) for dy in (0, 1) for dx in (0, 1)))
            if value > best_value:
                best_block, best_value = (bx, by), value
    bx, by = best_block
    regions = [16 * by + bx, 16 * by + bx + 1, 16 * (by + 1) + bx, 16 * (by + 1) + bx + 1]
    env = SmallEnv(loads, regions, model)
    optimum = enumerate_policy(env)
    greedy = greedy_baseline(env)
    keep = keep_baseline(env)
    rng = random.Random(11)
    random_value = random_baseline(env, rng)
    tabular = tabular_q(env)
    dqn = train_dqn(env, seed=11)
    rows = [
        {"method": "exact DP", "cost_ms": optimum / 1e6, "gap": 0.0},
        {"method": "tabular Q-learning", "cost_ms": tabular / 1e6,
         "gap": (tabular - optimum) / optimum},
        {"method": "Double DQN (4 regions)", "cost_ms": dqn / 1e6,
         "gap": (dqn - optimum) / optimum},
        {"method": "cost greedy (myopic)", "cost_ms": greedy / 1e6,
         "gap": (greedy - optimum) / optimum},
        {"method": "keep SCAN", "cost_ms": keep / 1e6, "gap": (keep - optimum) / optimum},
        {"method": "random legal", "cost_ms": random_value / 1e6,
         "gap": (random_value - optimum) / optimum},
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    import csv
    with (args.output / "small_learnability.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['method']:26s} {row['cost_ms']:8.3f} ms  gap {row['gap']:+.3f}")


if __name__ == "__main__":
    main()
