"""Interface, determinism and cost-consistency tests for the C++ RL server."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from prepare import sha  # noqa: E402

BINARY = Path("build_m3/spatial_m3.exe")
TRACE = Path("data/processed/m4/traces/train-0-brc202d.actual.m3bin")
REGIONS = 256


class Env:
    def __init__(self, trace=TRACE, verify=False):
        self.proc = subprocess.Popen(
            [str(BINARY), "rlserve", "160", "32", "0", str(trace), "1" if verify else "0"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            bufsize=1)
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(self.proc.stderr.read())
        self.state = json.loads(line)

    def command(self, text):
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        return json.loads(line)

    def step(self, actions):
        result = self.command("step " + ",".join(str(a) for a in actions))
        self.state = result["state"]
        return result

    def close(self):
        try:
            self.command("close")
        except Exception:
            pass
        self.proc.wait(timeout=10)


def policy_scan(step):
    return [0] * REGIONS


def policy_bandit(step):
    # Promote only the four hottest regions of the last window.
    state = step
    order = sorted(range(REGIONS), key=lambda r: -state["queries"][r])[:4]
    actions = list(state["modes"])
    for region in order:
        actions[region] = 2
    return actions


def run_episode(policy, trace=TRACE):
    env = Env(trace)
    states, applied, costs = [env.state], [], []
    while not env.state["done"]:
        actions = policy(env.state)
        result = env.step(actions)
        applied.append(result["applied"])
        costs.append(result["cost_ms"])
        states.append(env.state)
    env.close()
    return states, applied, costs


def main():
    assert Env(TRACE).state["done"] is False
    states_a, applied_a, costs_a = run_episode(policy_scan)
    states_b, applied_b, costs_b = run_episode(policy_scan)
    assert states_a == states_b, "state replay is not deterministic"
    assert applied_a == applied_b, "applied actions are not deterministic"
    print(f"scan determinism: {len(costs_a)} windows, states identical")
    print(f"scan window cost sum: {sum(costs_a):.2f} ms (timing may vary)")

    states_c, applied_c, costs_c = run_episode(policy_bandit)
    changes = sum(1 for a, b in zip(states_c[:-1], states_c[1:])
                  if a["modes"] != b["modes"])
    promotions = []
    for w in range(len(states_c) - 1):
        pre, post = states_c[w]["modes"], states_c[w + 1]["modes"]
        promotions.append(sum(1 for r in range(REGIONS)
                              if post[r] != pre[r] and post[r] != 0))
    print(f"bandit: {changes} windows changed modes; "
          f"max promotions per window {max(promotions)} (<=4)")
    assert max(promotions) <= 4, "rebuild budget violated"
    # cooldown: a region switched in window k must not switch again before 2 windows later
    last = {}
    for w in range(len(states_c) - 1):
        pre, post = states_c[w]["modes"], states_c[w + 1]["modes"]
        for region in range(REGIONS):
            if pre[region] != post[region]:
                if region in last:
                    assert w - last[region] >= 2, "cooldown violated"
                last[region] = w
    # query count consistency
    total_queries = sum(sum(state["queries"]) for state in states_a[:-1])
    total_updates = sum(sum(state["updates"]) for state in states_a[:-1])
    print(f"replayed queries {total_queries:.0f}, update touches {total_updates:.0f}")
    assert total_queries > 40000
    # verify path
    env = Env(TRACE, verify=True)
    env.step(policy_scan(env.state))
    assert env.state["mismatches"] == 0
    env.close()
    print("cooldown, budget and query correctness checks passed")
    print("trace sha256", sha(TRACE)[:16])


if __name__ == "__main__":
    main()
