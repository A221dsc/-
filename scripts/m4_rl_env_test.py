"""Interface, determinism, cost conservation and scheduler tests for the RL server."""
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


def run_episode(policy):
    env = Env()
    states, applied, costs, rejected = [env.state], [], [], []
    while not env.state["done"]:
        actions, benefits = policy(env.state)
        result = env.step(actions, benefits)
        applied.append(result["applied"])
        rejected.append(result["rejected"])
        costs.append(result)
        states.append(env.state)
    env.close()
    return states, applied, costs, rejected


def policy_scan(state):
    return [0] * REGIONS, None


def policy_bandit(state):
    order = sorted(range(REGIONS), key=lambda r: -state["queries"][r])[:8]
    actions = list(state["modes"])
    benefits = [0.0] * REGIONS
    for rank, region in enumerate(order):
        actions[region] = 2
        benefits[region] = 1.0 - rank * 0.01
    return actions, benefits


def policy_all_promote(state):
    return [2] * REGIONS, [1.0] * REGIONS


def main():
    states_a, applied_a, costs_a, _ = run_episode(policy_scan)
    states_b, applied_b, _, _ = run_episode(policy_scan)
    assert states_a == states_b, "state replay is not deterministic"
    assert applied_a == applied_b, "applied actions are not deterministic"
    print(f"scan determinism: {len(costs_a)} windows")

    for result in costs_a:
        parts = (result["query_ms"] + result["base_update_ms"] +
                 result["maintenance_ms"] + result["switch_ms"])
        assert abs(parts - result["cost_ms"]) < 1e-3, "cost components do not sum"
        assert abs(result["region_sum_ms"] - result["cost_ms"]) <= 0.02 * result["cost_ms"] + 1e-6, \
            "regional attribution does not conserve the window cost"
    print("cost conservation: components sum and regional attribution agree")

    states_c, _, costs_c, _ = run_episode(policy_bandit)
    promotions = []
    for w in range(len(states_c) - 1):
        pre, post = states_c[w]["modes"], states_c[w + 1]["modes"]
        promotions.append(sum(1 for r in range(REGIONS)
                              if post[r] != pre[r] and post[r] != 0))
    print(f"bandit: max promotions per window {max(promotions)} (<=4)")
    assert max(promotions) <= 4
    last = {}
    for w in range(len(states_c) - 1):
        pre, post = states_c[w]["modes"], states_c[w + 1]["modes"]
        for region in range(REGIONS):
            if pre[region] != post[region]:
                if region in last:
                    assert w - last[region] >= 2, "cooldown violated"
                last[region] = w
    first = costs_c[0]
    assert len(first["rejected"]) == REGIONS
    assert sum(first["rejected"]) > 0
    print("cooldown respected, scheduler reported rejection codes")

    env = Env()
    result = env.step([2] * REGIONS, benefits=[1.0] * REGIONS)
    applied = result["applied"]
    assert sum(1 for a in applied if a != 0) <= 4
    assert 3 in result["rejected"] or 4 in result["rejected"], "budget rejection not reported"
    print("budget scheduler: at most four promotions, rejections reported")
    env.close()

    env = Env()
    result = env.step([2] * REGIONS, benefits=[-1.0] * REGIONS)
    assert all(a == 0 for a in result["applied"]), "negative benefit promoted a region"
    assert result["rejected"].count(5) > 0
    print("negative benefit rejected")
    env.close()

    env = Env(verify=True)
    result = env.step(*policy_scan(env.state))
    assert env.state["mismatches"] == 0
    env.close()
    print("query correctness verified against the array oracle")
    print("trace sha256", sha(TRACE)[:16])


if __name__ == "__main__":
    main()
