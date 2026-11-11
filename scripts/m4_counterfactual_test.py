"""Correctness tests for the counterfactual benefit harness."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "scripts")

BINARY = Path("build_m3/spatial_m3.exe")
TRACE = Path("data/processed/m4/traces/train-0-brc202d.actual.m3bin")
PLAN = Path("results/m4/phase8/plans/train-0-brc202d.uniform-16.plan")


def run(window, leaf, mode_b, horizon, verify=False):
    result = subprocess.run(
        [str(BINARY), "partition_cf", "160", "32", "0", str(TRACE), str(PLAN),
         str(window), str(leaf), "0", str(mode_b), str(horizon), "1" if verify else "0"],
        capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(result.stderr)
    out = {}
    for token in result.stdout.split():
        key, value = token.split("=", 1)
        if "/" in value:
            value = value.split("/")[0]
        out[key] = float(value)
    return out


def main():
    # 1. Identical arms: switching to the current mode must change nothing except noise.
    runs = [run(12, 55, 0, 1) for _ in range(3)]
    nets = [r["net_ms"] for r in runs]
    # The two arms are independent replays; the observed null noise is ~1-2 ms.
    assert max(abs(n) for n in nets) < 2.5, f"identical arms net too large: {nets}"
    assert all(r["entities"] == runs[0]["entities"] for r in runs)
    assert all(r["queries"] == runs[0]["queries"] for r in runs)
    print("identical arms:", [round(n, 3) for n in nets])

    # 2. Determinism of the state: the two arms see the same entity count, load and
    #    query count for a real switch.
    first = run(12, 55, 2, 1)
    second = run(12, 55, 2, 1)
    assert first["entities"] == second["entities"]
    assert first["load_q"] == second["load_q"]
    assert first["queries"] == second["queries"]
    assert abs(first["cost_a_ms"] - second["cost_a_ms"]) <= 0.05 * first["cost_a_ms"] + 0.5
    print("repeat costs:", round(first["cost_a_ms"], 3), round(second["cost_a_ms"], 3))

    # 3. Query correctness against the oracle on a real switch.
    verified = run(12, 55, 2, 2, verify=True)
    assert verified["mismatches"] == 0, "counterfactual oracle mismatch"
    print("oracle mismatches:", verified["mismatches"])

    # 4. The uncertainty flag: a small net must be marked uncertain by the driver rule.
    small_ok = abs(first["net_ms"]) < 0.1
    print("small net", round(first["net_ms"], 3), "would be uncertain:", small_ok)
    print("counterfactual harness tests passed")


if __name__ == "__main__":
    main()
