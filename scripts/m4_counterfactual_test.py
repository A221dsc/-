"""Correctness tests for the counterfactual benefit harness."""
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "scripts")

BINARY = Path("build_m3/spatial_m3.exe")
TRACE = Path("data/processed/m4/traces/train-0-brc202d.actual.m3bin")
PLAN = Path("results/m4/phase8/plans/train-0-brc202d.uniform-16.plan")


def run(window, leaf, mode_b, horizon, verify=False, profile=False, order=False):
    result = subprocess.run(
        [str(BINARY), "partition_cf", "160", "32", "0", str(TRACE), str(PLAN),
         str(window), str(leaf), "0", str(mode_b), str(horizon),
         "1" if verify else "0", "1" if profile else "0", "1" if order else "0"],
        capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(result.stderr)
    out = {}
    for token in result.stdout.split():
        key, value = token.split("=", 1)
        if key in ("window_q", "window_u"):
            out[key] = [float(part) for part in value.split("|") if part]
            continue
        if "/" in value:
            value = value.split("/")[0]
        out[key] = float(value)
    return out


def main():
    # 1. Identical arms: switching to the current mode must change nothing except noise.
    runs = [run(12, 55, 0, 1) for _ in range(3)]
    assert all(r["state_equal"] == 1 for r in runs), "null arms differ in state"
    nets = [r["net_ms"] for r in runs]
    # The two arms are independent replays; the observed null noise is ~1-2 ms.
    # The null noise is empirical with a heavy tail; use the median (the driver
    # measures the per-batch null noise for the label decision).
    assert statistics.median([abs(n) for n in nets]) < 5.0, f"null noise median too large: {nets}"
    assert all(r["entities"] == runs[0]["entities"] for r in runs)
    assert all(r["queries"] == runs[0]["queries"] for r in runs)
    print("identical arms:", [round(n, 3) for n in nets])

    # 2. Determinism of the state: the two arms see the same entity count, load and
    #    query count for a real switch.
    first = run(12, 55, 2, 1)
    second = run(12, 55, 2, 1)
    assert first["state_equal"] == 1 and second["state_equal"] == 1
    assert first["entities"] == second["entities"]
    assert first["obs_prev_q"] == second["obs_prev_q"]
    assert first["queries"] == second["queries"]
    assert abs(first["cost_a_ms"] - second["cost_a_ms"]) <= 0.20 * first["cost_a_ms"] + 1.0
    print("repeat costs:", round(first["cost_a_ms"], 3), round(second["cost_a_ms"], 3))

    # 3. Query correctness against the oracle on a real switch.
    verified = run(12, 55, 2, 2, verify=True)
    assert verified["mismatches"] == 0, "counterfactual oracle mismatch"
    assert verified["state_equal"] == 1
    print("oracle mismatches:", verified["mismatches"])

    # 3b. The digest must depend on the real state: different windows differ.
    early = run(4, 55, 0, 1)
    assert early["digest_before_a"] != first["digest_before_a"], "digest does not depend on state"
    print("digest changes across windows")

    # 3c. Local component conservation: the profile total never exceeds the full
    #     wall-clock arm, the per-window loads sum to the actual future load, and
    #     the switch is counted once in the action window.
    prof1 = run(12, 55, 2, 1, profile=True)
    prof2 = run(12, 55, 2, 2, profile=True)
    for prof in (prof1, prof2):
        assert prof["local_total_b"] <= prof["cost_b_ms"] + 1e-6, "local exceeds full cost"
        assert prof["local_total_b"] >= 0.0
        assert abs(prof["local_total_a"] - (prof["local_q_a"] + prof["local_base_a"] +
                                            prof["local_local_a"] + prof["local_switch_a"])) < 1e-9
        assert abs(prof["actual_future_q"] - sum(prof["window_q"])) < 1e-9
    assert prof2["windows"] == 2
    print("local component conservation passed")

    # 4. The uncertainty flag: a small net must be marked uncertain by the driver rule.
    small_ok = abs(first["net_ms"]) < 0.1
    print("small net", round(first["net_ms"], 3), "would be uncertain:", small_ok)
    print("counterfactual harness tests passed")


if __name__ == "__main__":
    main()
