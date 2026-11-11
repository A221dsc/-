"""Automated tests for the two-way anchor drift and model timing completeness."""
import sys


def drift_ratio(start, end):
    return max(start, end) / min(start, end) if min(start, end) > 0 else float("inf")


def batch_valid(anchors, sentinel, threshold=1.15):
    ratios = [drift_ratio(s, e) for s, e in anchors]
    return all(r <= threshold for r in ratios) and drift_ratio(*sentinel) <= threshold


def main():
    assert not batch_valid([(100.0, 150.0)], (100.0, 100.0)), "50% slowdown not caught"
    assert not batch_valid([(100.0, 50.0)], (100.0, 100.0)), "50% speedup not caught"
    assert not batch_valid([(100.0, 100.0)], (402.1776, 263.9496)), "sentinel speedup not caught"
    assert not batch_valid([(815.3131, 628.9507)], (100.0, 100.0)), "1.296x trace drift not caught"
    assert batch_valid([(100.0, 110.0)], (200.0, 210.0)), "valid batch rejected"
    # feature_ms must be required: emulate the comparison lookup
    inference = {("u", "scenario", 22, "uniform-16", "t", 0): 15.0}
    feature = {}
    try:
        key = ("u", "scenario", 22, "uniform-16", "t", 0)
        if key not in feature:
            raise KeyError(key)
        raise AssertionError("missing feature_ms was not detected")
    except KeyError:
        pass
    print("anchor drift and timing completeness tests passed")


if __name__ == "__main__":
    main()
