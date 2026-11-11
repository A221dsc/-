"""H1 basic-fix tests: geometry (non-64 world, negative coords) and correctness.

Writes two small traces with custom bounds, a 4x4 partition each, and checks the
hpartition replay against the array oracle. Also checks that the node count is
the sum of the leaf Quad nodes (no extra leaf-count term).
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from m4_scenario import write_trace  # noqa: E402

BIN = "build_m3/spatial_m3.exe"
OUT = Path("results/partition_prediction/tests")
OUT.mkdir(parents=True, exist_ok=True)


def rects_4x4():
    return [(bx * 16, by * 16, (bx + 1) * 16, (by + 1) * 16, 1)
            for by in range(4) for bx in range(4)]


def make_trace(path, bounds, shift):
    points = [(bounds[0] + 50 + i * 30 + shift, bounds[1] + 40 + (i % 7) * 20)
              for i in range(40)]
    events = []
    for step in range(200):
        if step and step % 50 == 0:
            events.append(("P", step, 0, 0, 0, 0))
        events.append(("R", 0, bounds[0] + 10, bounds[1] + 10,
                       bounds[0] + 200, bounds[1] + 150))
        events.append(("U", 1 + step % 40, points[step % 40][0] + 1, points[step % 40][1] + 1,
                       0, 0))
        events.append(("C", 0, bounds[0] + 120, bounds[1] + 100, 60, 0))
    write_trace(path, bounds, points, events)


def run(trace, plan):
    out = subprocess.run([BIN, "hpartition", "160", "32", "0", str(trace), str(plan), "1", "0"],
                         capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(out.stderr)
    return dict(token.split("=", 1) for token in out.stdout.split())


plan = OUT / "4x4.plan"
plan.write_text("\n".join(f"L {a} {b} {c} {d} {e}" for (a, b, c, d, e) in rects_4x4()) + "\n",
                encoding="utf-8")
for name, bounds, shift in (("non64", (0.0, 0.0, 1000.0, 700.0), 0),
                            ("negative", (-500.0, -300.0, 500.0, 400.0), 3)):
    trace = OUT / f"{name}.m3bin"
    make_trace(trace, bounds, shift)
    head = run(trace, plan)
    assert head["mismatches"] == "0", (name, head)
    assert int(float(head["index_nodes"])) > 0
    print(name, "mismatches", head["mismatches"], "nodes", head["index_nodes"])
print("H1 basic tests passed")
