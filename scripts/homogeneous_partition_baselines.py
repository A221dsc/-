"""H0 baseline plans: the same PR Quadtree in every leaf, only the partition differs.

All plans are L-only (no mode changes): the benchmark forces every leaf to
LOCAL_QUAD, i.e. one PR Quadtree per leaf, the same Quad class used by the
full-map PR Quadtree baseline.
"""
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from m4_partition_experiment import partition_adaptive, partition_uniform, read_prior  # noqa: E402

config = __import__("json").load(open("configs/m4.json"))
names = ["brc100d", "brc203d", "den011d"]
out = Path("results/homogeneous_partition/plans")
out.mkdir(parents=True, exist_ok=True)
variants = [("uniform-1", None), ("uniform-4", None), ("uniform-8", None), ("uniform-16", None),
            ("uniform-32", None), ("entity", (0.0, 1.0, 0.0)), ("query", (1.0, 0.0, 0.0)),
            ("combined", (1.0, 1.0, 1.0))]
for index in range(3):
    ident = f"test-{index}-{Path(config['maps']['test'][index]).stem}"
    for actual in range(3):
        suffix = "" if actual == 0 else f".actual{actual}"
        prior = read_prior(Path("data/processed/m4/scenarios") / f"{ident}.nominal_samples_m32.csv")
        for name, weights in variants:
            if weights is None:
                rects = partition_uniform(int(name.split("-")[1]))
            else:
                rects = partition_adaptive(prior, weights[0], weights[1], weights[2])
            lines = [f"L {x0} {y0} {x1} {y1} {level}" for (x0, y0, x1, y1, level) in rects]
            (out / f"{ident}{suffix}.{name}.plan").write_text("\n".join(lines) + "\n",
                                                              encoding="utf-8")
        print(f"{ident} a{actual}: plans written", flush=True)
