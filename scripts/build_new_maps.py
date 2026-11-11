import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from m4_build import build_one

path = Path("configs/m4.json")
config = json.loads(path.read_text(encoding="utf-8"))
# Frozen seeds for the new independent maps.
extra = {"train": 4401, "validation": 4402, "test": 4403}
for split, seed in extra.items():
    while len(config["scenario_seeds"][split]) < len(config["maps"][split]):
        config["scenario_seeds"][split].append(seed)
path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
directory = Path("data/processed/m4")
for split, map_name in (("train", "den101d.map"), ("validation", "orz200d.map"),
                        ("test", "den011d.map")):
    index = config["maps"][split].index(map_name)
    build_one(config, map_name, split, index, directory, "medium")
    print(f"built {split}-{index}-{Path(map_name).stem}", flush=True)
