import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.prepare import prepare_tdrive


class PrepareTest(unittest.TestCase):
    def test_sort_dedup_projection_and_disjoint_id_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "sample.zip"
            with zipfile.ZipFile(archive, "w") as z:
                for i in range(1, 41):
                    z.writestr(f"taxi_log/{i}.txt",
                               f"{i},2008-02-02 15:45:00,116.401,39.9\n"
                               f"{i},2008-02-02 15:36:00,116.4,39.9\n"
                               f"{i},2008-02-02 15:36:00,116.4,39.9\n"
                               f"{i},2008-02-02 15:45:00,116.402,39.9\n"
                               f"{i},2008-02-02 15:50:00,,39.9\n")
            output = root / "out"
            prepare_tdrive(archive, output)
            groups = []
            for split in ("train", "validation", "test"):
                with (output / f"{split}.csv").open(newline="") as file:
                    rows = list(csv.DictReader(file))
                groups.append({r["entity_id"] for r in rows})
                self.assertTrue(groups[-1])
                self.assertEqual(len(rows), len(groups[-1]) * 2)
                first, second = [r for r in rows if r["entity_id"] == rows[0]["entity_id"]]
                self.assertEqual(first["event_type"], "INSERT")
                self.assertEqual(first["timestamp"], "2008-02-02 15:36:00")
                self.assertEqual(second["event_type"], "UPDATE")
                self.assertAlmostEqual(float(first["x"]), 0, places=5)
                self.assertTrue(165 < float(second["x"]) < 175)
            self.assertEqual(len(set.union(*groups)), sum(map(len, groups)))
            quality = json.loads((output / "quality.json").read_text())
            self.assertEqual(quality["counts"]["duplicate_exact"], 40)
            self.assertEqual(quality["counts"]["duplicate_conflict"], 40)
            self.assertEqual(quality["counts"]["invalid"], 40)


if __name__ == "__main__":
    unittest.main()
