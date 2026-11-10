"""Load Moving AI Lab benchmark maps and build walkability-based routes.

Map format: https://movingai.com/benchmarks/ (Sturtevant, 2012). The DAO map set
is distributed for research use with attribution to BioWare.
"""
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np

WALKABLE = set(".G")
CELL_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class GameMap:
    def __init__(self, path, max_extent_m=20000.0):
        self.file_path = Path(path)
        raw = self.file_path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        lines = raw.decode("latin-1").splitlines()
        header = {}
        index = 0
        while index < len(lines) and " " in lines[index]:
            key, _, value = lines[index].partition(" ")
            header[key.strip().lower()] = value.strip()
            index += 1
            if key.strip().lower() == "map":
                break
        assert header.get("type") == "octile", f"{path}: unsupported map type"
        self.height = int(header["height"])
        self.width = int(header["width"])
        body = lines[index:index + self.height]
        assert len(body) == self.height, f"{path}: truncated map body"
        grid = np.zeros((self.height, self.width), dtype=bool)
        for row, text in enumerate(body):
            for column, char in enumerate(text[:self.width]):
                grid[row, column] = char in WALKABLE
        self.walkable = grid
        self.walkable_count = int(grid.sum())
        self.cell_m = max_extent_m / max(self.width, self.height)
        self.world_w = self.width * self.cell_m
        self.world_h = self.height * self.cell_m
        self.bounds = (0.0, 0.0, self.world_w, self.world_h)
        self._fields = {}

    def cell_center(self, cell):
        x, y = cell
        return ((x + 0.5) * self.cell_m, (y + 0.5) * self.cell_m)

    def world_to_cell(self, point):
        x, y = point
        return (min(self.width - 1, max(0, int(x / self.cell_m))),
                min(self.height - 1, max(0, int(y / self.cell_m))))

    def region_of(self, point, side=16):
        x, y = point
        bx = min(side - 1, max(0, int((x - self.bounds[0]) / (self.world_w / side))))
        by = min(side - 1, max(0, int((y - self.bounds[1]) / (self.world_h / side))))
        return side * by + bx

    def random_walkable_cells(self, rng, count):
        cells = np.argwhere(self.walkable)
        picks = rng.choice(len(cells), size=count, replace=True)
        return [(int(cells[i][1]), int(cells[i][0])) for i in picks]

    def distance_field(self, goal):
        """8-connected BFS distance in cells from every walkable cell to goal."""
        key = tuple(goal)
        if key in self._fields:
            return self._fields[key]
        field = np.full((self.height, self.width), -1, dtype=np.int32)
        gx, gy = goal
        assert self.walkable[gy, gx], f"goal {goal} is not walkable"
        field[gy, gx] = 0
        queue = deque([(gx, gy)])
        walkable = self.walkable
        while queue:
            x, y = queue.popleft()
            base = field[y, x] + 1
            for dx, dy in CELL_NEIGHBOURS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and walkable[ny, nx] and field[ny, nx] < 0:
                    field[ny, nx] = base
                    queue.append((nx, ny))
        self._fields[key] = field
        return field

    def route_cells(self, start, goal):
        """Gradient-descent path from start to goal along the BFS field."""
        field = self.distance_field(goal)
        x, y = start
        if field[y, x] < 0:
            raise ValueError(f"start {start} is unreachable from {goal}")
        cells = [(x, y)]
        limit = 4 * self.width * self.height
        while field[y, x] > 0:
            best = None
            for dx, dy in CELL_NEIGHBOURS:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                value = field[ny, nx]
                if value >= 0 and (best is None or value < best[0]):
                    best = (value, nx, ny)
            if best is None:
                raise ValueError(f"no descent from {(x, y)} to {goal}")
            x, y = best[1], best[2]
            cells.append((x, y))
            limit -= 1
            if limit <= 0:
                raise ValueError("path descent did not converge")
        return cells

    def objective_cells(self, rng, count, margin=0.08):
        """Pick spread-out walkable objective cells with a minimum separation."""
        cells = np.argwhere(self.walkable)
        xs, ys = cells[:, 1], cells[:, 0]
        min_dx = self.width * margin
        min_dy = self.height * margin
        goals = []
        tries = 0
        while len(goals) < count and tries < 20000:
            tries += 1
            i = int(rng.integers(len(cells)))
            cand = (int(xs[i]), int(ys[i]))
            if all(abs(cand[0] - g[0]) >= min_dx or abs(cand[1] - g[1]) >= min_dy for g in goals):
                goals.append(cand)
        if len(goals) < count:
            raise RuntimeError("could not place enough objectives")
        return goals

    def manifest(self):
        return {"file": self.file_path.name, "sha256": self.sha256, "width": self.width,
                "height": self.height, "walkable_fraction": self.walkable_count / (self.width * self.height),
                "cell_m": self.cell_m, "world_bounds": list(self.bounds),
                "format": "octile", "source": "https://movingai.com/benchmarks/grids.html"}


def write_manifest(directory, maps):
    payload = {"source": "https://movingai.com/benchmarks/dao/dao-map.zip",
               "license": "Moving AI Lab 2D benchmarks: free for research with attribution "
                          "(Sturtevant 2012); DAO maps distributed with BioWare permission.",
               "maps": [m.manifest() for m in maps]}
    Path(directory, "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import sys
    for name in sys.argv[1:]:
        game = GameMap(name)
        print(json.dumps(game.manifest(), indent=2))
