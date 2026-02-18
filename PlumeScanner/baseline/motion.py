from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
from .grid import Grid, Cell


@dataclass(frozen=True)
class MotionModel:
    """
    Simple motion-time model in "cell units":
      - move time scales with delta_x and delta_y
      - choose either max (axis-limited) or sum (additive)
    """
    x_rate_cells_per_s: float = 40.0
    y_rate_cells_per_s: float = 20.0
    settle_s: float = 0.02
    mode: str = "max"  # "max" or "sum"

    def move_time(self, grid: Grid, prev: Cell, nxt: Cell) -> float:
        px, py = grid.normalize(prev)
        nx, ny = grid.normalize(nxt)
        dx = grid.x_dist(px, nx)
        dy = abs(py - ny)

        tx = dx / self.x_rate_cells_per_s if self.x_rate_cells_per_s > 0 else float("inf")
        ty = dy / self.y_rate_cells_per_s if self.y_rate_cells_per_s > 0 else float("inf")

        if self.mode == "max":
            t = max(tx, ty)
        elif self.mode == "sum":
            t = tx + ty
        else:
            raise ValueError("MotionModel.mode must be 'max' or 'sum'.")

        return t + self.settle_s
