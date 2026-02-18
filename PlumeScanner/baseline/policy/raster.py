from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Tuple
from ..grid import Grid, Cell


@dataclass(frozen=True)
class RasterConfig:
    dwell_s: float = 0.12
    start: Cell = (0, 0)


class RasterBaselinePolicy:
    """
    Phase 1 baseline: infinite uniform raster scan (zigzag),
    fixed dwell per cell.
    """
    def __init__(self, grid: Grid, cfg: RasterConfig):
        self.grid = grid
        self.cfg = cfg

    def actions(self) -> Iterator[Tuple[Cell, float]]:
        for cell in self.grid.iter_raster_zigzag(start=self.cfg.start):
            yield (cell, self.cfg.dwell_s)
