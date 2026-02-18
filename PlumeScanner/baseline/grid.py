from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Tuple

Cell = Tuple[int, int]  # (x, y)


@dataclass(frozen=True)
class Grid:
    """
    2D discretization with:
      - x dimension wraps (circular)
      - y dimension bounded (no wrap)
    """
    width: int
    height: int
    wrap_x: bool = True

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Grid width and height must be positive.")

    def normalize(self, cell: Cell) -> Cell:
        x, y = cell
        if self.wrap_x:
            x = x % self.width
        if not (0 <= y < self.height):
            raise IndexError(f"y out of bounds: {y}")
        if not (0 <= x < self.width):
            raise IndexError(f"x out of bounds: {x}")
        return (x, y)

    def x_dist(self, x1: int, x2: int) -> int:
        """Wrap-aware shortest distance in x."""
        d = abs(x1 - x2)
        if self.wrap_x:
            d = min(d, self.width - d)
        return d

    def l1_dist(self, a: Cell, b: Cell) -> int:
        ax, ay = self.normalize(a)
        bx, by = self.normalize(b)
        return self.x_dist(ax, bx) + abs(ay - by)

    def iter_cells_rowmajor(self) -> Iterator[Cell]:
        for y in range(self.height):
            for x in range(self.width):
                yield (x, y)

    def iter_raster_zigzag(self, start: Cell = (0, 0)) -> Iterator[Cell]:
        """
        Infinite generator: one full zigzag raster ordering, repeated forever.
        """
        sx, sy = start
        sx, sy = self.normalize((sx, sy))

        # Build one full ordering
        order = []
        for dy in range(self.height):
            y = (sy + dy) % self.height
            xs = list(range(self.width))

            # Zigzag: reverse every other row to reduce motion
            if y % 2 == 1:
                xs.reverse()

            # Rotate first row to start at sx (simple choice)
            if dy == 0 and sx != 0:
                idx = xs.index(sx)
                xs = xs[idx:] + xs[:idx]

            for x in xs:
                order.append((x, y))

        # Repeat forever
        while True:
            for cell in order:
                yield cell
