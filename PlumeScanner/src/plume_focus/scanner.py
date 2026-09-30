"""Single-pixel/raster scanner model.

The scanner makes the temporal constraint explicit: a frame assembled one pixel
at a time is not an instantaneous picture. The `sampled_at` map lets downstream
prediction inflate uncertainty when the scan took longer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal

import numpy as np
from numpy.typing import NDArray


FieldFunction = Callable[[float], NDArray[np.float64]]


@dataclass(frozen=True)
class ScanResult:
    """A rasterized sensor frame plus per-pixel sampling times."""

    values: NDArray[np.float64]
    sampled_at: NDArray[np.float64]
    started_at: float
    completed_at: float
    dwell_time: float
    stabilization_time: float

    @property
    def scan_duration(self) -> float:
        """Total time spent building this frame."""

        return self.completed_at - self.started_at

    @property
    def max_age(self) -> float:
        """Age of the oldest pixel at scan completion."""

        return self.completed_at - float(np.min(self.sampled_at))

    @property
    def mean_age(self) -> float:
        """Mean pixel age at scan completion."""

        return self.completed_at - float(np.mean(self.sampled_at))


class SinglePixelScanner:
    """Simulate a sensor that updates one pixel/subset at a time."""

    def __init__(
        self,
        shape: tuple[int, int],
        *,
        dwell_time: float,
        stabilization_time: float = 0.0,
        order: Literal["row-major", "snake"] = "row-major",
    ) -> None:
        if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
            raise ValueError("shape must be a positive (rows, cols) tuple.")
        if dwell_time <= 0:
            raise ValueError("dwell_time must be positive.")
        if stabilization_time < 0:
            raise ValueError("stabilization_time must be non-negative.")

        self.shape = shape
        self.dwell_time = float(dwell_time)
        self.stabilization_time = float(stabilization_time)
        self.order = order

    def scan(self, field_at: FieldFunction, *, start_time: float = 0.0) -> ScanResult:
        """Scan a time-varying field and return the assembled frame.

        `field_at(t)` should return the true 2-D plume field at time `t`. This
        makes it possible to quantify how much motion occurs during one pass.
        """

        values = np.zeros(self.shape, dtype=float)
        sampled_at = np.zeros(self.shape, dtype=float)
        t = float(start_time)

        for row, col in self._positions():
            sample_time = t + self.stabilization_time
            field = np.asarray(field_at(sample_time), dtype=float)
            if field.shape != self.shape:
                raise ValueError(f"field_at returned shape {field.shape!r}, expected {self.shape!r}.")

            values[row, col] = field[row, col]
            sampled_at[row, col] = sample_time
            t = sample_time + self.dwell_time

        return ScanResult(
            values=values,
            sampled_at=sampled_at,
            started_at=float(start_time),
            completed_at=t,
            dwell_time=self.dwell_time,
            stabilization_time=self.stabilization_time,
        )

    def _positions(self) -> Iterable[tuple[int, int]]:
        rows, cols = self.shape
        for row in range(rows):
            if self.order == "row-major" or row % 2 == 0:
                col_range = range(cols)
            elif self.order == "snake":
                col_range = range(cols - 1, -1, -1)
            else:
                raise ValueError(f"Unsupported scan order: {self.order!r}.")

            for col in col_range:
                yield row, col
