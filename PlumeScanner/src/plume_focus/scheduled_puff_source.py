"""Explicit synthetic source shutoff: stop births, retain existing signal puffs.

    This is a more faithful cessation *mechanism*, not atmospheric calibration.
    The empirical puff amplitude, finite lifetime and shimmer remain synthetic.
"""
from dataclasses import asdict, dataclass, replace
import math
from typing import ClassVar
import numpy as np

from plume_focus.sensing_simulation import CleanMethaneSource, SourceExhausted
from plume_focus.video_sources import generate_synthetic_methane_values


def puff_breakpoints(config, start, end):
    """Birth and expiry times that lie strictly inside a requested exposure."""
    if not math.isfinite(start) or not math.isfinite(end) or end < start:
        raise ValueError("A finite chronological interval is required.")
    period = config.emission_interval/config.fps
    origin = -config.memory_frames/config.fps
    events = []
    for offset in (origin, 0.):
        first = max(0, int(math.floor((start-offset)/period)))
        last = max(first, int(math.ceil((end-offset)/period)))
        events.extend(t for n in range(first,last+1) if start < (t := offset+n*period) < end)
    return tuple(sorted(set(events)))


@dataclass(frozen=True, kw_only=True)
class ScheduledPuffSource(CleanMethaneSource):
    source_id: ClassVar[str] = "scheduled_signal_puffs_v3"
    emission_windows: tuple[tuple[float, float], ...]
    phase_seed: int | None = None

    def __post_init__(self):
        super().__post_init__()
        windows = tuple(tuple(float(v) for v in pair) for pair in self.emission_windows)
        previous = -float("inf")
        for pair in windows:
            if len(pair) != 2:
                raise ValueError("Emission windows need start/end pairs.")
            a,b = pair
            if not math.isfinite(a) or not math.isfinite(b) or not a < b or a < previous:
                raise ValueError("Emission windows must be finite, ordered, and nonoverlapping.")
            previous = b
        if self.phase_seed is not None and (type(self.phase_seed) is not int or self.phase_seed < 0):
            raise ValueError("The optional phase seed must be a nonnegative integer.")
        object.__setattr__(self, "emission_windows", windows)

    @classmethod
    def from_phase_seed(cls, config, *, emission_windows, phase_seed, end_time_seconds=None):
        if type(phase_seed) is not int or phase_seed < 0:
            raise ValueError("Phase seed must be a nonnegative integer.")
        phases = np.random.default_rng(phase_seed).uniform(0.,2*np.pi,3)
        resolved = replace(config, **dict(zip(("lateral_phase", "amplitude_phase", "shimmer_phase"), phases.tolist())))
        return cls(config=resolved, end_time_seconds=end_time_seconds,
                   emission_windows=emission_windows, phase_seed=phase_seed)

    def configuration(self):
        return {"source_id": self.source_id, "config": asdict(self.config),
                "end_time_seconds": self.end_time_seconds, "emission_windows": self.emission_windows,
                "phase_seed": self.phase_seed, "lifetime_convention": "half_open_v2",
                "units": "synthetic_normalized_signal"}

    def temporal_breakpoints(self, start, end):
        return puff_breakpoints(self.config, start, end)

    def signal_at(self, times, positions_yx):
        times = np.asarray(times, dtype=float); positions = np.asarray(positions_yx, dtype=float)
        if positions.shape != (*times.shape,2):
            raise ValueError("Positions must match times with a final (y,x) axis.")
        if not np.all(np.isfinite(positions)) or np.any((positions < 0) | (positions > 1)):
            raise ValueError("Source positions must be finite normalized FOV coordinates.")
        if (not np.all(np.isfinite(times)) or np.any(times < 0)
                or (self.end_time_seconds is not None and np.any(times > self.end_time_seconds+1e-10))):
            raise SourceExhausted("Synthetic source has no sample at the requested time.")
        return generate_synthetic_methane_values(self.config, times*self.config.fps,
            positions[...,0]*(self.config.shape[0]-1), positions[...,1]*(self.config.shape[1]-1),
            clean_signal=True, lifetime_convention="half_open_v2",
            emission_windows_frames=tuple((a*self.config.fps,b*self.config.fps) for a,b in self.emission_windows))
