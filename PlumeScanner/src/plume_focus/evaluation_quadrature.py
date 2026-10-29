"""Versioned time quadrature for evaluator-only physical-time integrals."""
import math
import numpy as np
from plume_focus.sensing_contracts import finite


def time_quadrature(start, duration, step, *, rule="left_endpoint_v1", breakpoints=()):
    finite("quadrature start", start)
    finite("quadrature duration", duration, positive=True)
    finite("quadrature step", step, positive=True)
    if rule not in ("left_endpoint_v1", "event_split_midpoint_v2"):
        raise ValueError("Unsupported temporal quadrature version.")
    # Preserve the exact legacy grid arithmetic and endpoint interpretation.
    left = start+np.arange(int(math.ceil(duration/step)))*step
    if rule == "left_endpoint_v1":
        if len(breakpoints):
            raise ValueError("Event splitting requires event_split_midpoint_v2.")
        return left, np.diff(np.append(left,start+duration))
    events = np.asarray(tuple(breakpoints), dtype=float)
    if events.ndim != 1 or not np.all(np.isfinite(events)):
        raise ValueError("Quadrature event times must be a finite vector.")
    end = start+duration
    edges = np.unique(np.r_[left[left < end], events[(events > start) & (events < end)], end])
    durations = np.diff(edges)
    return edges[:-1]+durations/2, durations
