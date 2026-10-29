"""Opt-in numerical sensor reference. Historical four-ray Simpson readings stay v1.

These operators integrate a declared 2-D synthetic signal, not a calibrated gas
column or emission rate. A finer sensor does not change the frozen midpoint filter.
"""
from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Literal

import numpy as np

from plume_focus.sensing_contracts import Footprint, Message, finite, position


@dataclass(frozen=True)
class SensorQuadratureV2(Message):
    quadrature_id: Literal["tensor_gauss_exposure_v2"] = "tensor_gauss_exposure_v2"
    spatial_order: int = 16
    temporal_order: int = 8
    max_panel_seconds: float = .01

    def __post_init__(self):
        if self.quadrature_id != "tensor_gauss_exposure_v2":
            raise ValueError("Unknown sensor quadrature version.")
        for name in ("spatial_order", "temporal_order"):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= 64:
                raise ValueError(f"{name} must be an integer between 1 and 64.")
        finite("maximum integration panel", self.max_panel_seconds, positive=True)


@lru_cache(maxsize=64)
def _gauss(order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes.setflags(write=False); weights.setflags(write=False)
    return nodes, weights


def rectangle_gauss(center, footprint: Footprint, order=16):
    """Nodes and normalized weights for the FOV-clipped rectangular average."""
    position(center)
    if type(order) is not int or not 1 <= order <= 64:
        raise ValueError("Spatial order must be an integer between 1 and 64.")
    lower = np.maximum(0., np.asarray(center) - np.array((footprint.height, footprint.width))/2)
    upper = np.minimum(1., np.asarray(center) + np.array((footprint.height, footprint.width))/2)
    mid, half = (lower+upper)/2, (upper-lower)/2
    nodes, weights = _gauss(order)
    yy, xx = np.meshgrid(mid[0]+half[0]*nodes, mid[1]+half[1]*nodes, indexing="ij")
    return np.column_stack((yy.ravel(), xx.ravel())), np.outer(weights, weights).ravel()/4


def exact_rectangle_observation_weights(shape, center, footprint: Footprint):
    """Exact rectangle average of the existing clamped, bilinear latent field.

    Split at each latent-cell centre and integrate the piecewise-linear hat
    functions analytically. This is an explicit reference operator; the frozen
    GaussianSignalBelief still uses legacy four-point/midpoint likelihoods.
    """
    position(center)
    if len(shape) != 2 or any(type(n) is not int or n < 2 for n in shape):
        raise ValueError("Latent shape needs two integer dimensions >= 2.")
    lower = np.maximum(0., np.asarray(center)-np.array((footprint.height, footprint.width))/2)
    upper = np.minimum(1., np.asarray(center)+np.array((footprint.height, footprint.width))/2)

    def axis_weights(n, lo, hi):
        centers = (np.arange(n)+.5)/n
        knots = np.r_[lo, centers[(centers > lo) & (centers < hi)], hi]
        coords = np.clip(knots*n-.5, 0, n-1)
        low = np.floor(coords).astype(int); high = np.minimum(low+1, n-1)
        frac = coords-low
        basis = np.zeros((len(knots), n))
        np.add.at(basis, (np.arange(len(knots)), low), 1-frac)
        np.add.at(basis, (np.arange(len(knots)), high), frac)
        return np.sum((basis[:-1]+basis[1:])*np.diff(knots)[:,None]/2, axis=0)/(hi-lo)

    return np.outer(axis_weights(shape[0], lower[0], upper[0]),
                    axis_weights(shape[1], lower[1], upper[1])).ravel()


def exposure_average(source, sample, position_at, profile: SensorQuadratureV2, *, split_times=()):
    """Integrate a moving footprint, splitting known signal/path discontinuities.

    max_panel_seconds and both orders are accuracy controls, not universal error
    guarantees. Arbitrary SensorSource implementations still require refinement.
    """
    lo = sample.acquired_at_seconds-sample.integration_seconds/2
    hi = sample.acquired_at_seconds+sample.integration_seconds/2
    breaks = list(split_times)
    if hasattr(source, "temporal_breakpoints"):
        breaks.extend(source.temporal_breakpoints(lo, hi))
    if any(not math.isfinite(float(t)) for t in breaks):
        raise ValueError("Integration breakpoints must be finite.")
    boundaries = np.unique([lo, *[float(t) for t in breaks if lo < t < hi], hi])
    counts = [max(1, math.ceil((b-a)/profile.max_panel_seconds)) for a,b in zip(boundaries[:-1], boundaries[1:])]
    if sum(counts)*profile.temporal_order*profile.spatial_order**2 > 2_000_000:
        raise ValueError("Requested exposure quadrature exceeds the per-reading node limit.")
    tn, tw = _gauss(profile.temporal_order)
    times, temporal_weights = [], []
    for a,b,count in zip(boundaries[:-1], boundaries[1:], counts):
        edges = np.linspace(a,b,count+1)
        half = np.diff(edges)/2; mid = (edges[:-1]+edges[1:])/2
        times.extend((mid[:,None]+half[:,None]*tn).ravel())
        temporal_weights.extend((half[:,None]*tw/sample.integration_seconds).ravel())
    positions = []; weights = []
    for t,weight in zip(times, temporal_weights):
        points, spatial_weights = rectangle_gauss(position_at(t), sample.footprint, profile.spatial_order)
        positions.append(points); weights.append(weight*spatial_weights)
    positions = np.concatenate(positions)
    query_times = np.repeat(times, profile.spatial_order**2)
    values = np.asarray(source.signal_at(query_times, positions), dtype=float)
    if values.shape != query_times.shape or not np.all(np.isfinite(values)):
        raise ValueError("Sensor source must return one finite signal per quadrature node.")
    result = float(values @ np.concatenate(weights))
    if not math.isfinite(result):
        raise ValueError("Integrated sensor signal must be finite.")
    return result
