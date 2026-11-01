"""Opt-in synthetic source inputs with an explicit in-FOV total.

These functions do not change the historical scanner or its peak-rate source.
Values are source-input density per second on an equal-area unit-FOV grid;
their spatial mean is total_rate. This is not a physical methane emission rate.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

POINT_NORMALIZED_V1 = 'point_sampled_fov_normalized_v1'
CELL_INTEGRATED_V1 = 'cell_integrated_fov_normalized_v1'
SOURCE_INPUT_IDS = (POINT_NORMALIZED_V1, CELL_INTEGRATED_V1)


def _normal_interval(a, b):
    # Avoid subtracting two values rounded to one in the positive tail.
    return np.where(a >= 0, ndtr(-a)-ndtr(-b), ndtr(b)-ndtr(a))


def gaussian_source_input(grid_shape, sources_yx, *, source_sigma, total_rate,
                          formulation_id):
    """Return one flattened source vector, or a row per candidate source.

    point_sampled_fov_normalized_v1 preserves the historical sampled shape but
    normalizes its total. cell_integrated_fov_normalized_v1 analytically integrates
    a Gaussian over each cell before normalizing its in-FOV mass. Both condition
    the input on the unit FOV: boundary truncation changes shape, not total_rate.
    This boundary convention is a controlled model assumption.
    """
    if (len(grid_shape) != 2 or any(not isinstance(n, (int, np.integer))
            or isinstance(n, (bool, np.bool_)) or n < 2 for n in grid_shape)):
        raise ValueError('Grid shape must contain two integers >= 2')
    shape = tuple(int(n) for n in grid_shape)
    sources = np.asarray(sources_yx, dtype=float)
    single = sources.shape == (2,)
    if single:
        sources = sources[None]
    if (sources.ndim != 2 or sources.shape[1] != 2 or len(sources) == 0
            or not np.all(np.isfinite(sources))
            or np.any((sources < 0) | (sources > 1))):
        raise ValueError('Sources must be normalized y,x pairs inside the FOV')
    if not np.isfinite(source_sigma) or source_sigma <= 0:
        raise ValueError('Source sigma must be finite and positive')
    rates = np.asarray(total_rate, dtype=float)
    if rates.ndim == 0:
        rates = np.full(len(sources), float(rates))
    if (rates.shape != (len(sources),) or not np.all(np.isfinite(rates))
            or np.any(rates < 0)):
        raise ValueError('Total rate must be nonnegative, scalar or one per source')
    if formulation_id == POINT_NORMALIZED_V1:
        centers = [(np.arange(n)+.5)/n for n in shape]
        dy2 = (centers[0][None]-sources[:, 0, None])**2
        dx2 = (centers[1][None]-sources[:, 1, None])**2
        distances = dy2[:, :, None]+dx2[:, None, :]
        log_density = -distances/(2*source_sigma**2)
        log_density -= log_density.max(axis=(1, 2), keepdims=True)
        mass = np.exp(log_density)
        mass /= mass.sum(axis=(1, 2), keepdims=True)
    elif formulation_id == CELL_INTEGRATED_V1:
        intervals = []
        for axis, n in enumerate(shape):
            z = (np.linspace(0., 1., n+1)[None]-sources[:, axis, None])/source_sigma
            intervals.append(_normal_interval(z[:, :-1], z[:, 1:]))
        mass = intervals[0][:, :, None]*intervals[1][:, None, :]
        a, b = -sources/source_sigma, (1.-sources)/source_sigma
        in_fov = np.prod(_normal_interval(a, b), axis=1)
        if np.any(in_fov <= 0) or not np.all(np.isfinite(in_fov)):
            raise ValueError('Gaussian FOV mass is not numerically resolved')
        mass /= in_fov[:, None, None]
    else:
        raise ValueError('Unknown source-input formulation')
    result = mass.reshape(len(sources), -1)*np.prod(shape)*rates[:, None]
    if not np.all(np.isfinite(result)):
        raise ValueError('Source input is not numerically resolved')
    return result[0] if single else result
