"""Persistent discrete wind-bias posterior with conditional Gaussian fields."""

from dataclasses import replace
from functools import lru_cache
import math

import numpy as np

from plume_focus.motion_config import PersistentMotionConfig
from plume_focus.signal_belief import BeliefView, GaussianSignalBelief, observation_weights, grid_positions, interpolation_stencil
from plume_focus.signal_distribution import PredictiveDistribution


@lru_cache(maxsize=12)
def _projection_stencil(shape, latent_shape):
    ids, weights = interpolation_stencil(grid_positions(shape), latent_shape)
    ids.setflags(write=False); weights.setflags(write=False)
    return ids, weights


def bias_quadrature(config: PersistentMotionConfig):
    cov = np.asarray(config.bias_covariance_yx)
    eigenvalues, vectors = np.linalg.eigh(cov)
    root = (vectors * np.sqrt(np.maximum(eigenvalues, 0))) @ vectors.T
    nodes, weights = np.polynomial.hermite.hermgauss(config.quadrature_order)
    nodes, weights = nodes * math.sqrt(2), weights / math.sqrt(math.pi)
    merged = {}
    for i, y in enumerate(nodes):
        for j, x in enumerate(nodes):
            offset = tuple(root @ np.array((y, x)))
            merged[offset] = merged.get(offset, 0.) + weights[i] * weights[j]
    offsets = np.asarray(list(merged))
    probabilities = np.asarray(list(merged.values()))
    return offsets, probabilities / probabilities.sum()


class PersistentWindBelief:
    def __init__(self, config, motion, *, time_seconds=0., transport_id="legacy_clipped_v1"):
        self.config, self.motion = config, motion
        self.transport_id = transport_id
        self.offsets, probabilities = bias_quadrature(motion)
        self.offsets.setflags(write=False)
        self.log_weights = np.log(probabilities)
        self.components = tuple(GaussianSignalBelief(config, time_seconds=time_seconds, transport_id=transport_id) for _ in probabilities)

    @property
    def time_seconds(self):
        return self.components[0].time_seconds

    @property
    def weights(self):
        weights = np.exp(self.log_weights - self.log_weights.max())
        return weights / weights.sum()

    @property
    def mean(self):
        return sum(w * state.mean for w, state in zip(self.weights, self.components))

    @property
    def covariance(self):
        mean = self.mean
        return sum(w * (state.covariance + np.outer(state.mean - mean, state.mean - mean))
                   for w, state in zip(self.weights, self.components))

    def clone(self):
        result = object.__new__(type(self))
        result.config, result.motion, result.offsets = self.config, self.motion, self.offsets
        result.transport_id = self.transport_id
        result.log_weights = self.log_weights.copy()
        result.components = tuple(state.clone() for state in self.components)
        return result

    def advance(self, target_seconds, wind, source=None):
        for state, offset in zip(self.components, self.offsets):
            velocity = tuple(np.asarray(wind.velocity_yx) + offset)
            state.advance(target_seconds, replace(wind, velocity_yx=velocity), source)

    def condition(self, value, weights, noise_variance):
        # Validate on a detached Gaussian before mutating any component/weight.
        trial = self.components[0].clone()
        trial.condition(value, weights, noise_variance)
        h = np.asarray(weights)
        likelihoods = []
        for state in self.components:
            variance = float(h @ state.covariance @ h + noise_variance)
            residual = value - float(h @ state.mean)
            likelihoods.append(-.5 * (math.log(2 * math.pi * variance) + residual**2 / variance))
        updated = self.log_weights + np.asarray(likelihoods)
        peak = updated.max()
        self.log_weights = updated - (peak + math.log(float(np.exp(updated - peak).sum())))
        for state in self.components:
            state.condition(value, h, noise_variance)

    def assimilate(self, reading, wind, source=None):
        self.advance(reading.acquired_at_seconds, wind, source)
        h = observation_weights(self.config.grid_shape, reading.position_yx, reading.footprint)
        self.condition((reading.value - self.config.offset) / self.config.gain, h,
                       (self.config.sensor_noise_std / self.config.gain)**2)

    def distribution_on(self, shape, *, fast_projection=False):
        if len(self.components) == 1:
            mean, variance = self.components[0].moments_on(shape)
            return PredictiveDistribution(self.time_seconds, self.weights, mean[None], variance[None])
        ids, weights = _projection_stencil(tuple(shape), self.config.grid_shape)
        means = np.stack([s.mean for s in self.components])
        covariances = np.stack([s.covariance for s in self.components])
        mean = np.sum(means[:, ids] * weights[None], axis=-1)
        if fast_projection:
            # Exact bilinear variance polynomial, evaluated separably for large
            # offline grids. The planner retains the original summation order.
            ny, nx = self.config.grid_shape
            yy = np.clip((np.arange(shape[0])+.5) * ny/shape[0]-.5, 0, ny-1)
            xx = np.clip((np.arange(shape[1])+.5) * nx/shape[1]-.5, 0, nx-1)
            y0, x0 = np.floor(yy).astype(int), np.floor(xx).astype(int)
            y1, x1 = np.minimum(y0+1,ny-1), np.minimum(x0+1,nx-1)
            fy, fx = yy-y0, xx-x0
            a = np.arange(ny)[:,None]*nx+x0
            b = np.arange(ny)[:,None]*nx+x1
            horizontal = ((1-fx)**2 * covariances[:,a,a] + 2*fx*(1-fx)*covariances[:,a,b]
                          + fx**2*covariances[:,b,b])
            cross = ((1-fx)**2*covariances[:,a[:-1],a[1:]]
                     + fx*(1-fx)*(covariances[:,a[:-1],b[1:]]+covariances[:,b[:-1],a[1:]])
                     + fx**2*covariances[:,b[:-1],b[1:]])
            variance = ((1-fy)[None,:,None]**2*horizontal[:,y0]
                        +2*(fy*(1-fy))[None,:,None]*cross[:,np.minimum(y0,ny-2)]
                        +fy[None,:,None]**2*horizontal[:,y1])
            return PredictiveDistribution(self.time_seconds,self.weights,mean.reshape((len(means),*shape)),np.maximum(variance,0))
        variance = np.zeros_like(mean)
        for j in range(4):
            for k in range(4):
                variance += weights[None, :, j] * weights[None, :, k] * covariances[:, ids[:, j], ids[:, k]]
        return PredictiveDistribution(self.time_seconds, self.weights,
            mean.reshape((len(means), *shape)), np.maximum(variance, 0).reshape((len(means), *shape)))

    def moments_on(self, shape):
        distribution = self.distribution_on(shape)
        return distribution.mean, distribution.variance

    def aggregate_moments_on(self, shape):
        """Offline loss refinement needs aggregate moments, not every band."""
        state = self.components[0].clone()
        state.mean, state.covariance = self.mean, self.covariance
        return state.moments_on(shape)

    def expected_mass_on(self, shape, weights, mass_floor, threshold=None):
        return self.distribution_on(shape).expected_mass(weights, mass_floor,
            self.config.background_threshold if threshold is None else threshold)

    def view(self):
        distribution = self.distribution_on(self.config.grid_shape)
        density = distribution.expected_excess(self.config.background_threshold)
        mass = density / density.sum() if density.sum() > 1e-15 else np.zeros_like(density)
        return BeliefView(self.time_seconds, distribution.mean, np.sqrt(distribution.variance), mass)

    def diagnostics(self):
        weights = self.weights
        bias = weights @ self.offsets
        centered = self.offsets - bias
        covariance = (centered.T * weights) @ centered
        return {"hypothesis_count": len(weights), "weights": weights.tolist(),
                "log_weights": self.log_weights.tolist(), "bias_offsets_yx": self.offsets.tolist(),
                "inferred_bias_yx": bias.tolist(), "bias_covariance_yx": covariance.tolist(),
                "effective_hypothesis_count": float(1 / np.sum(weights**2)),
                "maximum_weight": float(weights.max())}
