"""Fixed-evidence source-location mixtures; opt-in research, not a scan policy.

All components share dynamics, noise, footprint and evidence. A fixed source
changes only the additive mean input, so one conditional covariance is exact.
Source uncertainty is represented by discrete weights, not extra process noise.
No simulator, evaluator or filesystem access occurs here.
"""
from __future__ import annotations

import math
from functools import lru_cache
import numpy as np
from scipy.special import logsumexp, ndtr
from scipy.sparse import csr_matrix

from plume_focus.calibrated_scanner import DecayGaussianBelief, decay_factors
from plume_focus.signal_belief import observation_weights, grid_positions, interpolation_stencil, advection_stencil


@lru_cache(maxsize=256)
def transport_step(model, velocity, elapsed, transport_id, decay_rate):
    """Sparse equivalent of the existing transport/diffusion substep."""
    shape = model.grid_shape
    speed = np.max(np.abs(velocity)*np.asarray(shape))
    limit = min(model.max_transition_seconds, .5/max(speed,1e-12),
                .45/max(model.diffusivity*sum(n*n for n in shape),1e-12))
    count = max(1, int(math.ceil(elapsed/limit))); dt = elapsed/count
    n = math.prod(shape)
    ids, weights = advection_stencil(grid_positions(shape)-np.asarray(velocity)*dt, shape, transport_id)
    operator = csr_matrix((weights.ravel(), (np.repeat(np.arange(n),4),ids.ravel())),shape=(n,n))
    if model.diffusivity:
        grid = np.arange(n).reshape(shape)
        offsets = ((0,0),(-1,0),(1,0),(0,-1),(0,1))
        wy, wx = model.diffusivity*dt*shape[0]**2, model.diffusivity*dt*shape[1]**2
        ids = np.stack([np.roll(grid,(dy,dx),axis=(0,1)).ravel() for dy,dx in offsets],axis=1)
        weights = np.broadcast_to([1-2*wy-2*wx,wy,wy,wx,wx],ids.shape).copy()
        y,x = np.indices(shape)
        for k,(dy,dx) in enumerate(offsets):
            weights[:,k] *= ((y-dy>=0)&(y-dy<shape[0])&(x-dx>=0)&(x-dx<shape[1])).ravel()
        operator = csr_matrix((weights.ravel(),(np.repeat(np.arange(n),5),ids.ravel())),shape=(n,n)) @ operator
    attenuation, source_dt, _ = decay_factors(decay_rate,dt)
    return operator*attenuation, source_dt, count


class SourcePriorBank:
    """Several priors over a shared bank of fixed-location Gaussian filters.

This diagnostic supports chronologically acquired evidence. The runner rejects
delayed/out-of-order histories rather than silently smoothing them backwards.
"""
    def __init__(self, model, sources, priors, *, half_life_seconds=6.,
                 transport_id="zero_inflow_v1"):
        self.sources = np.asarray(sources, float)
        probabilities = np.asarray(priors, float)
        if (self.sources.ndim != 2 or self.sources.shape[1] != 2
                or not np.all(np.isfinite(self.sources))
                or np.any((self.sources < 0) | (self.sources > 1))):
            raise ValueError("Sources must be finite normalized y,x coordinates")
        if (probabilities.ndim != 2 or probabilities.shape[1] != len(self.sources)
                or not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0)
                or np.any(probabilities.sum(axis=1) <= 0)):
            raise ValueError("Each source prior needs finite nonnegative mass")
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        self.log_weights = np.full_like(probabilities, -np.inf)
        np.log(probabilities, out=self.log_weights, where=probabilities > 0)
        self.reference = DecayGaussianBelief(model, half_life_seconds=half_life_seconds,
                                           transport_id=transport_id)
        self.means = np.tile(self.reference.mean, (len(self.sources), 1))
        self.emissions = model.source_rate * np.exp(
            -np.sum((self.reference.positions[None] - self.sources[:, None])**2, axis=2)
            / (2 * model.source_sigma**2))

    @property
    def weights(self):
        return np.exp(self.log_weights)

    def _forecast(self, means, emissions, target, wind):
        state = self.reference
        if target < state.time_seconds - 1e-10:
            raise ValueError("Cannot forecast backwards")
        elapsed = max(0., target - state.time_seconds)
        result = means.copy()
        if elapsed <= 1e-12:
            return result
        operator, source_dt, count = transport_step(state.config, tuple(wind.velocity_yx),
            elapsed, state.transport_id, state.decay_rate)
        for _ in range(count):
            result = (operator @ result.T).T + emissions * source_dt
        return result

    def advance(self, target, wind):
        self.means = self._forecast(self.means, self.emissions, target, wind)
        # Its mean is unused; its covariance has exactly the same recursion as
        # every source component, with no mixture-collapse approximation.
        self.reference.advance(target, wind, None)

    def forecast_aggregate(self, target, wind):
        weights = self.weights
        return self._forecast(weights @ self.means, weights @ self.emissions, target, wind)

    def condition(self, value, h, noise_variance):
        h = np.asarray(h, float)
        if (h.shape != self.reference.mean.shape or not np.all(np.isfinite(h))
                or np.any(h < 0) or not np.isclose(h.sum(), 1)
                or not np.isfinite(value) or not np.isfinite(noise_variance)
                or noise_variance <= 0):
            raise ValueError("Invalid Gaussian observation")
        cross = self.reference.covariance @ h
        variance = float(h @ cross + noise_variance)
        predicted = self.means @ h
        residual = value - predicted
        likelihood = -.5 * (math.log(2 * math.pi * variance) + residual**2 / variance)
        log_evidence = logsumexp(self.log_weights + likelihood[None], axis=1)
        predictive_mean = self.weights @ predicted
        predictive_cdf = self.weights @ ndtr(residual / math.sqrt(variance))
        self.log_weights += likelihood[None] - log_evidence[:, None]
        self.means += residual[:, None] * (cross / variance)[None]
        self.reference.condition(value, h, noise_variance)
        return {"log_predictive_density": log_evidence,
                "prediction": predictive_mean, "predictive_cdf": predictive_cdf}

    def assimilate(self, reading, wind):
        self.advance(reading.acquired_at_seconds, wind)
        model = self.reference.config
        h = observation_weights(model.grid_shape, reading.position_yx, reading.footprint)
        return self.condition((reading.value-model.offset)/model.gain, h,
                              (model.sensor_noise_std/model.gain)**2)

    def projected_snapshot(self, target, wind, shape, truth_values=None):
        means = self._forecast(self.means, self.emissions, target, wind)
        state = self.reference.clone()
        state.advance(target, wind, None)
        ids, interpolation = interpolation_stencil(grid_positions(shape), state.config.grid_shape)
        projected = sum(means[:, ids[:, k]] * interpolation[:, k] for k in range(4))
        _, within = state.moments_on(shape)
        within = within.ravel()
        probabilities = self.weights
        mean = probabilities @ projected
        between = np.maximum(0., probabilities @ (projected**2) - mean**2)
        result = {"mean": mean, "std": np.sqrt(within[None]+between),
                  "within_variance": within, "between_variance": between,
                  "weights": probabilities.copy()}
        if truth_values is not None:
            cdf = probabilities @ ndtr((np.asarray(truth_values)[None]-projected)
                                       / np.sqrt(np.maximum(within, 1e-30))[None])
            result["coverage95"] = np.mean((cdf >= .025) & (cdf <= .975), axis=1)
        return result
