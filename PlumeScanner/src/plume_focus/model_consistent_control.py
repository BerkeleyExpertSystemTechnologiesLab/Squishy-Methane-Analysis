"""Evaluator-only linear Gaussian truth generator for a frozen acquisition path.

This samples the estimator's discrete model, not the physical/synthetic plume.
It never conditions truth on observations. Signed Gaussian states are intentional.
The stencil implementation is independent of SourcePriorBank.transport_step.
"""
import math
import numpy as np
from plume_focus.calibrated_scanner import DecayGaussianBelief, decay_factors
from plume_focus.signal_belief import advection_stencil, observation_weights


class MatchedFieldGenerator:
    def __init__(self, model, source, *, half_life_seconds, transport_id, seed=None):
        template = DecayGaussianBelief(model, half_life_seconds=half_life_seconds,
                                      transport_id=transport_id)
        self.model, self.source = model, np.asarray(source, float)
        self.positions, self.transport_id = template.positions, transport_id
        self.decay_rate = template.decay_rate
        self.rng = None if seed is None else np.random.default_rng(seed)
        self.state = template.mean.copy()
        self.kernel_factor = np.linalg.cholesky(template.kernel)
        if self.rng is not None:
            self.state += np.linalg.cholesky(template.covariance) @ self.rng.standard_normal(len(self.state))
        self.emission = model.source_rate * np.exp(-np.sum((self.positions-self.source)**2, axis=1)
                                                   / (2*model.source_sigma**2))
        self.time_seconds = 0.

    def advance(self, target, wind):
        if not np.isfinite(target) or target < self.time_seconds-1e-10:
            raise ValueError('Invalid forward time')
        elapsed = max(0., target-self.time_seconds)
        if elapsed <= 1e-12:
            self.time_seconds = target
            return
        model = self.model; shape = model.grid_shape
        speed = np.max(np.abs(wind.velocity_yx)*np.array(shape))
        limit = min(model.max_transition_seconds, .5/max(speed,1e-12),
                    .45/max(model.diffusivity*sum(n*n for n in shape),1e-12))
        count = max(1, int(math.ceil(elapsed/limit))); dt = elapsed/count
        ids, weights = advection_stencil(self.positions-np.array(wind.velocity_yx)*dt,
                                         shape, self.transport_id)
        attenuation, source_dt, noise_dt = decay_factors(self.decay_rate,dt)
        wy, wx = model.diffusivity*dt*shape[0]**2, model.diffusivity*dt*shape[1]**2
        for _ in range(count):
            self.state = np.sum(self.state[ids]*weights,axis=1)
            if model.diffusivity:
                a = np.pad(self.state.reshape(shape),1)
                self.state = ((1-2*wy-2*wx)*a[1:-1,1:-1] + wy*(a[:-2,1:-1]+a[2:,1:-1])
                              +wx*(a[1:-1,:-2]+a[1:-1,2:])).ravel()
            self.state = self.state*attenuation+self.emission*source_dt
            if self.rng is not None:
                self.state += model.process_std_per_sqrt_second*math.sqrt(noise_dt)*(
                    self.kernel_factor @ self.rng.standard_normal(len(self.state)))
        self.time_seconds = target

    def observe(self, reading):
        if abs(reading.acquired_at_seconds-self.time_seconds)>1e-10:
            raise ValueError('Advance truth to acquisition time first')
        h = observation_weights(self.model.grid_shape, reading.position_yx, reading.footprint)
        value = self.model.gain*float(h@self.state)+self.model.offset
        if self.rng is not None: value += self.model.sensor_noise_std*self.rng.standard_normal()
        return value
