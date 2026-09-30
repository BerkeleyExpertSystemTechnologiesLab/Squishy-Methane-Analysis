"""Finite linear-Gaussian signal inference; no source, evaluator, or I/O access.

Spatial bilinear interpolation and footprint quadrature are declared numerical
approximations. Gaussian conditioning is exact for the resulting finite model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from plume_focus.sensing_contracts import CausalWind, Footprint, Message, Position, SensorReading, finite, position


@dataclass(frozen=True)
class SignalModelConfig(Message):
    grid_shape: tuple[int, int] = (12, 16)
    signal_units: str = "synthetic_normalized_signal"
    gain: float = 1.0
    offset: float = 0.0
    sensor_noise_std: float = 0.05
    initial_mean: float = 0.0
    initial_std: float = 0.2
    correlation_length: float = 0.08
    process_std_per_sqrt_second: float = 0.01
    diffusivity: float = 0.00005
    source_rate: float = 0.15
    source_sigma: float = 0.035
    background_threshold: float = 0.0
    max_transition_seconds: float = 0.5

    def __post_init__(self) -> None:
        if len(self.grid_shape) != 2 or any(type(n) is not int or n < 2 for n in self.grid_shape):
            raise ValueError("Inference grid needs at least two cells on each axis.")
        if math.prod(self.grid_shape) > 1024:
            raise ValueError("Finite covariance is limited to 1024 states; use a coarse inference grid.")
        if not self.signal_units:
            raise ValueError("signal_units is required.")
        for name in ("gain", "sensor_noise_std", "initial_std", "correlation_length", "source_sigma", "max_transition_seconds"):
            finite(name, getattr(self, name), positive=True)
        for name in ("process_std_per_sqrt_second", "diffusivity", "source_rate"):
            finite(name, getattr(self, name))
        for name in ("offset", "initial_mean", "background_threshold"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite.")


@dataclass(frozen=True)
class BeliefView:
    time_seconds: float
    mean: np.ndarray
    std: np.ndarray
    mass: np.ndarray


@dataclass(frozen=True)
class ExpectedSignalMass:
    mean: np.ndarray
    variance: np.ndarray
    density: np.ndarray
    weighted_mass: np.ndarray
    total: float
    normalized: np.ndarray | None


def grid_positions(shape: tuple[int, int]) -> np.ndarray:
    y, x = np.indices(shape, dtype=float)
    return np.stack(((y + 0.5) / shape[0], (x + 0.5) / shape[1]), axis=-1).reshape(-1, 2)


def interpolation_stencil(points: np.ndarray, shape: tuple[int, int], *, outside_zero: bool = False):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    valid = np.all((points >= 0) & (points <= 1), axis=1)
    coords = np.clip(points * np.array(shape) - 0.5, 0, np.array(shape) - 1)
    low = np.floor(coords).astype(int)
    high = np.minimum(low + 1, np.array(shape) - 1)
    frac = coords - low
    indices, weights = [], []
    for iy, ix in ((0, 0), (0, 1), (1, 0), (1, 1)):
        yy = high[:, 0] if iy else low[:, 0]
        xx = high[:, 1] if ix else low[:, 1]
        indices.append(yy * shape[1] + xx)
        weight = (frac[:, 0] if iy else 1 - frac[:, 0]) * (frac[:, 1] if ix else 1 - frac[:, 1])
        weights.append(weight * valid if outside_zero else weight)
    return np.stack(indices, axis=1), np.stack(weights, axis=1)


TRANSPORT_IDS = ("legacy_clipped_v1", "zero_inflow_v1")


def advection_stencil(points, shape, transport_id="legacy_clipped_v1"):
    """Cell-centred transport only; observation interpolation is unchanged.

    Zero exterior cells retain their missing interpolation weight. Clipping is
    used only to make array indices safe, never to repeat an incoming edge value.
    For uniform translation this is the positive, conservative upwind remap.
    """
    if transport_id not in TRANSPORT_IDS:
        raise ValueError("Unknown transport identifier.")
    if transport_id == "legacy_clipped_v1":
        return interpolation_stencil(points, shape, outside_zero=True)
    coords = np.asarray(points).reshape(-1, 2) * np.asarray(shape) - .5
    low = np.floor(coords).astype(int)
    frac = coords-low
    indices, weights = [], []
    for iy, ix in ((0, 0), (0, 1), (1, 0), (1, 1)):
        cell = low + (iy, ix)
        valid = np.all((cell >= 0) & (cell < np.asarray(shape)), axis=1)
        safe = np.clip(cell, 0, np.asarray(shape)-1)
        indices.append(safe[:, 0]*shape[1]+safe[:, 1])
        weights.append(valid*(frac[:, 0] if iy else 1-frac[:, 0])
                       *(frac[:, 1] if ix else 1-frac[:, 1]))
    return np.stack(indices, axis=1), np.stack(weights, axis=1)


def footprint_points(center: Position, footprint: Footprint) -> np.ndarray:
    """Four-point rectangular Gauss quadrature; clip footprint to the FOV."""
    lower = np.maximum(0, np.array(center) - np.array((footprint.height, footprint.width)) / 2)
    upper = np.minimum(1, np.array(center) + np.array((footprint.height, footprint.width)) / 2)
    mid, half = (lower + upper) / 2, (upper - lower) / 2
    return np.array([mid + half * np.array((y, x)) / math.sqrt(3) for y in (-1, 1) for x in (-1, 1)])


def observation_weights(shape: tuple[int, int], center: Position, footprint: Footprint) -> np.ndarray:
    position(center)
    indices, weights = interpolation_stencil(footprint_points(center, footprint), shape)
    result = np.zeros(math.prod(shape))
    np.add.at(result, indices.ravel(), weights.ravel() / len(indices))
    return result


def positive_expectation(mean: np.ndarray, variance: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    delta = np.asarray(mean, dtype=float) - threshold
    sigma = np.sqrt(np.maximum(variance, 0))
    z = np.divide(delta, sigma, out=np.zeros_like(delta), where=sigma > 1e-15)
    cdf = np.fromiter((0.5 * (1 + math.erf(v / math.sqrt(2))) for v in z.ravel()), float).reshape(z.shape)
    expected = delta * cdf + sigma * np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    return np.where(sigma > 1e-15, np.maximum(expected, 0), np.maximum(delta, 0))


class GaussianSignalBelief:
    def __init__(self, config: SignalModelConfig = SignalModelConfig(), *, time_seconds: float = 0.0,
                 transport_id: str = "legacy_clipped_v1"):
        finite("belief time", time_seconds)
        if transport_id not in TRANSPORT_IDS:
            raise ValueError("Unknown transport identifier.")
        self.transport_id = transport_id
        self.config = config
        self.time_seconds = time_seconds
        self.positions = grid_positions(config.grid_shape)
        distance2 = np.sum((self.positions[:, None] - self.positions[None, :]) ** 2, axis=-1)
        self.kernel = np.exp(-distance2 / (2 * config.correlation_length**2))
        self.mean = np.full(len(self.positions), config.initial_mean, dtype=float)
        self.covariance = config.initial_std**2 * self.kernel + np.eye(len(self.positions)) * 1e-12

    def clone(self):
        clone = object.__new__(type(self))
        clone.config, clone.time_seconds = self.config, self.time_seconds
        clone.transport_id = self.transport_id
        clone.positions, clone.kernel = self.positions, self.kernel
        clone.mean, clone.covariance = self.mean.copy(), self.covariance.copy()
        return clone

    def _propagation_factors(self, dt):
        """Historical no-decay coefficients; versioned models may override."""
        return 1., dt, dt

    def condition(self, value: float, weights: np.ndarray, noise_variance: float) -> None:
        h = np.asarray(weights, dtype=float)
        if h.shape != self.mean.shape or np.any(~np.isfinite(h)) or np.any(h < 0) or not np.isclose(h.sum(), 1):
            raise ValueError("Observation weights must be finite, nonnegative, and sum to one.")
        if not math.isfinite(value):
            raise ValueError("Observation must be finite.")
        finite("noise variance", noise_variance, positive=True)
        cross = self.covariance @ h
        innovation_variance = float(h @ cross + noise_variance)
        self.mean += cross * ((value - float(h @ self.mean)) / innovation_variance)
        self.covariance -= np.outer(cross, cross) / innovation_variance
        self.covariance = (self.covariance + self.covariance.T) * 0.5

    def assimilate(self, reading: SensorReading, wind: CausalWind, source: Position | None = None) -> None:
        self.advance(reading.acquired_at_seconds, wind, source)
        h = observation_weights(self.config.grid_shape, reading.position_yx, reading.footprint)
        self.condition((reading.value - self.config.offset) / self.config.gain, h,
                       (self.config.sensor_noise_std / self.config.gain) ** 2)

    def _linear_step(self, indices: np.ndarray, weights: np.ndarray) -> None:
        self.mean = np.sum(self.mean[indices] * weights, axis=1)
        left = sum(weights[:, k, None] * self.covariance[indices[:, k], :] for k in range(indices.shape[1]))
        self.covariance = sum(left[:, indices[:, k]] * weights[None, :, k] for k in range(indices.shape[1]))

    def advance(self, target_seconds: float, wind: CausalWind, source: Position | None = None) -> None:
        finite("prediction time", target_seconds)
        if target_seconds < self.time_seconds - 1e-10:
            raise ValueError("Belief cannot advance backwards.")
        if source is not None:
            position(source)
        remaining = max(0, target_seconds - self.time_seconds)
        if remaining <= 1e-12:
            self.time_seconds = target_seconds
            return
        shape = self.config.grid_shape
        speed = np.max(np.abs(wind.velocity_yx) * np.array(shape))
        diffusion_rate = self.config.diffusivity * sum(n * n for n in shape)
        max_dt = min(self.config.max_transition_seconds, 0.5 / max(speed, 1e-12), 0.45 / max(diffusion_rate, 1e-12))
        count = max(1, int(math.ceil(remaining / max_dt)))
        dt = remaining / count
        adv_indices, adv_weights = advection_stencil(self.positions - np.array(wind.velocity_yx) * dt, shape, self.transport_id)
        ids = np.arange(math.prod(shape)).reshape(shape)
        offsets = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
        wy, wx = self.config.diffusivity * dt * shape[0]**2, self.config.diffusivity * dt * shape[1]**2
        heat_weights = np.array((1 - 2 * wy - 2 * wx, wy, wy, wx, wx))
        heat_indices = np.stack([np.roll(ids, (dy, dx), axis=(0, 1)).ravel() for dy, dx in offsets], axis=1)
        heat = np.broadcast_to(heat_weights, heat_indices.shape).copy()
        # Absorbing exterior, not periodic wraparound.
        for k, (dy, dx) in enumerate(offsets):
            y, x = np.indices(shape)
            heat[:, k] *= ((y - dy >= 0) & (y - dy < shape[0]) & (x - dx >= 0) & (x - dx < shape[1])).ravel()
        emission = np.zeros(len(self.mean))
        if source is not None:
            emission = self.config.source_rate * np.exp(-np.sum((self.positions - source)**2, axis=1) / (2 * self.config.source_sigma**2))
        for _ in range(count):
            self._linear_step(adv_indices, adv_weights)
            if self.config.diffusivity:
                self._linear_step(heat_indices, heat)
            attenuation, source_dt, noise_dt = self._propagation_factors(dt)
            if attenuation != 1.:
                self.mean *= attenuation
                self.covariance *= attenuation**2
            self.mean += emission * source_dt
            self.covariance += self.config.process_std_per_sqrt_second**2 * noise_dt * self.kernel
        self.covariance = (self.covariance + self.covariance.T) * 0.5
        self.time_seconds = target_seconds

    def moments_on(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        ids, weights = interpolation_stencil(grid_positions(shape), self.config.grid_shape)
        mean = np.sum(self.mean[ids] * weights, axis=1)
        variance = np.zeros(len(ids))
        for j in range(4):
            for k in range(4):
                variance += weights[:, j] * weights[:, k] * self.covariance[ids[:, j], ids[:, k]]
        return mean.reshape(shape), np.maximum(variance, 0).reshape(shape)

    def mass_on(self, shape: tuple[int, int]) -> np.ndarray:
        mean, variance = self.moments_on(shape)
        mass = positive_expectation(mean, variance, self.config.background_threshold) / math.prod(shape)
        total = mass.sum()
        return mass / total if total > 1e-15 else np.zeros(shape)

    def expected_mass_on(self, shape: tuple[int, int], weights: np.ndarray, mass_floor: float) -> ExpectedSignalMass:
        """Explicit v1 quadrature and meaningful-signal floor; legacy mass_on is unchanged."""
        finite("mass floor", mass_floor)
        weights = np.asarray(weights, dtype=float)
        if weights.shape != shape or np.any(~np.isfinite(weights)) or np.any(weights <= 0) or not np.isclose(weights.sum(), 1, atol=1e-10, rtol=0):
            raise ValueError("Mass quadrature must have positive normalized weights on the physical lattice.")
        mean, variance = self.moments_on(shape)
        density = positive_expectation(mean, variance, self.config.background_threshold)
        weighted = weights * density
        total = float(weighted.sum())
        return ExpectedSignalMass(mean, variance, density, weighted, total,
                                  weighted / total if total > mass_floor else None)

    def view(self) -> BeliefView:
        shape = self.config.grid_shape
        return BeliefView(self.time_seconds, self.mean.reshape(shape).copy(),
                          np.sqrt(np.maximum(np.diag(self.covariance), 0)).reshape(shape), self.mass_on(shape))
