"""Detached finite Gaussian-mixture marginals; no controller, truth or I/O."""

from dataclasses import dataclass

import numpy as np

from plume_focus.signal_belief import ExpectedSignalMass, positive_expectation


@dataclass(frozen=True)
class PredictiveDistribution:
    time_seconds: float
    weights: np.ndarray
    component_means: np.ndarray
    component_variances: np.ndarray

    def __post_init__(self):
        for name in ("weights", "component_means", "component_variances"):
            values = np.array(getattr(self, name), dtype=float, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if (self.component_means.shape != self.component_variances.shape
                or self.component_means.shape[0] != len(self.weights)
                or np.any(self.weights < 0) or not np.isclose(self.weights.sum(), 1)
                or any(not np.all(np.isfinite(x)) for x in (self.weights, self.component_means, self.component_variances))
                or np.any(self.component_variances < 0)):
            raise ValueError("Invalid finite predictive mixture.")

    def _average(self, values):
        return np.sum(values * self.weights.reshape((-1,) + (1,) * (values.ndim - 1)), axis=0)

    @property
    def mean(self):
        return self._average(self.component_means)

    @property
    def within_variance(self):
        return self._average(self.component_variances)

    @property
    def between_variance(self):
        return self._average((self.component_means - self.mean)**2)

    @property
    def variance(self):
        return self.within_variance + self.between_variance

    def expected_excess(self, threshold):
        return self._average(positive_expectation(self.component_means, self.component_variances, threshold))

    def cdf(self, value):
        """Pointwise latent-signal CDF, including deterministic component atoms."""
        from scipy.special import ndtr
        value = np.asarray(value, dtype=float)
        if np.any(np.isnan(value)):
            raise ValueError("CDF locations cannot be NaN.")
        sigma = np.sqrt(self.component_variances)
        delta = value - self.component_means
        z = np.divide(delta, sigma, out=np.zeros_like(delta), where=sigma > 0)
        return self._average(np.where(sigma > 0, ndtr(z), delta >= 0))

    def quantile(self, probability):
        """Generalized inverse of the actual finite-mixture CDF; no moment match.

        Component quantiles bracket the mixture quantile. Zero-weight states
        remain in the estimator but do not contribute to the predictive CDF.
        Deterministic atoms use the same right-continuous CDF convention.
        """
        from scipy.special import ndtri
        if not np.isfinite(probability) or not 0 < probability < 1:
            raise ValueError("Quantile probability must be strictly between zero and one.")
        component = self.component_means + ndtri(probability)*np.sqrt(self.component_variances)
        active = self.weights > 0
        lo, hi = component[active].min(axis=0), component[active].max(axis=0)
        if np.count_nonzero(active) == 1:
            return hi.copy()
        for _ in range(64):
            mid = lo + (hi-lo)/2
            left = self.cdf(mid) >= probability
            hi, lo = np.where(left, mid, hi), np.where(left, lo, mid)
            if np.all(hi-lo <= 1e-11*np.maximum(1., np.maximum(np.abs(lo), np.abs(hi)))):
                break
        return hi

    def expected_excess_vectorized(self, threshold):
        """Offline evaluation equivalent, tested against the scalar-erf reference.

        Planning deliberately retains expected_excess and its historical floating
        point arithmetic. This kernel does not alter any evidence or weights.
        """
        from scipy.special import ndtr
        delta = self.component_means - threshold
        sigma = np.sqrt(self.component_variances)
        z = np.divide(delta, sigma, out=np.zeros_like(delta), where=sigma > 1e-15)
        expected = delta * ndtr(z) + sigma * np.exp(-.5 * z*z) / np.sqrt(2*np.pi)
        return self._average(np.where(sigma > 1e-15, np.maximum(expected, 0), np.maximum(delta, 0)))

    def expected_mass(self, cell_weights, mass_floor, threshold):
        density = self.expected_excess(threshold)
        weighted = np.asarray(cell_weights) * density
        total = float(weighted.sum())
        return ExpectedSignalMass(self.mean, self.variance, density, weighted, total,
                                  weighted / total if total > mass_floor else None)

    def envelope(self, kappa):
        radius = kappa * np.sqrt(self.component_variances)
        # Include every hypothesis, even when exp(log_weight) underflows to zero.
        return ((self.component_means - radius).min(axis=0),
                (self.component_means + radius).max(axis=0))
