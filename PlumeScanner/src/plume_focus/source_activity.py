"""Evidence-driven source-activity inference over the calibrated decay belief.

Maintains two conditional beliefs -- an *active* model that keeps injecting the
declared source and a *ceased* model that does not -- plus a scalar log-odds of
activity. Each delivered reading updates the log-odds by the innovation
likelihood ratio of the two models *before* both are conditioned on it. The
decision surface (mean/covariance and all derived views) is the
posterior-weighted Gaussian mixture.

This is source-*anchored* activity inference: it asks whether a *declared known*
source position is still emitting, not whether an unknown source exists
(Exclusion Principle). It never removes the continuing-source signal to gain a
false recovery advantage; the ceased model simply has no injection, so the
active/ceased distinction is driven by evidence, not by zeroing the belief.

The shared acquisition engine only calls `advance`, `assimilate`, `clone`,
`view`, and `moments_on` (plus the predictive query helpers). This binding keeps
`self.mean`/`self.covariance` synchronized to the mixture, so the inherited
`moments_on`, `mass_on`, `expected_mass_on`, and `view` read the activity-weighted
state without any change to the controller, gate, or evaluation layers.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

import numpy as np

from plume_focus.calibrated_scanner import (
    CalibrationSettings,
    CalibrationSession,
    CalibratedScanner,
    DecayGaussianBelief,
)
from plume_focus.full_fov_loop import FullFovSession, FullFovSettings
from plume_focus.sensing_contracts import CausalWind, Position, SensorReading, finite
from plume_focus.sensing_session import InstrumentProfileV1, required_fields
from plume_focus.signal_belief import SignalModelConfig, observation_weights

SOURCE_ACTIVITY_FORMULATION_ID = "beam_reconstruction_source_activity_v1_candidate"
SOURCE_ACTIVITY_POLICY_ID = "full_fov_source_activity_loop_v1_candidate"


def _innovation_log_likelihood(mean, covariance, value, weights, noise_variance):
    """Gaussian innovation log-likelihood of one model for one reading."""
    innovation = value - float(weights @ mean)
    variance = float(weights @ (covariance @ weights)) + noise_variance
    return -0.5 * (innovation * innovation / variance + math.log(2.0 * math.pi * variance))


class SourceActivityBelief(DecayGaussianBelief):
    def __init__(self, config, *, half_life_seconds=None, prior_active_probability=0.5, **kwargs):
        super().__init__(config, half_life_seconds=half_life_seconds, **kwargs)
        if not math.isfinite(prior_active_probability) or not 0.0 < prior_active_probability < 1.0:
            raise ValueError("Prior active probability must lie strictly inside (0, 1).")
        self.prior_active_probability = float(prior_active_probability)
        self.log_odds = math.log(self.prior_active_probability / (1.0 - self.prior_active_probability))
        self.belief_on = DecayGaussianBelief(config, half_life_seconds=half_life_seconds,
                                             transport_id=self.transport_id, time_seconds=self.time_seconds)
        self.belief_off = DecayGaussianBelief(config, half_life_seconds=half_life_seconds,
                                              transport_id=self.transport_id, time_seconds=self.time_seconds)
        self._sync_mixture()

    @property
    def activity_probability(self):
        """Posterior probability the declared source is still active."""
        z = self.log_odds
        if z >= 0.0:
            return 1.0 / (1.0 + math.exp(-z))
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)

    def _mixture(self):
        p = self.activity_probability
        delta = self.belief_on.mean - self.belief_off.mean
        mean = p * self.belief_on.mean + (1.0 - p) * self.belief_off.mean
        covariance = (p * self.belief_on.covariance
                      + (1.0 - p) * self.belief_off.covariance
                      + p * (1.0 - p) * np.outer(delta, delta))
        covariance = (covariance + covariance.T) * 0.5
        return mean, covariance

    def _sync_mixture(self):
        self.mean, self.covariance = self._mixture()

    def advance(self, target_seconds: float, wind: CausalWind, source: Position | None = None) -> None:
        self.belief_on.advance(target_seconds, wind, source)
        self.belief_off.advance(target_seconds, wind, None)
        self.time_seconds = target_seconds
        self._sync_mixture()

    def condition(self, value: float, weights, noise_variance: float) -> None:
        self.belief_on.condition(value, weights, noise_variance)
        self.belief_off.condition(value, weights, noise_variance)
        self._sync_mixture()

    def assimilate(self, reading: SensorReading, wind: CausalWind, source: Position | None = None) -> None:
        self.advance(reading.acquired_at_seconds, wind, source)
        weights = observation_weights(self.config.grid_shape, reading.position_yx, reading.footprint)
        value = (reading.value - self.config.offset) / self.config.gain
        noise_variance = (self.config.sensor_noise_std / self.config.gain) ** 2
        self.log_odds += (
            _innovation_log_likelihood(self.belief_on.mean, self.belief_on.covariance, value, weights, noise_variance)
            - _innovation_log_likelihood(self.belief_off.mean, self.belief_off.covariance, value, weights, noise_variance)
        )
        self.condition(value, weights, noise_variance)

    def clone(self):
        clone = super().clone()
        clone.prior_active_probability = self.prior_active_probability
        clone.log_odds = self.log_odds
        clone.belief_on = self.belief_on.clone()
        clone.belief_off = self.belief_off.clone()
        return clone


@dataclass(frozen=True)
class SourceActivitySession(CalibrationSession):
    """Calibrated estimator with the source-activity belief; gate/policy unchanged."""

    settings: CalibrationSettings
    policy_id: str = SOURCE_ACTIVITY_POLICY_ID
    formulation_id: str = SOURCE_ACTIVITY_FORMULATION_ID
    prior_active_probability: float = 0.5

    def __post_init__(self):
        if not isinstance(self.settings, CalibrationSettings):
            raise TypeError("CalibrationSettings are required for the source-activity binding.")
        # Reuse the parent validation through an equivalent default-policy session;
        # only the formulation/policy identity and the activity prior differ.
        original = FullFovSession(self.instrument, self.model,
            FullFovSettings(**{f.name: getattr(self.settings, f.name) for f in fields(FullFovSettings)}),
            self.source_position, self.home_yx, self.transport_id, clock_origin=self.clock_origin)
        for name in ("instrument", "model", "source_position", "home_yx"):
            object.__setattr__(self, name, getattr(original, name))
        if self.policy_id != SOURCE_ACTIVITY_POLICY_ID or self.formulation_id != SOURCE_ACTIVITY_FORMULATION_ID:
            raise ValueError("Unsupported source-activity session binding.")
        if self.half_life_seconds is not None:
            finite("half life", self.half_life_seconds, positive=True)
        if not math.isfinite(self.prior_active_probability) or not 0.0 < self.prior_active_probability < 1.0:
            raise ValueError("Prior active probability must lie strictly inside (0, 1).")

    @classmethod
    def _from_fields(cls, value):
        required_fields(cls, value)
        return cls(instrument=InstrumentProfileV1.from_dict(value["instrument"]),
                   model=SignalModelConfig._from_fields(value["model"]),
                   settings=CalibrationSettings._from_fields(value["settings"]),
                   **{k: v for k, v in value.items() if k not in ("instrument", "model", "settings")})


class SourceActivityScanner(CalibratedScanner):
    """Calibrated scanner whose belief infers whether the declared source is active."""

    def __init__(self, session):
        if not isinstance(session, SourceActivitySession):
            raise TypeError("SourceActivitySession required.")
        super().__init__(session)
        self._belief = SourceActivityBelief(session.model, half_life_seconds=session.half_life_seconds,
                                            prior_active_probability=session.prior_active_probability,
                                            transport_id=session.transport_id)

    @property
    def activity_probability(self) -> float:
        return float(self._belief.activity_probability)

    @property
    def activity_log_odds(self) -> float:
        return float(self._belief.log_odds)
