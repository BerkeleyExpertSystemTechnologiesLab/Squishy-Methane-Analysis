"""Source-activity inference with log-odds forgetting (v2 candidate).

Identical to `source_activity.py` in every way except one: the scalar log-odds
of activity is exponentially decayed toward the prior (log-odds 0, i.e. p=0.5)
as time advances. This lets the positive evidence accumulated during an active
phase be forgotten after the source ceases, so the posterior activity
probability can actually fall and a stop can be announced.

Forgetting is applied purely on time advance, before the per-reading innovation
likelihood-ratio update, as ``L <- L * exp(-ln2 / tau * dt)`` with ``tau =
log_odds_half_life_seconds``. With ``tau = None`` (or ``0.0``) the behaviour is
bit-identical to the v1 `SourceActivityBelief`; a finite ``tau`` is the new,
distinct formulation. The belief means/covariances are untouched, so the
on/off models and the mixture decision surface are exactly those of v1.

This is still source-*anchored* activity inference (known declared source, is
it still emitting) -- not unknown-source discovery (Exclusion Principle).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

from plume_focus.calibrated_scanner import CalibrationSettings, CalibratedScanner
from plume_focus.calibration_execution import CalibrationRunConfig
from plume_focus.full_fov_loop import FullFovSession, FullFovSettings
from plume_focus.sensing_contracts import CausalWind, Position, finite
from plume_focus.sensing_session import InstrumentProfileV1, required_fields
from plume_focus.signal_belief import SignalModelConfig
from plume_focus.source_activity import (
    SourceActivityBelief,
    SourceActivityScanner,
    SourceActivitySession,
)

SOURCE_ACTIVITY_FORGETTING_FORMULATION_ID = "beam_reconstruction_source_activity_v2_forgetting_candidate"
SOURCE_ACTIVITY_FORGETTING_POLICY_ID = "full_fov_source_activity_loop_v2_forgetting_candidate"


class SourceActivityForgettingBelief(SourceActivityBelief):
    """SourceActivityBelief plus exponential decay of the activity log-odds."""

    def __init__(self, config, *, half_life_seconds=None, prior_active_probability=0.5,
                 log_odds_half_life_seconds=None, **kwargs):
        super().__init__(config, half_life_seconds=half_life_seconds,
                         prior_active_probability=prior_active_probability, **kwargs)
        if log_odds_half_life_seconds is None:
            self.log_odds_half_life_seconds = None
            self._forget_rate = 0.0
        else:
            finite("log-odds half life", log_odds_half_life_seconds, positive=True)
            self.log_odds_half_life_seconds = float(log_odds_half_life_seconds)
            self._forget_rate = math.log(2.0) / self.log_odds_half_life_seconds

    def advance(self, target_seconds: float, wind: CausalWind, source: Position | None = None) -> None:
        if self._forget_rate > 0.0:
            dt = target_seconds - self.time_seconds
            if dt > 0.0:
                self.log_odds *= math.exp(-self._forget_rate * dt)
        super().advance(target_seconds, wind, source)

    def clone(self):
        clone = super().clone()
        clone.log_odds_half_life_seconds = self.log_odds_half_life_seconds
        clone._forget_rate = self._forget_rate
        return clone


@dataclass(frozen=True)
class SourceActivityForgettingSession(SourceActivitySession):
    """Calibrated estimator with the forgetting source-activity belief."""

    settings: CalibrationSettings
    policy_id: str = SOURCE_ACTIVITY_FORGETTING_POLICY_ID
    formulation_id: str = SOURCE_ACTIVITY_FORGETTING_FORMULATION_ID
    prior_active_probability: float = 0.5
    log_odds_half_life_seconds: float | None = None

    def __post_init__(self):
        if not isinstance(self.settings, CalibrationSettings):
            raise TypeError("CalibrationSettings are required for the source-activity-forgetting binding.")
        original = FullFovSession(self.instrument, self.model,
            FullFovSettings(**{f.name: getattr(self.settings, f.name) for f in fields(FullFovSettings)}),
            self.source_position, self.home_yx, self.transport_id, clock_origin=self.clock_origin)
        for name in ("instrument", "model", "source_position", "home_yx"):
            object.__setattr__(self, name, getattr(original, name))
        if self.policy_id != SOURCE_ACTIVITY_FORGETTING_POLICY_ID or self.formulation_id != SOURCE_ACTIVITY_FORGETTING_FORMULATION_ID:
            raise ValueError("Unsupported source-activity-forgetting session binding.")
        if self.half_life_seconds is not None:
            finite("half life", self.half_life_seconds, positive=True)
        if not math.isfinite(self.prior_active_probability) or not 0.0 < self.prior_active_probability < 1.0:
            raise ValueError("Prior active probability must lie strictly inside (0, 1).")
        if self.log_odds_half_life_seconds is not None:
            finite("log-odds half life", self.log_odds_half_life_seconds, positive=True)


class SourceActivityForgettingScanner(SourceActivityScanner):
    """Calibrated scanner whose belief infers activity with log-odds forgetting."""

    def __init__(self, session):
        if not isinstance(session, SourceActivityForgettingSession):
            raise TypeError("SourceActivityForgettingSession required.")
        CalibratedScanner.__init__(self, session)
        self._belief = SourceActivityForgettingBelief(
            session.model, half_life_seconds=session.half_life_seconds,
            prior_active_probability=session.prior_active_probability,
            log_odds_half_life_seconds=session.log_odds_half_life_seconds,
            transport_id=session.transport_id)


@dataclass(frozen=True)
class SourceActivityForgettingRunConfig(CalibrationRunConfig):
    """Same truth/mechanics/policy as the calibrated control; forgetting belief."""

    session: SourceActivityForgettingSession

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.session, SourceActivityForgettingSession):
            raise TypeError("Source-activity-forgetting session required.")

    @classmethod
    def _from_fields(cls, value):
        required_fields(cls, value)
        from plume_focus.sensing_contracts import _decode
        from typing import get_type_hints
        payload = dict(value)
        session = SourceActivityForgettingSession.from_dict(payload.pop("session"))
        annotations = get_type_hints(cls)
        return cls(session=session, **{k: _decode(annotations[k], v) for k, v in payload.items()})

    def make_controller(self):
        return SourceActivityForgettingScanner(self.session)
