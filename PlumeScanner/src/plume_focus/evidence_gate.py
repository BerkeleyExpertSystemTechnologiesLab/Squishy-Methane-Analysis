"""Observation-only evidence gate for the opt-in focused-scanning candidate.

The gate decides *whether* a reading-supported reason to focus exists. It reads
only delivered scalar readings, the fixed detector calibration and scheduled
action identities; the existing predicted upper-signal field still decides
*where* to focus. It never reads the posterior, the source model, the planner,
wind-derived pseudo-readings or evaluator truth, and it cannot change the
estimator.

The bounds below are Bonferroni measurement-error bounds on each clean exposure
response, conditional on the declared residual-noise model. They are NOT
calibrated field posterior bounds, instantaneous full-scene bounds, session-wide
error guarantees or tests of whether an emission source exists. Neighbouring
signal values need no independence for the union bound, but correlated or biased
detector errors would still require valid marginal error assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from scipy.special import ndtri

from plume_focus.sensing_contracts import Message, finite
from plume_focus.sensing_session import required_fields

EVIDENCE_GATE_POLICY_ID = "evidence_gated_focus_loop_v3_candidate"
DEFAULT_GATE_ALPHA = .05
DEFAULT_ENTRY_LEVEL = .15
DEFAULT_RETAIN_LEVEL = .10
EVIDENCE_STATES = ("supported", "below_level_on_observed_footprints", "inconclusive", "unavailable")


def normal_quantile(probability: float) -> float:
    """Standard normal quantile; the only distributional assumption in the gate."""
    if isinstance(probability, bool) or not 0. < probability < 1.:
        raise ValueError("A quantile probability must lie strictly inside (0, 1).")
    return float(ndtri(probability))


def full_action_alpha(action_index: int, alpha: float = DEFAULT_GATE_ALPHA) -> float:
    """Alpha for one-based physical action p in a full epoch; the partial sums stay below alpha."""
    if type(action_index) is not int or action_index < 1:
        raise ValueError("The full-epoch action index is one-based.")
    finite("gate alpha", alpha, positive=True)
    if not 0 < alpha < 1:
        raise ValueError("The gate alpha must lie strictly inside (0, 1).")
    return alpha / (action_index * (action_index + 1))


def bound_multiplier(alpha_action: float, scheduled_slots: int) -> float:
    """Two-sided Bonferroni multiplier for M scheduled slots in one physical action."""
    finite("action alpha", alpha_action, positive=True)
    if type(scheduled_slots) is not int or scheduled_slots < 1:
        raise ValueError("At least one scheduled acquisition slot is required.")
    if alpha_action >= 1:
        raise ValueError("The action alpha must lie strictly inside (0, 1).")
    return normal_quantile(1 - alpha_action / (2 * scheduled_slots))


def calibrated_value(value: float, *, gain: float, offset: float) -> float:
    """x = (reading - offset) / gain using the same fixed calibration as assimilation."""
    finite("reading value", value, minimum=-float("inf"))
    finite("gain", gain, positive=True)
    finite("offset", offset, minimum=-float("inf"))
    return (value - offset) / gain


def measurement_error_std(sensor_noise_std: float, *, gain: float) -> float:
    """s = sensor_noise_std / gain; the declared detector noise, never simulated noise."""
    finite("sensor noise std", sensor_noise_std)
    finite("gain", gain, positive=True)
    return sensor_noise_std / gain


def reading_bounds(values, *, gain: float, offset: float, sensor_noise_std: float,
                   alpha_action: float, scheduled_slots: int) -> tuple[tuple[float, float], ...]:
    """Per-reading (lower, upper) bounds; M is frozen before execution, not the count returned."""
    multiplier = bound_multiplier(alpha_action, scheduled_slots)
    error = measurement_error_std(sensor_noise_std, gain=gain)
    return tuple((calibrated_value(v, gain=gain, offset=offset) - multiplier * error,
                  calibrated_value(v, gain=gain, offset=offset) + multiplier * error) for v in values)


def observed_value_cutoff(level: float, *, gain: float, offset: float, sensor_noise_std: float,
                          alpha_action: float, scheduled_slots: int) -> float:
    """Raw reading a single observation must exceed to clear the level at this action."""
    finite("level", level)
    return offset + gain * level + bound_multiplier(alpha_action, scheduled_slots) * sensor_noise_std


def classify_bounds(lower, upper, level: float) -> str:
    """Diagnostic evidence state for one level; support uses strict > and equality is not support."""
    finite("level", level)
    lower, upper = tuple(lower), tuple(upper)
    if len(lower) != len(upper):
        raise ValueError("Bounds must pair lower and upper values.")
    if not lower:
        return "unavailable"
    if any(value > level for value in lower):
        return "supported"
    if all(value < level for value in upper):
        return "below_level_on_observed_footprints"
    return "inconclusive"


@dataclass(frozen=True)
class GateReadingEvidence(Message):
    """One accepted delivered reading and its measurement-error bounds."""
    reading_id: str
    slot_index: int
    raw_value: float
    calibrated_value: float
    lower_bound: float
    upper_bound: float
    acquired_at_seconds: float
    available_at_seconds: float
    position_yx: tuple[float, float]

    def __post_init__(self):
        if not self.reading_id or type(self.slot_index) is not int or self.slot_index < 0:
            raise ValueError("A reading identity and scheduled slot index are required.")
        for name in ("raw_value", "calibrated_value", "lower_bound", "upper_bound"):
            finite(name, getattr(self, name), minimum=-float("inf"))
        finite("acquisition midpoint", self.acquired_at_seconds)
        finite("reading availability", self.available_at_seconds)
        if self.lower_bound > self.upper_bound:
            raise ValueError("A lower bound cannot exceed its upper bound.")
        object.__setattr__(self, "position_yx", tuple(float(v) for v in self.position_yx))


@dataclass(frozen=True)
class GateActionEvidence(Message):
    """Evidence accepted from one opened physical action; M is the frozen slot count."""
    scan_id: str
    epoch: int
    action_index: int | None
    mode: Literal["FULL", "FOCUS"]
    alpha_action: float
    scheduled_slots: int
    delivered_slots: int
    bound_multiplier: float
    fully_executed: bool
    contract_violation: bool
    contract_notes: tuple[str, ...]
    completed_at_seconds: float
    received_at_seconds: float
    readings: tuple[GateReadingEvidence, ...] = ()

    def __post_init__(self):
        if not self.scan_id or self.mode not in ("FULL", "FOCUS"):
            raise ValueError("An action identity and mode are required.")
        if type(self.epoch) is not int or self.epoch < 1:
            raise ValueError("Full-scan epochs are one-based.")
        if self.action_index is not None and (type(self.action_index) is not int or self.action_index < 1):
            raise ValueError("Full-epoch action indices are one-based when applicable.")
        for name in ("alpha_action", "bound_multiplier"):
            finite(name, getattr(self, name), positive=True)
        if type(self.scheduled_slots) is not int or self.scheduled_slots < 1:
            raise ValueError("A scheduled slot count is frozen when the plan opens.")
        if type(self.delivered_slots) is not int or not 0 <= self.delivered_slots <= self.scheduled_slots:
            raise ValueError("Delivered slots must lie within the frozen schedule.")
        if self.delivered_slots != len(self.readings):
            raise ValueError("Delivered slot count and accepted readings must agree.")
        for name in ("fully_executed", "contract_violation"):
            if type(getattr(self, name)) is not bool:
                raise ValueError("Execution and contract flags must be boolean.")
        finite("action completion", self.completed_at_seconds)
        finite("action receipt", self.received_at_seconds, self.completed_at_seconds)
        slots = [r.slot_index for r in self.readings]
        if len(set(slots)) != len(slots) or any(not 0 <= s < self.scheduled_slots for s in slots):
            raise ValueError("Accepted readings must occupy distinct scheduled slots.")

    def to_dict(self):
        result = super().to_dict()
        result["readings"] = [reading.to_dict() for reading in self.readings]
        return result

    @classmethod
    def _from_fields(cls, value):
        required_fields(cls, value)
        payload = dict(value)
        readings = tuple(GateReadingEvidence.from_dict(item) for item in payload.pop("readings"))
        payload["contract_notes"] = tuple(payload["contract_notes"])
        return cls(readings=readings, **payload)

    @property
    def max_lower_bound(self) -> float | None:
        return max((r.lower_bound for r in self.readings), default=None)

    @property
    def max_upper_bound(self) -> float | None:
        return max((r.upper_bound for r in self.readings), default=None)

    @property
    def accepted(self) -> bool:
        return bool(self.readings) and not self.contract_violation

    def state(self, level: float) -> str:
        return classify_bounds([r.lower_bound for r in self.readings],
                               [r.upper_bound for r in self.readings], level)

    def supporting(self, level: float) -> tuple[GateReadingEvidence, ...]:
        return tuple(r for r in self.readings if r.lower_bound > level)


@dataclass(frozen=True)
class GateActionDiagnostic(Message):
    """Compact per-action provenance for bounds pooled into a gate decision."""
    scan_id: str
    epoch: int
    action_index: int | None
    alpha_action: float
    scheduled_slots: int
    delivered_slots: int
    bound_multiplier: float
    observed_value_cutoff: float

    def __post_init__(self):
        if not self.scan_id or type(self.epoch) is not int or self.epoch < 1:
            raise ValueError("Evidence diagnostics require an action and epoch identity.")
        if self.action_index is not None and (type(self.action_index) is not int or self.action_index < 1):
            raise ValueError("Evidence action indices are one-based.")
        multiplier = bound_multiplier(self.alpha_action, self.scheduled_slots)
        if type(self.delivered_slots) is not int or not 0 <= self.delivered_slots <= self.scheduled_slots:
            raise ValueError("Evidence delivered slots must fit the frozen schedule.")
        finite("evidence multiplier", self.bound_multiplier, positive=True)
        if abs(multiplier - self.bound_multiplier) > 1e-12:
            raise ValueError("Evidence multiplier must match the declared alpha and slots.")
        finite("evidence cutoff", self.observed_value_cutoff, minimum=-float("inf"))


def action_diagnostic(evidence, *, level, gain, offset, sensor_noise_std):
    return GateActionDiagnostic(scan_id=evidence.scan_id, epoch=evidence.epoch,
        action_index=evidence.action_index, alpha_action=evidence.alpha_action,
        scheduled_slots=evidence.scheduled_slots, delivered_slots=evidence.delivered_slots,
        bound_multiplier=evidence.bound_multiplier,
        observed_value_cutoff=observed_value_cutoff(level, gain=gain, offset=offset,
            sensor_noise_std=sensor_noise_std, alpha_action=evidence.alpha_action,
            scheduled_slots=evidence.scheduled_slots))


@dataclass(frozen=True)
class GateDecision(Message):
    """Gate provenance recorded at one completed-scan boundary."""
    kind: Literal["full_epoch_entry", "focus_renewal"]
    epoch: int
    action_index: int | None
    level: float
    state: str
    focus_allowed: bool
    reason: str | None
    alpha_action: float
    epoch_alpha_allocated: float
    scheduled_slots: int
    delivered_slots: int
    max_lower_bound: float | None
    max_upper_bound: float | None
    observed_value_cutoff: float | None
    support_reading_ids: tuple[str, ...]
    support_acquired_at_seconds: tuple[float, ...]
    support_available_at_seconds: tuple[float, ...]
    support_age_seconds: tuple[float, ...]
    fully_executed: bool
    contract_violation: bool
    decided_at_seconds: float

    def __post_init__(self):
        if self.kind not in ("full_epoch_entry", "focus_renewal") or self.state not in EVIDENCE_STATES:
            raise ValueError("Unsupported gate decision kind or evidence state.")
        if type(self.focus_allowed) is not bool:
            raise ValueError("A gate decision must declare whether focus is allowed.")
        finite("level", self.level)
        finite("decision time", self.decided_at_seconds)
        finite("action alpha", self.alpha_action, positive=True)
        finite("epoch alpha", self.epoch_alpha_allocated, positive=True)
        for name in ("max_lower_bound", "max_upper_bound", "observed_value_cutoff"):
            value = getattr(self, name)
            if value is not None:
                finite(name, value, minimum=-float("inf"))
        if len(set(self.support_reading_ids)) != len(self.support_reading_ids):
            raise ValueError("Supporting reading identities must be unique.")
        for name in ("support_acquired_at_seconds", "support_available_at_seconds", "support_age_seconds"):
            values = getattr(self, name)
            if len(values) != len(self.support_reading_ids):
                raise ValueError("Support provenance must align with supporting readings.")
            for value in values:
                finite("support provenance", value)

    @property
    def reading_supported(self) -> bool:
        return self.state == "supported"


@dataclass(frozen=True)
class GateTransition(Message):
    """One mode-timeline switch caused by the gate."""
    at_seconds: float
    from_mode: Literal["FULL", "FOCUS"]
    to_mode: Literal["FULL", "FOCUS"]
    evidence_state: str
    level: float
    reason: str

    def __post_init__(self):
        finite("transition time", self.at_seconds)
        if self.from_mode not in ("FULL", "FOCUS") or self.to_mode not in ("FULL", "FOCUS") or self.from_mode == self.to_mode:
            raise ValueError("A transition switches between FULL and FOCUS.")
        if self.evidence_state not in EVIDENCE_STATES or not self.reason:
            raise ValueError("A transition records its evidence state and reason.")
        finite("level", self.level)


@dataclass(frozen=True)
class EvidenceGateState(Message):
    """Serializable gate state: epoch evidence, action index, provenance and transitions."""
    policy_id: str
    epoch: int
    action_index: int
    epoch_alpha_allocated: float
    focus_allowed: bool
    contract_violations: int
    focus_evidence: GateActionEvidence | None
    last_decision: GateDecision | None
    full_epoch_evidence: tuple[GateActionEvidence, ...] = ()
    transitions: tuple[GateTransition, ...] = ()

    def __post_init__(self):
        if self.policy_id != EVIDENCE_GATE_POLICY_ID:
            raise ValueError("Unsupported evidence-gate policy identity.")
        if type(self.epoch) is not int or self.epoch < 1 or type(self.action_index) is not int or self.action_index < 0:
            raise ValueError("Invalid gate epoch or action index.")
        finite("epoch alpha allocation", self.epoch_alpha_allocated)
        if type(self.focus_allowed) is not bool or type(self.contract_violations) is not int or self.contract_violations < 0:
            raise ValueError("Invalid gate decision or violation count.")

    def to_dict(self):
        result = super().to_dict()
        result["focus_evidence"] = None if self.focus_evidence is None else self.focus_evidence.to_dict()
        result["last_decision"] = None if self.last_decision is None else self.last_decision.to_dict()
        result["full_epoch_evidence"] = [item.to_dict() for item in self.full_epoch_evidence]
        result["transitions"] = [item.to_dict() for item in self.transitions]
        return result

    @classmethod
    def _from_fields(cls, value):
        required_fields(cls, value)
        payload = dict(value)
        payload["focus_evidence"] = None if payload["focus_evidence"] is None else GateActionEvidence.from_dict(payload["focus_evidence"])
        payload["last_decision"] = None if payload["last_decision"] is None else GateDecision.from_dict(payload["last_decision"])
        payload["full_epoch_evidence"] = tuple(GateActionEvidence.from_dict(item) for item in payload["full_epoch_evidence"])
        payload["transitions"] = tuple(GateTransition.from_dict(item) for item in payload["transitions"])
        return cls(**payload)


def accepted_readings(evidences) -> tuple[GateReadingEvidence, ...]:
    """Union of readings whose physical action satisfied the gate contract."""
    return tuple(reading for evidence in evidences if not evidence.contract_violation
                 for reading in evidence.readings)


def entry_decision(evidences, *, level: float, decided_at_seconds: float, offset: float, gain: float,
                   sensor_noise_std: float) -> GateDecision:
    """Decision at a completed full epoch; historical support from other epochs cannot reopen focus."""
    evidences = tuple(evidences)
    if not evidences:
        raise ValueError("A full-epoch entry decision needs at least one completed action.")
    readings = accepted_readings(evidences)
    violation = any(evidence.contract_violation for evidence in evidences)
    state = classify_bounds([r.lower_bound for r in readings], [r.upper_bound for r in readings], level)
    supporting = tuple(r for r in readings if r.lower_bound > level)
    focus_allowed = state == "supported" and not violation
    reason = None if focus_allowed else ("gate_contract_violation" if violation else "entry_not_supported")
    last = evidences[-1]
    return GateDecision(kind="full_epoch_entry", epoch=last.epoch, action_index=last.action_index, level=level,
                        state=state, focus_allowed=focus_allowed, reason=reason, alpha_action=last.alpha_action,
                        epoch_alpha_allocated=sum(e.alpha_action for e in evidences),
                        scheduled_slots=sum(e.scheduled_slots for e in evidences),
                        delivered_slots=sum(e.delivered_slots for e in evidences),
                        max_lower_bound=max((r.lower_bound for r in readings), default=None),
                        max_upper_bound=max((r.upper_bound for r in readings), default=None),
                        observed_value_cutoff=observed_value_cutoff(level, gain=gain, offset=offset,
                            sensor_noise_std=sensor_noise_std, alpha_action=last.alpha_action,
                            scheduled_slots=last.scheduled_slots),
                        support_reading_ids=tuple(r.reading_id for r in supporting),
                        support_acquired_at_seconds=tuple(r.acquired_at_seconds for r in supporting),
                        support_available_at_seconds=tuple(r.available_at_seconds for r in supporting),
                        support_age_seconds=tuple(decided_at_seconds - r.acquired_at_seconds for r in supporting),
                        fully_executed=all(e.fully_executed for e in evidences), contract_violation=violation,
                        decided_at_seconds=decided_at_seconds)


def renewal_decision(evidence: GateActionEvidence, *, level: float, decided_at_seconds: float, offset: float,
                     gain: float, sensor_noise_std: float) -> GateDecision:
    """Decision after one focused sweep; a partial or conflicting sweep cannot renew focus."""
    if evidence.mode != "FOCUS":
        raise ValueError("Renewal decisions follow a focused sweep.")
    readings = accepted_readings((evidence,))
    state = classify_bounds([r.lower_bound for r in readings], [r.upper_bound for r in readings], level)
    supporting = tuple(r for r in readings if r.lower_bound > level)
    focus_allowed = bool(state == "supported" and evidence.fully_executed and not evidence.contract_violation)
    if focus_allowed:
        reason = None
    elif evidence.contract_violation:
        reason = "gate_contract_violation"
    elif not evidence.fully_executed:
        reason = "focus_partially_executed"
    else:
        reason = "focus_not_renewed"
    return GateDecision(kind="focus_renewal", epoch=evidence.epoch, action_index=evidence.action_index, level=level,
                        state=state, focus_allowed=focus_allowed, reason=reason, alpha_action=evidence.alpha_action,
                        epoch_alpha_allocated=evidence.alpha_action, scheduled_slots=evidence.scheduled_slots,
                        delivered_slots=evidence.delivered_slots,
                        max_lower_bound=max((r.lower_bound for r in readings), default=None),
                        max_upper_bound=max((r.upper_bound for r in readings), default=None),
                        observed_value_cutoff=observed_value_cutoff(level, gain=gain, offset=offset,
                            sensor_noise_std=sensor_noise_std, alpha_action=evidence.alpha_action,
                            scheduled_slots=evidence.scheduled_slots),
                        support_reading_ids=tuple(r.reading_id for r in supporting),
                        support_acquired_at_seconds=tuple(r.acquired_at_seconds for r in supporting),
                        support_available_at_seconds=tuple(r.available_at_seconds for r in supporting),
                        support_age_seconds=tuple(decided_at_seconds - r.acquired_at_seconds for r in supporting),
                        fully_executed=evidence.fully_executed, contract_violation=evidence.contract_violation,
                        decided_at_seconds=decided_at_seconds)


def build_action_evidence(*, scan_id: str, epoch: int, action_index: int | None, mode: str, alpha_action: float,
                          scheduled_slots: int, readings, completed_at_seconds: float, received_at_seconds: float,
                          plan_completed_at_seconds: float, contract_notes=()) -> GateActionEvidence:
    """Assemble accepted bounds for one completed action from already-validated readings."""
    readings = tuple(readings)
    notes = tuple(contract_notes)
    return GateActionEvidence(scan_id=scan_id, epoch=epoch, action_index=action_index, mode=mode,
        alpha_action=alpha_action, scheduled_slots=scheduled_slots, delivered_slots=len(readings),
        bound_multiplier=bound_multiplier(alpha_action, scheduled_slots),
        fully_executed=bool(len(readings) == scheduled_slots and completed_at_seconds >= plan_completed_at_seconds - 1e-9),
        contract_violation=bool(notes), contract_notes=notes, completed_at_seconds=float(completed_at_seconds),
        received_at_seconds=float(received_at_seconds), readings=readings)
