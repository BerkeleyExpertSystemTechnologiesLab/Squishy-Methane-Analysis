"""Opt-in evidence-gated full-FoV/focus lifecycle over the shared acquisition engine.

The mode decision changes only at completed-scan boundaries: evidence controls
*whether* to focus, the calibrated predicted upper-signal field still controls
*where*. Full scans use their complete physical duration; a widening action means
the charged return home and another complete full-FoV scan. The calibrated
estimator, transport and physical compiler are unchanged and shared.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

from plume_focus.calibrated_scanner import CalibratedScanner, CalibrationSession, CalibrationSettings
from plume_focus.evidence_gate import (
    DEFAULT_ENTRY_LEVEL, DEFAULT_GATE_ALPHA, DEFAULT_RETAIN_LEVEL, EVIDENCE_GATE_POLICY_ID, EVIDENCE_STATES,
    EvidenceGateState, GateActionDiagnostic, GateActionEvidence, GateDecision, GateReadingEvidence, GateTransition,
    action_diagnostic, build_action_evidence, calibrated_value, entry_decision, full_action_alpha, reading_bounds,
    renewal_decision,
)
from plume_focus.full_fov_loop import FullFovPlan, FullFovSession, FullFovSettings
from plume_focus.sensing_contracts import finite
from plume_focus.sensing_session import InstrumentProfileV1, required_fields
from plume_focus.signal_belief import SignalModelConfig

FORMULATION_ID = "beam_reconstruction_decay_v1_candidate"


@dataclass(frozen=True)
class EvidenceGateSettings(CalibrationSettings):
    """Frozen candidate confidence parameters; not fitted calibration."""
    alpha: float = DEFAULT_GATE_ALPHA
    entry_level: float = DEFAULT_ENTRY_LEVEL
    retain_level: float = DEFAULT_RETAIN_LEVEL

    def __post_init__(self):
        super().__post_init__()
        finite("gate alpha", self.alpha, positive=True)
        if not 0 < self.alpha < 1:
            raise ValueError("The gate alpha must lie strictly inside (0, 1).")
        finite("entry level", self.entry_level)
        finite("retain level", self.retain_level)
        if self.retain_level > self.entry_level:
            raise ValueError("The retain level is a hysteresis floor below the entry level.")


@dataclass(frozen=True)
class EvidenceGatedSession(CalibrationSession):
    """Same estimator binding as the calibrated candidate, different mode gate."""
    settings: EvidenceGateSettings
    policy_id: str = EVIDENCE_GATE_POLICY_ID

    def __post_init__(self):
        if not isinstance(self.settings, EvidenceGateSettings):
            raise TypeError("EvidenceGateSettings are required for the evidence-gated binding.")
        # Reuse the parent validation through an equivalent default-policy session;
        # only the policy identity and the gate settings differ from calibration.
        original = FullFovSession(self.instrument, self.model,
            FullFovSettings(**{f.name: getattr(self.settings, f.name) for f in fields(FullFovSettings)}),
            self.source_position, self.home_yx, self.transport_id, clock_origin=self.clock_origin)
        for name in ("instrument", "model", "source_position", "home_yx"):
            object.__setattr__(self, name, getattr(original, name))
        if self.policy_id != EVIDENCE_GATE_POLICY_ID or self.formulation_id != FORMULATION_ID:
            raise ValueError("Unsupported evidence-gated session binding.")
        if self.half_life_seconds is not None:
            finite("half life", self.half_life_seconds, positive=True)

    @classmethod
    def _from_fields(cls, value):
        required_fields(cls, value)
        return cls(instrument=InstrumentProfileV1.from_dict(value["instrument"]),
                   model=SignalModelConfig._from_fields(value["model"]),
                   settings=EvidenceGateSettings._from_fields(value["settings"]),
                   **{k: v for k, v in value.items() if k not in ("instrument", "model", "settings")})


@dataclass(frozen=True)
class EvidenceGatedPlan(FullFovPlan):
    """Physical plan plus the gate decision and reading-evidence provenance."""
    gate_policy_id: str = EVIDENCE_GATE_POLICY_ID
    gate_evidence_state: str = "unavailable"
    gate_decision_kind: str | None = None
    gate_decision_reason: str | None = None
    gate_entry_level: float = DEFAULT_ENTRY_LEVEL
    gate_retain_level: float = DEFAULT_RETAIN_LEVEL
    gate_applicable_level: float | None = None
    gate_action_index: int | None = None
    gate_alpha_action: float = 0.
    gate_epoch_alpha_allocated: float = 0.
    gate_scheduled_slots: int = 0
    gate_delivered_slots: int = 0
    gate_max_lower_bound: float | None = None
    gate_max_upper_bound: float | None = None
    gate_observed_value_cutoff: float | None = None
    gate_support_reading_ids: tuple[str, ...] = ()
    gate_support_acquired_at_seconds: tuple[float, ...] = ()
    gate_support_available_at_seconds: tuple[float, ...] = ()
    gate_support_age_seconds: tuple[float, ...] = ()
    gate_last_fully_executed: bool | None = None
    gate_contract_violation: bool = False
    gate_fallback_reason: str | None = None
    gate_full_coverage_incomplete: bool = False
    gate_evidence_actions: tuple[GateActionDiagnostic, ...] = ()
    selected_cells: tuple[int, ...] = ()
    padding_cells: tuple[int, ...] = ()
    clipped_selected_cells: tuple[int, ...] = ()
    vertical_margin_fov: float = 0.

    def __post_init__(self):
        super().__post_init__()
        if self.gate_policy_id != EVIDENCE_GATE_POLICY_ID:
            raise ValueError("Unsupported evidence-gate policy identity.")
        if self.gate_evidence_state not in EVIDENCE_STATES:
            raise ValueError("Unsupported diagnostic evidence state.")
        finite("gate application alpha", self.gate_alpha_action)
        finite("gate epoch alpha", self.gate_epoch_alpha_allocated)
        for name in ("gate_scheduled_slots", "gate_delivered_slots"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError("Gate slot diagnostics must be nonnegative integers.")
        if self.gate_delivered_slots > self.gate_scheduled_slots and self.gate_scheduled_slots:
            raise ValueError("Delivered gate slots cannot exceed the frozen schedule.")
        for name in ("gate_max_lower_bound", "gate_max_upper_bound", "gate_observed_value_cutoff", "gate_applicable_level"):
            value = getattr(self, name)
            if value is not None:
                finite(name, value, minimum=-float("inf"))
        if self.mode == "FOCUS":
            if set(self.selected_cells) | set(self.padding_cells) != set(self.requested_cells):
                raise ValueError("Core and padding must partition requested cells.")
            if set(self.selected_cells) & set(self.padding_cells):
                raise ValueError("Overlapping core/padding.")


def gate_slot_map(plan, readings):
    """Map stable reading identities and positions; actual timing is validated by acquisition."""
    expected = {f"{plan.scan_id}:{index}": index for index in range(len(plan.samples))}
    slots, notes = {}, []
    for reading in readings:
        index = expected.get(reading.reading_id)
        if index is None:
            notes.append(f"unmappable_reading:{reading.reading_id}")
            continue
        if index in slots.values():
            notes.append(f"duplicate_slot:{reading.reading_id}")
            continue
        sample = plan.samples[index]
        if reading.scan_id != plan.scan_id:
            notes.append(f"scan_mismatch:{reading.reading_id}")
            continue
        if any(abs(a - b) > 1e-12 for a, b in zip(reading.position_yx, sample.position_yx)):
            notes.append(f"position_mismatch:{reading.reading_id}")
            continue
        slots[reading.reading_id] = index
    return slots, tuple(notes)


def gate_reading_evidence(reading, slot_index, *, gain, offset, sensor_noise_std, alpha_action, scheduled_slots):
    """Bounds for one accepted reading; the same fixed calibration as assimilation."""
    lower, upper = reading_bounds((reading.value,), gain=gain, offset=offset, sensor_noise_std=sensor_noise_std,
                                  alpha_action=alpha_action, scheduled_slots=scheduled_slots)[0]
    return GateReadingEvidence(reading_id=reading.reading_id, slot_index=slot_index, raw_value=reading.value,
        calibrated_value=calibrated_value(reading.value, gain=gain, offset=offset), lower_bound=lower,
        upper_bound=upper, acquired_at_seconds=reading.acquired_at_seconds,
        available_at_seconds=reading.available_at_seconds, position_yx=tuple(reading.position_yx))


class EvidenceGatedScanner(CalibratedScanner):
    """Calibrated estimator with an observation-supported mode decision."""

    def __init__(self, session):
        if not isinstance(session, EvidenceGatedSession):
            raise TypeError("EvidenceGatedSession required.")
        super().__init__(session)
        self._gate_epoch = 1
        self._gate_action_index = 0
        self._gate_epoch_alpha = 0.
        self._gate_full_evidence = []
        self._gate_focus_evidence = None
        self._gate_focus_allowed = False
        self._gate_last_decision = None
        self._gate_decision_evidence = ()
        self._gate_transitions = []
        self._gate_contract_violations = 0
        self._gate_open_action = None
        self._gate_last_fully_executed = None
        self._bound = True

    @property
    def gate_focus_allowed(self) -> bool:
        return bool(self._gate_focus_allowed)

    @property
    def gate_epoch(self) -> int:
        return int(self._gate_epoch)

    @property
    def gate_last_decision(self):
        return self._gate_last_decision

    def gate_state(self) -> EvidenceGateState:
        """Detached, serializable snapshot of the current gate state."""
        return EvidenceGateState(policy_id=EVIDENCE_GATE_POLICY_ID, epoch=self._gate_epoch,
            action_index=self._gate_action_index, epoch_alpha_allocated=self._gate_epoch_alpha,
            focus_allowed=bool(self._gate_focus_allowed), contract_violations=self._gate_contract_violations,
            focus_evidence=self._gate_focus_evidence, last_decision=self._gate_last_decision,
            full_epoch_evidence=tuple(self._gate_full_evidence), transitions=tuple(self._gate_transitions))

    def _focus_supported(self, gate_mean):
        return not self.session.settings.full_scan_only and self.gate_focus_allowed

    def _gate_prospective_action(self, plan):
        """Allocation the opened plan will receive; committed only when a plan actually opens."""
        alpha = self.session.settings.alpha
        if plan.mode == "FULL":
            index = self._gate_action_index + 1 if plan.full_epoch == self._gate_epoch else 1
            return index, full_action_alpha(index, alpha)
        return None, alpha

    def _on_plan_opened(self, plan):
        if plan.mode == "FULL":
            if plan.full_epoch != self._gate_epoch:
                self._gate_epoch = plan.full_epoch
                self._gate_action_index = 0
                self._gate_epoch_alpha = 0.
                self._gate_full_evidence = []
            self._gate_action_index += 1
            action_alpha = full_action_alpha(self._gate_action_index, self.session.settings.alpha)
            self._gate_epoch_alpha += action_alpha
            index = self._gate_action_index
        else:
            action_alpha, index = self.session.settings.alpha, None
        self._gate_open_action = {"scan_id": plan.scan_id, "epoch": plan.full_epoch, "action_index": index,
            "mode": plan.mode, "alpha_action": action_alpha, "scheduled_slots": len(plan.samples)}

    def _on_scan_finished(self, receipt):
        opened, self._gate_open_action = self._gate_open_action, None
        if opened is None:
            return
        plan, settings, model = self._active, self.session.settings, self.session.model
        readings = sorted(self._pending.values(), key=lambda r: (r.acquired_at_seconds, r.reading_id))
        slots, notes = gate_slot_map(plan, readings)
        evidence = build_action_evidence(scan_id=plan.scan_id, epoch=opened["epoch"],
            action_index=opened["action_index"], mode=plan.mode, alpha_action=opened["alpha_action"],
            scheduled_slots=opened["scheduled_slots"],
            readings=tuple(gate_reading_evidence(r, slots[r.reading_id], gain=model.gain, offset=model.offset,
                sensor_noise_std=model.sensor_noise_std, alpha_action=opened["alpha_action"],
                scheduled_slots=opened["scheduled_slots"]) for r in readings if r.reading_id in slots),
            completed_at_seconds=receipt.completed_at_seconds, received_at_seconds=receipt.received_at_seconds,
            plan_completed_at_seconds=plan.completed_at_seconds, contract_notes=notes)
        self._gate_last_fully_executed = evidence.fully_executed
        if evidence.contract_violation:
            self._gate_contract_violations += 1
        if plan.mode == "FULL":
            self._gate_full_evidence.append(evidence)
            if self._full_missing is None:
                self._record_gate_decision(entry_decision(tuple(self._gate_full_evidence), level=settings.entry_level,
                    decided_at_seconds=receipt.received_at_seconds, offset=model.offset, gain=model.gain,
                    sensor_noise_std=model.sensor_noise_std), tuple(self._gate_full_evidence))
        else:
            self._gate_focus_evidence = evidence
            self._record_gate_decision(renewal_decision(evidence, level=settings.retain_level,
                decided_at_seconds=receipt.received_at_seconds, offset=model.offset, gain=model.gain,
                sensor_noise_std=model.sensor_noise_std), (evidence,))

    def _record_gate_decision(self, decision, evidences):
        previous = bool(self._gate_focus_allowed)
        self._gate_focus_allowed = decision.focus_allowed
        self._gate_last_decision = decision
        self._gate_decision_evidence = evidences
        if previous != bool(decision.focus_allowed):
            self._gate_transitions.append(GateTransition(at_seconds=decision.decided_at_seconds,
                from_mode="FOCUS" if previous else "FULL", to_mode="FOCUS" if decision.focus_allowed else "FULL",
                evidence_state=decision.state, level=decision.level,
                reason=decision.reason or ("reading_supported_entry" if decision.kind == "full_epoch_entry"
                                           else "reading_supported_renewal")))

    def _decorate_plan(self, plan, core):
        settings, decision = self.session.settings, self._gate_last_decision
        index, alpha = self._gate_prospective_action(plan)
        if plan.mode == "FULL":
            epoch_alpha = alpha if plan.full_epoch != self._gate_epoch else self._gate_epoch_alpha + alpha
        else:
            epoch_alpha = self._gate_epoch_alpha
        radius = math.ceil(settings.vertical_margin_fov * plan.lattice_shape[0] - 1e-12)
        rows, cols = plan.lattice_shape
        fallback = None
        if plan.mode in ("FULL", "NONE") and plan.reason in ("signal_not_supported",):
            fallback = (decision.reason if decision is not None else "evidence_unavailable")
        elif plan.mode == "FULL" and plan.reason in ("empty_focus_region", "focus_infeasible"):
            fallback = "focus_infeasible_with_support"
        elif plan.mode == "FULL" and plan.reason == "resume_full_scan":
            fallback = "coverage_incomplete"
        elif plan.mode == "NONE":
            fallback = "incomplete_full_scan_budget"
        return EvidenceGatedPlan(**{f.name: getattr(plan, f.name) for f in fields(FullFovPlan)},
            selected_cells=core,
            padding_cells=tuple(sorted(set(plan.requested_cells) - set(core))) if core else (),
            clipped_selected_cells=tuple(i for i in core if i // cols - radius < 0 or i // cols + radius >= rows),
            vertical_margin_fov=settings.vertical_margin_fov,
            gate_evidence_state=(decision.state if decision is not None else "unavailable"),
            gate_decision_kind=(decision.kind if decision is not None else None),
            gate_decision_reason=(decision.reason if decision is not None else None),
            gate_entry_level=settings.entry_level, gate_retain_level=settings.retain_level,
            gate_applicable_level=(decision.level if decision is not None else None),
            gate_action_index=index, gate_alpha_action=alpha, gate_epoch_alpha_allocated=epoch_alpha,
            gate_scheduled_slots=(decision.scheduled_slots if decision is not None else 0),
            gate_delivered_slots=(decision.delivered_slots if decision is not None else 0),
            gate_max_lower_bound=(decision.max_lower_bound if decision is not None else None),
            gate_max_upper_bound=(decision.max_upper_bound if decision is not None else None),
            gate_observed_value_cutoff=(decision.observed_value_cutoff if decision is not None else None),
            gate_support_reading_ids=(decision.support_reading_ids if decision is not None else ()),
            gate_support_acquired_at_seconds=(decision.support_acquired_at_seconds if decision is not None else ()),
            gate_support_available_at_seconds=(decision.support_available_at_seconds if decision is not None else ()),
            gate_support_age_seconds=(decision.support_age_seconds if decision is not None else ()),
            gate_last_fully_executed=self._gate_last_fully_executed,
            gate_contract_violation=(decision.contract_violation if decision is not None else False),
            gate_fallback_reason=fallback, gate_full_coverage_incomplete=bool(self._full_missing is not None),
            gate_evidence_actions=tuple(action_diagnostic(e, level=decision.level, gain=self.session.model.gain,
                offset=self.session.model.offset, sensor_noise_std=self.session.model.sensor_noise_std)
                for e in self._gate_decision_evidence))
