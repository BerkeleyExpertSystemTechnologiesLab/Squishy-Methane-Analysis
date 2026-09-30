"""Transport-v3 risk decisions; no change to legacy mass-target semantics."""

from dataclasses import dataclass, fields
from typing import ClassVar, Literal

from plume_focus.sensing_contracts import Message, finite
from plume_focus.sensing_messages_v2 import ScanPlanV2


@dataclass(frozen=True)
class RiskCandidate(Message):
    proposal: str
    core_cells: tuple[int, ...]
    requested_cells: tuple[int, ...]
    duration_seconds: float
    expected_integrated_map_risk: float
    within_component_risk: float
    between_component_risk: float
    monte_carlo_standard_error: float
    hypothetical_sweeps: int

    def __post_init__(self):
        for name in ("core_cells", "requested_cells"):
            object.__setattr__(self,name,tuple(getattr(self,name)))
            if any(type(i) is not int or i < 0 for i in getattr(self,name)):
                raise ValueError("Candidate cells must be nonnegative integers.")
        if type(self.hypothetical_sweeps) is not int or self.hypothetical_sweeps < 1:
            raise ValueError("An executable candidate needs at least one forecast sweep.")
        for name in ("duration_seconds", "expected_integrated_map_risk", "within_component_risk",
                     "between_component_risk", "monte_carlo_standard_error"):
            finite(name, getattr(self, name))


@dataclass(frozen=True)
class ScanPlanV3(ScanPlanV2):
    schema_version: ClassVar[int] = 3
    status: Literal["value_selected", "startup_full_scene", "reacquisition", "no_legal_action", "candidate"]
    risk_candidates: tuple[RiskCandidate, ...] = ()
    selected_candidate: int | None = None
    lookahead_seconds: float = 0.
    expected_integrated_map_risk: float | None = None
    idle_integrated_map_risk: float | None = None
    risk_step_seconds: float = .1
    history_count: int = 0
    risk_sampling_seed: int = 0
    service_required_cells: tuple[int, ...] = ()
    service_status: str = "not_due"
    selection_criterion: str = "expected_physical_time_map_risk"
    numerical_status: str = "configured_approximation"
    expected_delivery: Literal["stream", "batch"] = "stream"
    expected_delivery_delay_seconds: float = 0.
    expected_completion_overrun_seconds: float = 0.
    lookahead_wind_yx: tuple[float, float] = (0., 0.)

    def __post_init__(self):
        if self.status not in ("value_selected", "startup_full_scene", "reacquisition", "no_legal_action", "candidate"):
            raise ValueError("Unsupported risk decision status.")
        values = {f.name: getattr(self, f.name) for f in fields(ScanPlanV2)}
        ScanPlanV2(**{**values, "status": "candidate" if self.status == "value_selected" else self.status})
        if self.selected_candidate is not None and not 0 <= self.selected_candidate < len(self.risk_candidates):
            raise ValueError("Selected risk candidate is unavailable.")
        if self.status == "value_selected" and (self.selected_candidate is None or self.expected_integrated_map_risk is None):
            raise ValueError("A value-selected plan must identify its scored candidate.")
        if self.expected_delivery not in ("stream", "batch"):
            raise ValueError("Expected delivery must be stream or batch.")
        object.__setattr__(self,"lookahead_wind_yx",tuple(self.lookahead_wind_yx))
        if len(self.lookahead_wind_yx) != 2:
            raise ValueError("Lookahead wind needs two components.")
        for value in self.lookahead_wind_yx:
            finite("lookahead wind",value,minimum=float('-inf'))
        for name in ("expected_delivery_delay_seconds", "expected_completion_overrun_seconds"):
            finite(name,getattr(self,name))
        for name in ("lookahead_seconds", "risk_step_seconds"):
            finite(name, getattr(self, name))
        for name in ("expected_integrated_map_risk", "idle_integrated_map_risk"):
            if getattr(self, name) is not None:
                finite(name, getattr(self, name))


def risk_plan(plan, **changes):
    return ScanPlanV3(**{**{f.name: getattr(plan, f.name) for f in fields(ScanPlanV2)}, **changes})
