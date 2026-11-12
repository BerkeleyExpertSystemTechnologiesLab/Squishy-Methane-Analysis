"""Single-cell C3 REFRESH for an explicit evidence request to B1."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Literal

import numpy as np

from plume_focus.c3_refresh import clear_focus_after_refresh
from plume_focus.evidence_refresh import EvidenceRefreshDecision, EvidenceRefreshRequest, _TIME_TOLERANCE
from plume_focus.full_fov_loop import FullFovPlan
from plume_focus.physical_scan import PhysicalPlan
from plume_focus.sensing_contracts import CausalWind, Position
from plume_focus.sensing_messages_v2 import ScanRequestV1
from plume_focus.source_activity_refresh import SourceActivityRefreshScanner


LOCAL_EVIDENCE_REFRESH_ID = "source_activity_local_refresh_v1_candidate"


class LocalSourceActivityRefreshScanner(SourceActivityRefreshScanner):
    """Opt-in C3 path selection over the same native B1 acquisition."""

    candidate_id = LOCAL_EVIDENCE_REFRESH_ID

    def plan_local_evidence_refresh(
        self,
        request: EvidenceRefreshRequest,
        now_seconds: float,
        current_position: Position,
        constraints: ScanRequestV1,
        *,
        terminal: Literal["receipt", "home_full"],
        preparation_seconds: float,
        wind: CausalWind | None = None,
    ) -> EvidenceRefreshDecision:
        """Select one legal action and reserve any required native recovery.

        The preparation allowance applies to each opened action. With
        home_full, a local action reserves a subsequent return and FULL from
        its estimated receipt pose and detector state. The adapter plans that
        recovery at the actual receipt using the remaining session budget.
        """
        if terminal not in ("receipt", "home_full"):
            raise ValueError("Declare the required terminal: receipt or home_full.")

        def full_only(reason=None):
            decision = self.plan_evidence_refresh(
                request, now_seconds, current_position, constraints,
                preparation_seconds=preparation_seconds, wind=wind,
            )
            return replace(decision, version=LOCAL_EVIDENCE_REFRESH_ID, local_reason=reason)

        status = self.evidence_status(request, now_seconds=now_seconds)
        if status.state != "pending" or self.active_plan is not None:
            return full_only()
        if len(request.cells) != 1:
            return full_only("multiple_cells")
        if (not self.startup_complete or self._full_missing is not None
                or self.session.settings.full_scan_only or not self.gate_focus_allowed):
            return full_only("full_required")

        self._validate_refresh_context(now_seconds, current_position, constraints, preparation_seconds, wind)
        start = now_seconds + preparation_seconds
        shape = self.capabilities.lattice_shape
        mask = np.zeros(shape, dtype=bool)
        mask.flat[request.cells[0]] = True
        local = self._quote_refresh_path(mask, start, current_position, last=self._last_detector_reading)
        if local.swept_cells != request.cells or len(local.samples) != 1:
            raise RuntimeError("A single-cell REFRESH must compile to exactly one physical exposure.")
        missing = set(status.missing_cells)
        local_acceptance = self._estimated_acceptance(request, missing, local)
        if local_acceptance is None or local_acceptance > request.deadline_seconds + _TIME_TOLERANCE:
            return full_only("window_unreachable")

        local_receipt = self._estimated_receipt(local)
        terminal_at = local_receipt
        if terminal == "home_full":
            recovery = self._quote_refresh_path(
                np.ones(shape, dtype=bool), local_receipt + preparation_seconds,
                local.final_position_yx, last=local.samples[-1], return_home=self.session.home_yx,
                scan_offset=1,
            )
            terminal_at = self._estimated_receipt(recovery)
        required = terminal_at - now_seconds
        if required > constraints.max_duration_seconds + 1e-9:
            return full_only("terminal_budget")

        full = self._quote_full_refresh(start, current_position)
        full_duration = self._estimated_receipt(full) - now_seconds
        full_acceptance = self._estimated_acceptance(request, missing, full)
        if (full_acceptance is not None and full_acceptance <= request.deadline_seconds + _TIME_TOLERANCE
                and full_duration <= constraints.max_duration_seconds + 1e-9
                and full_duration <= required + 1e-9):
            return full_only("full_preferred")

        plan = FullFovPlan(
            **{field.name: getattr(local, field.name) for field in fields(PhysicalPlan)},
            mode="REFRESH", reason="single_cell_evidence_requested",
            gate_max_mean=float(np.max(self.predictive_moments(start, shape)[0])),
            threshold=self.session.settings.threshold, eligible_cells=(), omitted_cells=(),
            full_epoch=self._epoch, return_home_until_seconds=start,
            required_duration_seconds=local_receipt - start,
        )
        self._belief.advance(start, self._wind, self.source_position)
        if wind is not None:
            self._wind = wind
        # Like C3 REFRESH, this action has no FULL-entry or FOCUS-renewal gate record.
        self._gate_open_action = None
        self._open_scan(plan)
        return EvidenceRefreshDecision(
            status=replace(status, active_scan_id=plan.scan_id),
            outcome="scheduled", plan=plan, required_duration_seconds=required,
            estimated_last_acceptance_seconds=local_acceptance, version=LOCAL_EVIDENCE_REFRESH_ID,
        )

    def _on_scan_finished(self, receipt):
        if self.active_plan.mode == "REFRESH":
            clear_focus_after_refresh(self)
        else:
            super()._on_scan_finished(receipt)
