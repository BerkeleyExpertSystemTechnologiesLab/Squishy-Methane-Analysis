"""Explicit known-source evidence requests using the B1 belief and native gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from plume_focus.evidence_gated_loop import EvidenceGatedSession
from plume_focus.evidence_refresh import EvidenceRefreshRequest, EvidenceRefreshScanner
from plume_focus.source_activity import SourceActivityBelief


SOURCE_ACTIVITY_REFRESH_ID = "source_activity_evidence_refresh_v1_candidate"


@dataclass(frozen=True)
class SourceActivityState:
    """Current aggregate model comparison at the controller's causal boundary."""

    active_model_weight: float
    log_odds: float
    belief_time_seconds: float
    controller_time_seconds: float


class SourceActivityRefreshScanner(EvidenceRefreshScanner):
    """Opt-in composition with one native acquisition, gate, and receipt path."""

    candidate_id = SOURCE_ACTIVITY_REFRESH_ID

    def __init__(
        self,
        session: EvidenceGatedSession,
        *,
        delivery: Literal["stream", "batch"],
        prior_active_probability: float = 0.5,
    ):
        if not isinstance(session, EvidenceGatedSession):
            raise TypeError("EvidenceGatedSession required.")
        if session.source_position is None:
            raise ValueError("Source-activity requests require a declared source position.")
        super().__init__(session, delivery=delivery)
        self._belief = SourceActivityBelief(
            session.model,
            half_life_seconds=session.half_life_seconds,
            transport_id=session.transport_id,
            prior_active_probability=prior_active_probability,
        )

    def source_activity_request(
        self,
        request_id: str,
        *,
        fresh_since_seconds: float,
        deadline_seconds: float,
    ) -> EvidenceRefreshRequest:
        """Request an exposure in the physical cell containing the declared source."""
        return EvidenceRefreshRequest(
            request_id=request_id,
            instrument_hash=self.session.instrument.digest,
            cells=(self._physical_cell(self.session.source_position),),
            fresh_since_seconds=fresh_since_seconds,
            deadline_seconds=deadline_seconds,
        )

    def source_activity_state(self) -> SourceActivityState:
        """Report the weight from all accepted readings at the current boundary."""
        return SourceActivityState(
            active_model_weight=float(self._belief.activity_probability),
            log_odds=float(self._belief.log_odds),
            belief_time_seconds=self._belief.time_seconds,
            controller_time_seconds=self._last_available,
        )
