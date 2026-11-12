"""Known-source evidence refresh driven by the log-odds-forgetting belief (v2).

`SourceActivityRefreshScanner` (codex) composes the explicit-evidence request
loop with the B1 `SourceActivityBelief`. This module swaps in the B2
`SourceActivityForgettingBelief`, whose activity log-odds exponentially decays
toward the prior on time advance, so a stopped source can actually lower the
posterior and close the request loop. With ``log_odds_half_life_seconds=None``
the belief is bit-identical to B1, so the scanner degrades gracefully to the
codex candidate. This is still source-*anchored* activity inference, not
unknown-source discovery.
"""

from __future__ import annotations

from typing import Literal

from plume_focus.evidence_gated_loop import EvidenceGatedSession
from plume_focus.source_activity_forgetting import SourceActivityForgettingBelief
from plume_focus.source_activity_refresh import SourceActivityRefreshScanner

SOURCE_ACTIVITY_REFRESH_FORGETTING_ID = "source_activity_evidence_refresh_v2_forgetting_candidate"


class SourceActivityForgettingRefreshScanner(SourceActivityRefreshScanner):
    """Refresh request loop whose source-activity belief applies log-odds forgetting."""

    candidate_id = SOURCE_ACTIVITY_REFRESH_FORGETTING_ID

    def __init__(
        self,
        session: EvidenceGatedSession,
        *,
        delivery: Literal["stream", "batch"],
        prior_active_probability: float = 0.5,
        log_odds_half_life_seconds: float | None = None,
    ):
        super().__init__(session, delivery=delivery, prior_active_probability=prior_active_probability)
        self._belief = SourceActivityForgettingBelief(
            session.model,
            half_life_seconds=session.half_life_seconds,
            transport_id=session.transport_id,
            prior_active_probability=prior_active_probability,
            log_odds_half_life_seconds=log_odds_half_life_seconds,
        )
