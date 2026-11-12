"""Forgetting-belief wiring inside the explicit known-source evidence request loop."""

import math

import pytest

from plume_focus.sensing_contracts import CausalWind
from plume_focus.source_activity_forgetting import SourceActivityForgettingBelief
from plume_focus.source_activity_refresh_forgetting import (
    SourceActivityForgettingRefreshScanner,
)
from test_source_activity_refresh import open_request, reading, source_session


def test_forgetting_refresh_scanner_uses_forgetting_belief():
    session = source_session()
    scanner = SourceActivityForgettingRefreshScanner(
        session, delivery="stream", log_odds_half_life_seconds=15.0,
    )
    assert isinstance(scanner._belief, SourceActivityForgettingBelief)
    assert scanner._belief.log_odds_half_life_seconds == 15.0
    assert scanner.source_activity_state().active_model_weight == 0.5
    plain = SourceActivityForgettingRefreshScanner(source_session(), delivery="stream")
    assert isinstance(plain._belief, SourceActivityForgettingBelief)
    assert plain._belief.log_odds_half_life_seconds is None


def test_forgetting_refresh_request_loop_closes_like_b1():
    session = source_session()
    scanner = SourceActivityForgettingRefreshScanner(
        session, delivery="stream", log_odds_half_life_seconds=15.0,
    )
    request, plan = open_request(scanner)
    assert request.cells == (0,)
    item = reading(plan, 0, value=1.0)
    available_at = item.available_at_seconds
    events = scanner.ingest_readings([item], available_at_seconds=available_at)
    assert len(events) == 1 and events[0].reading_id == item.reading_id
    state = scanner.source_activity_state()
    assert state.active_model_weight != 0.5
    assert state.belief_time_seconds == item.acquired_at_seconds
    assert state.controller_time_seconds == available_at
    status = scanner.evidence_status(request, now_seconds=available_at)
    assert status.state == "satisfied"


def test_forgetting_refresh_decays_log_odds_between_readings():
    session = source_session()
    tau = 15.0
    scanner = SourceActivityForgettingRefreshScanner(
        session, delivery="stream", log_odds_half_life_seconds=tau,
    )
    _, plan = open_request(scanner)
    item = reading(plan, 0, value=1.0)
    scanner.ingest_readings([item], available_at_seconds=item.available_at_seconds)
    after = scanner._belief.log_odds
    assert after != 0.0
    later = item.acquired_at_seconds + 5 * tau
    scanner._belief.advance(later, CausalWind(), session.source_position)
    expected = after * math.exp(-math.log(2.0) / tau * (later - item.acquired_at_seconds))
    assert scanner._belief.log_odds == pytest.approx(expected)
    assert abs(scanner._belief.log_odds) < abs(after)
