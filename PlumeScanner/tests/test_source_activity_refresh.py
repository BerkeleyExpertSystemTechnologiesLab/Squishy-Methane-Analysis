"""Check the composition at the typed plan, reading, and receipt boundaries."""

from dataclasses import replace

import numpy as np
import pytest

from plume_focus.controller import SensingController
from plume_focus.evidence_gated_loop import EvidenceGatedScanner
from plume_focus.evidence_refresh import EvidenceRefreshRequest
from plume_focus.sensing_contracts import CausalWind, ExecutionReceipt
from plume_focus.sensing_messages_v2 import ScanRequestV1
from plume_focus.source_activity import SourceActivityBelief
from plume_focus.source_activity_refresh import SourceActivityRefreshScanner
from test_evidence_gate import gate_session
from test_evidence_refresh import finish_startup, reading


BUDGET = ScanRequestV1(100.0)


def source_session():
    session = gate_session()
    return replace(
        session,
        source_position=(0.125, 0.125),
        model=replace(session.model, source_rate=0.3, source_sigma=0.15),
        half_life_seconds=6.0,
    )


def open_request(scanner, deadline_seconds=100.0):
    request = scanner.source_activity_request(
        "check-source", fresh_since_seconds=0.0, deadline_seconds=deadline_seconds,
    )
    result = scanner.plan_evidence_refresh(
        request, 0.0, scanner.session.home_yx, BUDGET, preparation_seconds=0.0,
    )
    assert result.outcome == "scheduled"
    assert result.plan.mode == "FULL"
    return request, result.plan


def test_source_cell_reading_updates_belief_once_and_full_closes_at_receipt():
    session = source_session()
    scanner = SourceActivityRefreshScanner(session, delivery="stream")
    request, plan = open_request(scanner)
    assert request.cells == (0,)
    item = reading(plan, 0, value=1.0)
    available_at = item.available_at_seconds
    assert scanner.source_activity_state().active_model_weight == 0.5

    events = scanner.ingest_readings([item], available_at_seconds=available_at)
    assert len(events) == 1 and events[0].reading_id == item.reading_id
    reference = SourceActivityBelief(
        session.model, half_life_seconds=session.half_life_seconds,
        transport_id=session.transport_id,
    )
    reference.assimilate(item, CausalWind(), session.source_position)
    state = scanner.source_activity_state()
    assert state.log_odds == pytest.approx(reference.log_odds)
    assert state.active_model_weight != 0.5
    assert state.belief_time_seconds == item.acquired_at_seconds
    assert state.controller_time_seconds == available_at

    assert not scanner.ingest_readings([item], available_at_seconds=available_at)
    assert scanner.source_activity_state() == state
    status = scanner.evidence_status(request, now_seconds=available_at)
    assert status.state == "satisfied" and status.active_scan_id == plan.scan_id
    assert status.evidence[0].reading == item
    assert status.evidence[0].accepted_at_seconds == available_at
    reused = scanner.plan_evidence_refresh(
        request, available_at, session.home_yx, BUDGET, preparation_seconds=0.0,
    )
    assert reused.outcome == "satisfied" and reused.plan is None
    assert scanner.source_activity_state() == state
    with pytest.raises(RuntimeError, match="Finish the current scan"):
        scanner.plan_next(available_at, session.home_yx, BUDGET)

    for index in range(1, len(plan.samples)):
        item = reading(plan, index)
        scanner.ingest_readings([item], available_at_seconds=item.available_at_seconds)
    receipt = ExecutionReceipt(
        plan.scan_id, plan.completed_at_seconds, plan.completed_at_seconds, plan.final_position_yx,
    )
    scanner.finish_scan(receipt)
    closed = scanner.evidence_status(request, now_seconds=receipt.received_at_seconds)
    assert closed.active_scan_id is None and closed.outstanding_full_cells == 0
    assert closed.evidence == status.evidence


def test_out_of_order_readings_preserve_original_acceptance_and_replay_b1():
    session = source_session()
    ordered = SourceActivityRefreshScanner(session, delivery="stream")
    reordered = SourceActivityRefreshScanner(session, delivery="stream")
    _, plan = open_request(ordered)
    _, alternate = open_request(reordered)
    assert alternate == plan
    early, later = reading(plan, 0, value=0.9), reading(plan, 1, value=0.1)
    later_cell = reordered._physical_cell(later.position_yx)
    request = EvidenceRefreshRequest(
        "both-cells", session.instrument.digest, (0, later_cell), 0.0, 100.0,
    )

    ordered.ingest_readings([early, later], available_at_seconds=later.available_at_seconds)
    reordered.ingest_readings([later], available_at_seconds=later.available_at_seconds)
    replay_at = later.available_at_seconds + 0.1
    events = reordered.ingest_readings([early], available_at_seconds=replay_at)
    assert tuple(event.reading_id for event in events) == (early.reading_id, later.reading_id)
    assert reordered.source_activity_state().log_odds == pytest.approx(ordered.source_activity_state().log_odds)
    for actual, expected in zip(reordered.predictive_moments(replay_at, (2, 2)),
                                ordered.predictive_moments(replay_at, (2, 2))):
        np.testing.assert_allclose(actual, expected)
    status = reordered.evidence_status(request, now_seconds=replay_at)
    assert status.state == "satisfied"
    assert tuple(entry.accepted_at_seconds for entry in status.evidence) == (
        replay_at, later.available_at_seconds,
    )


def test_late_measurement_updates_current_belief_but_misses_the_request():
    scanner = SourceActivityRefreshScanner(source_session(), delivery="stream")
    request, plan = open_request(scanner, deadline_seconds=1.0)
    item = reading(plan, 0)
    assert item.available_at_seconds == pytest.approx(request.deadline_seconds)
    accepted_at = 1.1
    scanner.ingest_readings([item], available_at_seconds=accepted_at)
    assert scanner.source_activity_state().active_model_weight != 0.5
    status = scanner.evidence_status(request, now_seconds=accepted_at)
    assert status.state == "deadline_missed"
    assert status.evidence == () and status.missing_cells == (0,)


def test_native_gate_denies_focus_even_with_high_active_model_weight():
    session = source_session()
    session = replace(session, model=replace(session.model, source_rate=0.005))
    scanner = SourceActivityRefreshScanner(session, delivery="stream", prior_active_probability=0.999)
    _, plan = open_request(scanner)
    for index in range(len(plan.samples)):
        item = reading(plan, index, value=0.0)
        scanner.ingest_readings([item], available_at_seconds=item.available_at_seconds)
    receipt = ExecutionReceipt(
        plan.scan_id, plan.completed_at_seconds, plan.completed_at_seconds, plan.final_position_yx,
    )
    scanner.finish_scan(receipt)
    assert scanner.source_activity_state().active_model_weight > 0.9
    assert not scanner.gate_focus_allowed
    assert scanner.plan_next(receipt.received_at_seconds, receipt.final_position_yx, BUDGET).mode == "FULL"


def test_requested_source_evidence_advances_full_when_normal_loop_would_focus():
    session = source_session()
    normal = SourceActivityRefreshScanner(session, delivery="stream")
    requested = SourceActivityRefreshScanner(session, delivery="stream")
    finish_startup(normal)
    receipt = finish_startup(requested)
    now, pose = receipt.received_at_seconds, receipt.final_position_yx
    assert normal.plan_next(now, pose, BUDGET).mode == "FOCUS"
    request = requested.source_activity_request(
        "after-startup", fresh_since_seconds=now, deadline_seconds=now + 100.0,
    )
    decision = requested.plan_evidence_refresh(
        request, now, pose, BUDGET, preparation_seconds=0.2,
    )
    assert decision.outcome == "scheduled"
    assert decision.plan.mode == "FULL"
    assert decision.plan.reason == "evidence_refresh_requested"
    assert decision.status.missing_cells == request.cells


def test_combined_scanner_is_explicit_and_requires_declared_source():
    session = source_session()
    assert type(SensingController.from_session(session)) is EvidenceGatedScanner
    with pytest.raises(ValueError, match="declared source"):
        SourceActivityRefreshScanner(replace(session, source_position=None), delivery="stream")
    edge = SourceActivityRefreshScanner(replace(session, source_position=(1.0, 1.0)), delivery="stream")
    request = edge.source_activity_request("edge", fresh_since_seconds=0.0, deadline_seconds=10.0)
    assert request.cells == (15,)
