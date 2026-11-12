"""Exercise local C3 action selection, B1 ingestion, and native recovery."""

import pytest

from plume_focus.full_fov_execution import decode_plan
from plume_focus.local_evidence_refresh import LocalSourceActivityRefreshScanner
from plume_focus.sensing_contracts import ExecutionReceipt
from plume_focus.sensing_messages_v2 import ScanRequestV1
from test_evidence_refresh import finish_startup, reading
from test_source_activity_refresh import source_session


def ready_scanner(delivery="stream"):
    scanner = LocalSourceActivityRefreshScanner(source_session(), delivery=delivery)
    receipt = finish_startup(scanner)
    assert scanner.gate_focus_allowed
    return scanner, receipt


def request_at_receipt(scanner, receipt, *, terminal, budget=100.0, deadline=100.0):
    now = receipt.received_at_seconds
    request = scanner.source_activity_request(
        "source-probe", fresh_since_seconds=now, deadline_seconds=now + deadline,
    )
    decision = scanner.plan_local_evidence_refresh(
        request, now, receipt.final_position_yx, ScanRequestV1(budget),
        terminal=terminal, preparation_seconds=0.2,
    )
    return request, decision


def test_single_cell_streams_to_b1_and_keeps_c3_reentry_rule():
    scanner, prefix = ready_scanner()
    request, decision = request_at_receipt(scanner, prefix, terminal="receipt")
    assert decision.outcome == "scheduled"
    plan = decision.plan
    assert plan.mode == "REFRESH"
    assert plan.requested_cells == plan.swept_cells == request.cells == (0,)
    assert len(plan.samples) == 1
    assert decode_plan(plan.to_dict()) == plan
    target = reading(plan, 0, value=0.7)
    accepted_at = target.available_at_seconds
    initial = scanner.source_activity_state()
    events = scanner.ingest_readings([target], available_at_seconds=accepted_at)
    assert len(events) == 1 and events[0].reading_id == target.reading_id
    updated = scanner.source_activity_state()
    assert updated.log_odds != initial.log_odds
    assert not scanner.ingest_readings([target], available_at_seconds=accepted_at)
    assert scanner.source_activity_state() == updated
    status = scanner.evidence_status(request, now_seconds=accepted_at)
    assert status.state == "satisfied" and status.active_scan_id == plan.scan_id
    assert status.evidence[0].accepted_at_seconds == pytest.approx(decision.estimated_last_acceptance_seconds)
    with pytest.raises(RuntimeError, match="Finish the current scan"):
        scanner.plan_next(accepted_at, plan.final_position_yx, ScanRequestV1(100))

    receipt = ExecutionReceipt(
        plan.scan_id, plan.completed_at_seconds, plan.completed_at_seconds, plan.final_position_yx,
    )
    scanner.finish_scan(receipt)
    assert not scanner.gate_focus_allowed
    assert scanner.completed_full_scans == 1 and not scanner.full_scan_missing
    recovery = scanner.plan_next(receipt.received_at_seconds, receipt.final_position_yx, ScanRequestV1(100))
    assert recovery.mode == "FULL"


def test_local_route_reserves_home_and_full_before_opening():
    probe, prefix = ready_scanner()
    _, local = request_at_receipt(probe, prefix, terminal="receipt")
    assert local.plan.mode == "REFRESH"
    scanner, prefix = ready_scanner()
    before = scanner.source_activity_state(), scanner.gate_state(), scanner.full_scan_missing
    _, declined = request_at_receipt(
        scanner, prefix, terminal="home_full", budget=local.required_duration_seconds + 0.1,
    )
    assert declined.plan is None and declined.local_reason == "terminal_budget"
    assert scanner.active_plan is None
    assert (scanner.source_activity_state(), scanner.gate_state(), scanner.full_scan_missing) == before


def test_startup_full_is_served_before_any_local_request():
    scanner = LocalSourceActivityRefreshScanner(source_session(), delivery="stream")
    request = scanner.source_activity_request("startup", fresh_since_seconds=0.0, deadline_seconds=100.0)
    decision = scanner.plan_local_evidence_refresh(
        request, 0.0, scanner.session.home_yx, ScanRequestV1(100),
        terminal="receipt", preparation_seconds=0.0,
    )
    assert decision.local_reason == "full_required" and decision.plan.mode == "FULL"


def test_both_on_time_prefers_lower_cost_full_at_same_terminal():
    scanner, prefix = ready_scanner()
    _, decision = request_at_receipt(scanner, prefix, terminal="home_full")
    assert decision.local_reason == "full_preferred" and decision.plan.mode == "FULL"


def test_batch_exposure_before_deadline_with_late_acceptance_still_misses():
    scanner, prefix = ready_scanner("batch")
    request, decision = request_at_receipt(scanner, prefix, terminal="receipt", deadline=8.0)
    plan = decision.plan
    assert plan.mode == "REFRESH"
    assert decision.estimated_last_acceptance_seconds == pytest.approx(plan.completed_at_seconds)
    item = reading(plan, 0)
    accepted_at = request.deadline_seconds + 0.1
    scanner.ingest_readings([item], available_at_seconds=accepted_at)
    status = scanner.evidence_status(request, now_seconds=accepted_at)
    assert item.available_at_seconds < request.deadline_seconds
    assert status.state == "deadline_missed" and status.missing_cells == request.cells
    assert status.evidence == ()
    scanner.finish_scan(
        ExecutionReceipt(plan.scan_id, plan.completed_at_seconds, accepted_at, plan.final_position_yx),
    )
    assert not scanner.gate_focus_allowed
