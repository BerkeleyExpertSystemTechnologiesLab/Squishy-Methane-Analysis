"""Exercise the request API through native plan, reading, and receipt calls."""

from dataclasses import replace

import pytest

from plume_focus.evidence_gated_loop import EvidenceGatedScanner
from plume_focus.evidence_refresh import EvidenceRefreshRequest, EvidenceRefreshScanner
from plume_focus.sensing_contracts import ExecutionReceipt, SensorReading
from plume_focus.sensing_messages_v2 import ScanRequestV1
from test_evidence_gate import gate_session


def reading(plan, index, *, value=1.0):
    sample = plan.samples[index]
    return SensorReading(
        plan.scan_id, f"{plan.scan_id}:{index}", sample.position_yx,
        sample.acquired_at_seconds,
        sample.acquired_at_seconds + sample.integration_seconds / 2,
        value, sample.footprint, sample.integration_seconds,
    )


def finish_startup(scanner):
    plan = scanner.plan_next(0.0, scanner.session.home_yx, ScanRequestV1(100.0))
    for i in range(len(plan.samples)):
        item = reading(plan, i)
        scanner.ingest_readings([item], available_at_seconds=item.available_at_seconds)
    receipt = ExecutionReceipt(plan.scan_id, plan.completed_at_seconds,
                               plan.completed_at_seconds, plan.final_position_yx)
    scanner.finish_scan(receipt)
    return receipt


def request(scanner, fresh_since=0.0, deadline=100.0, cells=(0,)):
    return EvidenceRefreshRequest("anchor-check", scanner.session.instrument.digest,
                                  cells, fresh_since, deadline)


def test_explicit_request_opens_full_when_normal_loop_would_focus():
    session = gate_session()
    control = EvidenceGatedScanner(session)
    scanner = EvidenceRefreshScanner(session, delivery="stream")
    finish_startup(control)
    receipt = finish_startup(scanner)
    now = receipt.received_at_seconds
    pose = receipt.final_position_yx
    budget = ScanRequestV1(100.0)
    assert control.plan_next(now, pose, budget).mode == "FOCUS"

    refresh = request(scanner, fresh_since=now, deadline=now + 100)
    decision = scanner.plan_evidence_refresh(refresh, now, pose, budget, preparation_seconds=0.2)
    assert decision.outcome == "scheduled"
    plan = decision.plan
    assert plan.mode == "FULL"
    assert plan.reason == "evidence_refresh_requested"
    assert len(plan.swept_cells) == len(plan.samples)
    assert set(plan.swept_cells) == set(range(scanner.capabilities.lattice_shape[0] *
                                               scanner.capabilities.lattice_shape[1]))
    assert plan.started_at_seconds == pytest.approx(now + 0.2)
    assert decision.required_duration_seconds == pytest.approx(plan.completed_at_seconds - now)
    busy = scanner.plan_evidence_refresh(refresh, plan.started_at_seconds, pose, budget,
                                         preparation_seconds=0.0)
    assert busy.outcome == "scan_in_progress" and busy.plan is None
    assert scanner.active_plan.scan_id == plan.scan_id

    item = reading(plan, 0)
    scanner.ingest_readings([item], available_at_seconds=item.available_at_seconds)
    status = scanner.evidence_status(refresh, now_seconds=item.available_at_seconds)
    assert status.state == "satisfied"
    assert status.evidence[0].reading == item
    assert status.active_scan_id == plan.scan_id
    with pytest.raises(RuntimeError, match="Finish the current scan"):
        scanner.plan_next(item.available_at_seconds, pose, budget)
    for i in range(1, len(plan.samples)):
        delivered = reading(plan, i)
        scanner.ingest_readings([delivered], available_at_seconds=delivered.available_at_seconds)
    full_receipt = ExecutionReceipt(plan.scan_id, plan.completed_at_seconds,
                                    plan.completed_at_seconds, plan.final_position_yx)
    scanner.finish_scan(full_receipt)
    done = scanner.evidence_status(refresh, now_seconds=full_receipt.received_at_seconds)
    assert done.active_scan_id is None and done.outstanding_full_cells == 0
    assert done.evidence == status.evidence


@pytest.mark.parametrize("budget,deadline,outcome", [
    (0.01, 100.0, "full_scan_exceeds_budget"),
    (100.0, 0.01, "window_unreachable"),
])
def test_refused_request_leaves_ready_controller_untouched(budget, deadline, outcome):
    scanner = EvidenceRefreshScanner(gate_session(), delivery="stream")
    before = (scanner.gate_state(), scanner.full_scan_missing)
    decision = scanner.plan_evidence_refresh(
        request(scanner, deadline=deadline), 0.0, scanner.session.home_yx,
        ScanRequestV1(budget), preparation_seconds=0.0,
    )
    assert decision.outcome == outcome and decision.plan is None
    assert scanner.active_plan is None
    assert before == (scanner.gate_state(), scanner.full_scan_missing)


def test_late_controller_acceptance_does_not_satisfy_the_deadline():
    scanner = EvidenceRefreshScanner(gate_session(), delivery="stream")
    refresh = request(scanner, deadline=1.0)
    decision = scanner.plan_evidence_refresh(refresh, 0.0, scanner.session.home_yx,
                                             ScanRequestV1(100.0), preparation_seconds=0.0)
    assert decision.outcome == "scheduled"
    item = reading(decision.plan, 0)
    assert item.available_at_seconds == pytest.approx(refresh.deadline_seconds)
    scanner.ingest_readings([item], available_at_seconds=1.1)
    status = scanner.evidence_status(refresh, now_seconds=1.1)
    assert status.state == "deadline_missed" and status.missing_cells == (0,)


def test_already_delivered_evidence_reused_and_freshness_cutoff_respected():
    scanner = EvidenceRefreshScanner(gate_session(), delivery="stream")
    receipt = finish_startup(scanner)
    now = receipt.received_at_seconds
    refresh = request(scanner)
    decision = scanner.plan_evidence_refresh(refresh, now, receipt.final_position_yx,
                                             ScanRequestV1(100.0), preparation_seconds=0.0)
    assert decision.outcome == "satisfied" and decision.plan is None
    assert scanner.active_plan is None
    later = replace(refresh, fresh_since_seconds=now)
    assert scanner.evidence_status(later, now_seconds=now).missing_cells == (0,)


def test_batch_requires_budget_through_delivery_and_keeps_partial_full_obligation():
    scanner = EvidenceRefreshScanner(gate_session(), delivery="batch")
    decision = scanner.plan_evidence_refresh(request(scanner, deadline=1.0), 0.0,
                                             scanner.session.home_yx, ScanRequestV1(100.0),
                                             preparation_seconds=0.0)
    assert decision.outcome == "window_unreachable"
    assert scanner.active_plan is None

    plan = scanner.plan_next(0.0, scanner.session.home_yx, ScanRequestV1(100.0))
    item = reading(plan, 0)
    scanner.ingest_readings([item], available_at_seconds=plan.completed_at_seconds)
    receipt = ExecutionReceipt(plan.scan_id, plan.completed_at_seconds,
                               plan.completed_at_seconds, plan.final_position_yx)
    scanner.finish_scan(receipt)
    before = scanner.full_scan_missing
    decision = scanner.plan_evidence_refresh(
        request(scanner, fresh_since=receipt.received_at_seconds), receipt.received_at_seconds,
        receipt.final_position_yx, ScanRequestV1(100.0), preparation_seconds=0.0,
    )
    assert decision.outcome == "current_full_lacks_cells"
    assert scanner.full_scan_missing == before and scanner.active_plan is None
