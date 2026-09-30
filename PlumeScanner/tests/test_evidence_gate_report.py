"""Regression tests for change-boundary scoring and evidence-gate verdicts."""
import importlib.util
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix


@pytest.fixture(scope="module")
def driver():
    path = Path(__file__).resolve().parents[1] / "scripts/run-evidence-gated-focus.py"
    spec = importlib.util.spec_from_file_location("evidence_gate_reporting", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scan(start, end, times, clean, mode="FULL"):
    return {"plan": {"mode": mode, "started_at_seconds": start, "requested_cells": [0],
                     "samples": [{"acquired_at_seconds": t} for t in times]},
            "receipt": {"received_at_seconds": end}, "missing_full_cells": [], "clean_readings": clean}


@pytest.mark.parametrize("mode", ["FULL", "FOCUS"])
def test_reacquisition_scores_post_resume_readings_in_an_ongoing_scan(driver, monkeypatch, mode):
    monkeypatch.setattr(driver, "full_scan_seconds", lambda config: 130.)
    config = SimpleNamespace(cessation_seconds=None, active_windows=((0., 120.), (240., 360.)), duration_seconds=360.)
    q = SimpleNamespace(times=np.array([200., 240., 250., 260.]), peak=np.ones(4), core=csr_matrix(np.ones((4, 1))))
    record = {"scans": [scan(200., 260., [210., 230., 250.], [.9, .9, .9], mode)], "termination": "horizon_complete"}
    resume = driver.change_metrics(record, q, config)[1]
    assert resume["reacquisition_latency_seconds"] == 10.
    assert not resume["reacquisition_focus_plan"]  # The old plan was selected before resumption.
    assert not resume["sufficient_remaining_time"]  # Frozen full-restart criterion is preserved.
    if mode == "FULL":
        assert resume["straddling_post_change_reading_slots"] == 1 and resume["reacquisition_opportunity"]


def test_only_pre_resume_hits_do_not_count_as_reacquisition(driver, monkeypatch):
    monkeypatch.setattr(driver, "full_scan_seconds", lambda config: 30.)
    config = SimpleNamespace(cessation_seconds=None, active_windows=((0., 120.), (240., 360.)), duration_seconds=360.)
    q = SimpleNamespace(times=np.array([200., 240., 250.]), peak=np.ones(3), core=csr_matrix(np.ones((3, 1))))
    record = {"scans": [scan(200., 260., [210., 250.], [.9, .1])], "termination": "horizon_complete"}
    assert driver.change_metrics(record, q, config)[1]["reacquisition_latency_seconds"] is None


def preservation_row(profile, case, change=0., coverage=.8):
    return {"status": "complete", "stage": "heldout", "profile": profile, "case": case,
            "J_over_T": {"control": 1., "candidate": 1. + change},
            "focus_core_geometric_coverage": {"control": .8, "candidate": coverage},
            "trailing_full_scan_core_coverage": {"control": .8, "candidate": .8},
            "spent_seconds": {"control": 1., "candidate": 1.}}


@pytest.mark.parametrize("change,coverage,status", [(0., .8, "supported"), (.06, .8, "unsupported"),
                                                   (0., .74, "unsupported"), (0., None, "unavailable")])
def test_preservation_verdict_is_not_overwritten_and_missing_coverage_is_unavailable(driver, change, coverage, status):
    document = driver.study()
    rows = [preservation_row(profile, case, change, coverage)
            for profile in document["reference_configs"] for case in ("continuous", "wind_error")]
    before = deepcopy(rows)
    criterion = driver.criteria(document, rows, [])["continuous_operation_preserved"]
    assert criterion["status"] == status
    assert all(detail["status"] == status for detail in criterion["detail"].values())
    assert rows == before


def test_missing_preservation_profile_does_not_become_a_failure_or_support(driver):
    document = driver.study()
    rows = [preservation_row("coarse", case) for case in ("continuous", "wind_error")]
    checks = driver.criteria(document, rows, [])
    assert checks["continuous_operation_preserved"]["status"] == "unavailable"
    empty = driver.criteria(document, [], [])
    assert empty["continuous_operation_preserved"]["status"] == "unavailable"
    assert empty["unsupported_focus_forces_full"]["status"] == "unavailable"


@pytest.mark.parametrize("destination", ["same", "child", "existing"])
def test_additive_review_refuses_frozen_or_existing_destinations(destination):
    repo = Path(__file__).resolve().parents[1]
    path = repo / "scripts/review-evidence-gated-focus.py"
    spec = importlib.util.spec_from_file_location("additive_gate_review", path)
    review = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(review)
    root = repo / "outputs/frozen-review-guard"
    output = {"same": root, "child": root / "reports", "existing": repo / "tests"}[destination]
    with pytest.raises(ValueError, match="separate|already exists"):
        review.review(root, output)
