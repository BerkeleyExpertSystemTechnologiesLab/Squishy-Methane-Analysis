"""Release invariants for the preselected C2 demonstration, not field evidence."""
import json
from pathlib import Path

import numpy as np
import pytest

from plume_focus.c2_p0v import construct_rule
from plume_focus.c2_r2_mechanics import verify_route_record
from plume_focus.evidence_gated_execution import EvidenceGatedRunConfig
from plume_focus.evidence_gated_loop import EvidenceGatedScanner
from plume_focus.full_fov_execution import replay_event
from plume_focus.scan_order_demo import run_demo
from plume_focus.sensing_session import configuration_hash
from plume_focus.sensing_simulation import FakeScanner


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    root = Path(__file__).parents[1]
    out = tmp_path_factory.mktemp("c2-release") / "run"
    result = run_demo(root, out)
    prepared = json.loads((out / "PREFIX.json").read_text())
    return out, result, prepared


def test_preselected_world_and_historical_smoke_value(demo):
    _, result, prepared = demo
    assert result["variant"] == "b00_L"
    assert result["configuration_hash"] == "7f3e8de745a118a093a10a94052b6c756060f45102edd0712f9abd8db1712fc7"
    assert prepared["world"]["world_seed"] == 2136000
    # Historical B/F zero-noise values are regression anchors, not returned data.
    # The original four-route integration partition is preserved.
    assert result["results"]["B"]["mse"] == pytest.approx(0.011410370760521676, abs=1e-10)
    assert result["results"]["L_up"]["mse"] == pytest.approx(0.011268315169741286, abs=1e-10)
    assert result["gain"] == result["results"]["B"]["mse"] - result["results"]["L_up"]["mse"]


def test_fees_support_pose_and_direction(demo):
    out, result, prepared = demo
    plans = []
    for name, fee in (("B", 0.13), ("L_up", 0.14)):
        rec = json.loads((out / (name + "_route.json")).read_text())
        plan = verify_route_record(rec)
        assert plan.started_at_seconds == pytest.approx(prepared["t0"] + fee)
        assert plan.start_position_yx == tuple(prepared["saved"]["last_pose"])
        assert result["results"][name]["action_seconds"] <= 8.0
        assert not result["results"][name]["decision"]["fallback"]
        plans.append(plan)
    assert plans[0].requested_cells == plans[1].requested_cells
    assert plans[0].swept_cells == plans[1].swept_cells
    assert result["results"]["L_up"]["column_order"] == sorted(result["results"]["L_up"]["column_order"])
    assert not any(result["results"]["L_up"]["directions_down"])


def test_saved_readings_are_fresh_at_each_route_exposure(demo):
    out, _, prepared = demo
    cfg = EvidenceGatedRunConfig.from_dict(prepared["world"]["config"])
    pairs = []
    for name in ("B", "L_up"):
        rec = json.loads((out / (name + "_route.json")).read_text())
        trace = json.loads((out / (name + "_readings.json")).read_text())
        plan = verify_route_record(rec)
        actual = FakeScanner(cfg.build_source(), noise_std=0., seed=0,
                             gain=cfg.model.gain, offset=cfg.model.offset).execute(plan)
        assert len(trace) == len(actual.readings) == 25
        for saved, r in zip(trace, actual.readings):
            assert saved["position"] == list(r.position_yx)
            assert saved["acquired"] == r.acquired_at_seconds
            assert saved["available"] == r.available_at_seconds
            assert saved["value"] == r.value
        pairs.append([(r.position_yx, r.acquired_at_seconds) for r in actual.readings])
    assert pairs[0] != pairs[1]


def test_whole_common_window_includes_planning_and_idle(demo):
    out, result, prepared = demo
    with np.load(out / "evaluation.npz") as a:
        assert a["duration_weights"].sum() == pytest.approx(12.)
        assert a["times"].min() < prepared["t0"] + 0.13
        assert a["times"].max() > prepared["t0"] + max(r["action_seconds"] for r in result["results"].values())
        for name in ("B", "L_up"):
            assert np.average(a[name + "_mse"], weights=a["duration_weights"]) == pytest.approx(result["results"][name]["mse"])


def test_common_prefix_replays_through_original_public_calls(demo):
    _, _, prepared = demo
    cfg = EvidenceGatedRunConfig.from_dict(prepared["world"]["config"])
    scanner = EvidenceGatedScanner(cfg.session)
    for event in prepared["prefix_events"]:
        replay_event(scanner, event)
    np.testing.assert_array_equal(scanner._belief.mean, prepared["saved"]["mean"])
    np.testing.assert_array_equal(scanner._belief.covariance, prepared["saved"]["covariance"])
    assert scanner._last_pose == tuple(prepared["saved"]["last_pose"])


def test_budget_rejection_never_refunds_planning_delay(demo):
    _, _, prepared = demo
    cfg = EvidenceGatedRunConfig.from_dict(prepared["world"]["config"])
    kw = dict(prepared["kwargs"], started_at_seconds=prepared["t0"] + 7.9)
    plan, meta = construct_rule(prepared["mask"], cfg.session.instrument, kw, "F", 0.1)
    assert plan is None and meta["rejected"] and meta["fallback"]
    assert kw["started_at_seconds"] == prepared["t0"] + 7.9


def test_existing_outputs_are_not_overwritten(demo):
    out, _, _ = demo
    before = (out / "RESULTS.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_demo(Path(__file__).parents[1], out)
    assert (out / "RESULTS.json").read_bytes() == before
