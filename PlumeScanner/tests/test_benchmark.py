from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from plume_focus.benchmark_cases import make_benchmark_suite
from plume_focus.benchmark_cli import main as benchmark_main
from plume_focus.benchmark_contracts import BenchmarkSuite
from plume_focus.benchmark_runner import POLICY_IDS, run_benchmark
from plume_focus.benchmark_statistics import (
    paired_bootstrap_confidence_interval,
)
from plume_focus.scan_policies import ScanBudget


def _micro_suite() -> BenchmarkSuite:
    development = make_benchmark_suite(
        "development",
        scans_per_run=1,
    )
    return replace(
        development,
        suite_id="methane-fair-benchmark-test-v1",
        episodes=development.episodes[:1],
        budgets=(ScanBudget("micro", columns=5, rows=8),),
    )


def test_manifests_are_deterministic_disjoint_and_correctly_sized() -> None:
    development = make_benchmark_suite("development")
    repeated = make_benchmark_suite("development")
    held_out = make_benchmark_suite("held-out")

    assert development.manifest_hash == repeated.manifest_hash
    assert (
        development.manifest_hash
        == "914a31f920a379eae8e4ac11bdfa12cab57190aa71547826051c563243cbf758"
    )
    assert (
        held_out.manifest_hash
        == "da4e87e3665973ca270905889b14919081b359aa0e47bb077c65ec924b896ecd"
    )
    assert len([case for case in development.episodes if not case.stress]) == 6
    assert len([case for case in development.episodes if case.stress]) == 2
    assert len([case for case in held_out.episodes if not case.stress]) == 24
    assert len([case for case in held_out.episodes if case.stress]) == 6
    assert {
        case.seed for case in development.episodes
    }.isdisjoint({case.seed for case in held_out.episodes})


def test_paired_bootstrap_is_deterministic_and_paired() -> None:
    deltas = np.asarray([0.1, 0.1, 0.1, 0.1], dtype=float)

    first = paired_bootstrap_confidence_interval(deltas)
    second = paired_bootstrap_confidence_interval(deltas)

    assert first == second == (0.1, 0.1)


def test_micro_benchmark_is_reproducible_and_exactly_cost_matched() -> None:
    suite = _micro_suite()

    first = run_benchmark(suite)
    second = run_benchmark(suite)

    assert first.to_json() == second.to_json()
    assert first.integrity.causal_policy_contract
    assert first.integrity.complete_result_matrix
    assert first.integrity.exact_cost_match
    assert first.integrity.timeline_integrity
    assert first.claim_status != "invalid"
    assert first.frozen_model_commit == "104f55c"
    assert first.policy_version == "2"
    assert len(first.results) == len(POLICY_IDS)
    expected_area = suite.budgets[0].area_fraction((220, 320))
    assert all(
        result.initial_area_fraction == expected_area
        and result.mean_swept_area_fraction == expected_area
        for result in first.results
    )
    oracle = [
        result
        for result in first.results
        if result.policy_id == "oracle-wind-adaptive"
    ]
    assert oracle[0].oracle_inputs == ("exact synthetic wind vector",)


def test_benchmark_cli_smoke_writes_schema_v1_report(tmp_path: Path) -> None:
    output = tmp_path / "benchmark-smoke.json"

    exit_code = benchmark_main(
        [
            "--suite",
            "development",
            "--smoke",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["report_schema_version"] == 1
    assert payload["integrity"]["exact_cost_match"]
    assert payload["claim_status"] in {"supported", "not_demonstrated"}
    assert payload["model_profile"]["profile_id"] == "synthetic_demo_v1"
    assert (
        payload["model_profile"]["calibration_status"]
        == "uncalibrated"
    )
    assert payload["model_profile"]["scan_policy"][
        "static_corridor"
    ]["downwind_center_px"] == 34.0
