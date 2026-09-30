import ast
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import plume_focus.experiment_runner as experiment_runner
from plume_focus.experiment_runner import (
    ExperimentContext,
    ExperimentRequest,
    ExperimentStep,
    run_experiment,
)
from plume_focus.video_rendering import VideoRenderer


class RecordingObserver:
    def __init__(self, *, fail_on_observe: bool = False) -> None:
        self.events: list[str] = []
        self.context: ExperimentContext | None = None
        self.steps: list[ExperimentStep] = []
        self.fail_on_observe = fail_on_observe

    def start(self, context: ExperimentContext) -> None:
        self.context = context
        self.events.append("start")

    def observe(self, step: ExperimentStep) -> None:
        self.events.append("observe")
        self.steps.append(step)
        if self.fail_on_observe:
            raise RuntimeError("observer failure")

    def close(self) -> None:
        self.events.append("close")


def _fast_methane_request(*, max_scans: int = 2) -> ExperimentRequest:
    return ExperimentRequest(
        scenario="methane",
        max_scans=max_scans,
        vertical_pass_seconds=0.005,
        horizontal_step_seconds=0.0,
        column_mass_fraction=1.0,
        row_mass_fraction=1.0,
        max_sweep_columns=320,
    )


def test_headless_experiment_never_constructs_a_video_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def reject_video_writer(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("headless execution attempted to render video")

    def reject_file_write(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("headless execution attempted to write a file")

    monkeypatch.setattr(cv2, "VideoWriter", reject_video_writer)
    monkeypatch.setattr(Path, "write_text", reject_file_write)

    result = run_experiment(_fast_methane_request())

    assert result.report.output_video == ""
    assert result.report.scans_completed == 2
    assert list(tmp_path.iterdir()) == []


def test_observer_receives_ordered_steps_and_always_closes() -> None:
    observer = RecordingObserver()

    result = run_experiment(
        _fast_methane_request(max_scans=3),
        observer=observer,
    )

    assert observer.context is not None
    assert observer.context.shape == (220, 320)
    assert observer.events == ["start", "observe", "observe", "observe", "close"]
    assert len(observer.steps) == result.report.scans_completed == 3
    assert all(
        step.scan.completed_at_seconds >= step.scan.started_at_seconds
        for step in observer.steps
    )


def test_operational_trace_is_observer_independent_and_checks_beliefs(
    tmp_path: Path,
) -> None:
    request = _fast_methane_request(max_scans=3)

    headless = run_experiment(request).report
    rendered = run_experiment(
        request,
        observer=VideoRenderer(tmp_path / "trace-proof.mp4"),
    ).report

    assert headless.report_schema_version == 3
    assert headless.operational_trace == rendered.operational_trace
    assert headless.belief_integrity == rendered.belief_integrity
    assert headless.sample_variation == rendered.sample_variation
    assert len(headless.operational_trace) == headless.scans_completed == 3
    assert headless.belief_integrity.status == "evaluated"
    assert headless.belief_integrity.distributions_checked == 6
    assert headless.belief_integrity.all_finite
    assert headless.belief_integrity.all_nonnegative
    assert headless.belief_integrity.all_normalized
    assert all(
        item.evidence_timestamp_count > 0
        and item.evidence_timestamp_min_seconds is not None
        and item.evidence_timestamp_max_seconds is not None
        and item.evidence_timestamp_min_seconds
        >= item.started_at_seconds
        and item.evidence_timestamp_max_seconds
        <= item.completed_at_seconds
        and len(item.requested_mask_sha256) == 64
        and len(item.planned_next_mechanical_sweep_mask_sha256) == 64
        and len(item.evidence_schedule_sha256) == 64
        and len(item.sampled_value_sha256) == 64
        and item.sampled_value_count > 0
        and item.sampled_value_min is not None
        and item.sampled_value_max is not None
        and item.sampled_value_min <= item.sampled_value_max
        and item.beliefs_normalized
        for item in headless.operational_trace
    )
    assert headless.sample_variation.status == "evaluated"
    assert headless.sample_variation.variation_observed
    assert headless.sample_variation.comparisons > 0
    assert headless.sample_variation.changed_values > 0
    assert headless.sample_variation.changed_fraction is not None
    assert headless.sample_variation.changed_fraction > 0.0
    assert not headless.sample_variation.full_frame_evidence_mode
    serialized = json.loads(headless.to_json())
    assert serialized["operational_trace"]["source_frame_indices"] == (
        "not_available_at_causal_runner_boundary"
    )
    assert len(serialized["operational_trace"]["schedule"]) == 3
    assert serialized["belief_integrity"]["status"] == "evaluated"
    assert serialized["sample_variation"]["comparisons"] > 0
    assert serialized["sample_variation"]["full_frame_evidence_mode"] is False


def test_observer_closes_when_step_consumption_raises() -> None:
    observer = RecordingObserver(fail_on_observe=True)

    with pytest.raises(RuntimeError, match="observer failure"):
        run_experiment(_fast_methane_request(), observer=observer)

    assert observer.events == ["start", "observe", "close"]


def test_observer_closes_after_finite_source_exhaustion(tmp_path: Path) -> None:
    input_video = tmp_path / "short-input.mp4"
    writer = cv2.VideoWriter(
        str(input_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (20, 16),
    )
    try:
        for index in range(6):
            writer.write(
                np.full(
                    (16, 20, 3),
                    20 + index,
                    dtype=np.uint8,
                )
            )
    finally:
        writer.release()

    observer = RecordingObserver()
    result = run_experiment(
        ExperimentRequest(
            input_video=input_video,
            scenario="object",
            max_scans=3,
            scan_rate_hz=100.0,
        ),
        observer=observer,
    )

    assert result.report.termination_reason == "source_exhausted_before_scan"
    assert observer.events == ["start", "close"]


def test_experiment_runner_has_no_video_adapter_dependencies() -> None:
    runner_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "plume_focus"
        / "experiment_runner.py"
    )
    tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    forbidden = {
        "plume_focus.video",
        "plume_focus.video_rendering",
        "plume_focus.video_workflow",
    }
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert imported_modules.isdisjoint(forbidden)


def test_rendering_relative_density_does_not_change_scan_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_observer = RecordingObserver()
    changed_observer = RecordingObserver()
    request = _fast_methane_request(max_scans=2)

    baseline = run_experiment(
        request,
        observer=baseline_observer,
    ).report
    monkeypatch.setattr(
        experiment_runner,
        "ATTENTION_OUTER_RELATIVE_DENSITY",
        0.99,
    )
    changed = run_experiment(
        request,
        observer=changed_observer,
    ).report

    assert len(baseline_observer.steps) == len(changed_observer.steps)
    assert all(
        np.array_equal(left.next_scan_mask, right.next_scan_mask)
        for left, right in zip(
            baseline_observer.steps,
            changed_observer.steps,
        )
    )
    assert (
        baseline.mean_actual_sweep_interval_mass_recall
        == changed.mean_actual_sweep_interval_mass_recall
    )
    assert baseline.scan_duration_seconds == changed.scan_duration_seconds
