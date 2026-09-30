from pathlib import Path

import cv2
import numpy as np

from plume_focus.video_workflow import run_video_proof


def test_source_free_methane_run_uses_full_scene_scan_then_focuses() -> None:
    output_video = Path("outputs") / "_test-source-free-methane.mp4"
    report_output = Path("outputs") / "_test-source-free-methane.json"
    try:
        report = run_video_proof(
            output_video=output_video,
            report_output=report_output,
            scenario="methane",
            max_frames=8,
            source_free=True,
            render_scale=1,
            vertical_pass_seconds=0.1,
            horizontal_step_seconds=0.0,
            min_sweep_columns=10,
            min_sweep_rows=20,
            max_sweep_columns=40,
        )
    finally:
        output_video.unlink(missing_ok=True)
        report_output.unlink(missing_ok=True)

    assert report.scan_start_mode == "source_free_full_scene_discovery"
    assert report.plume_mapping_mode == "source_free_scan_evidence"
    assert report.report_schema_version == 3
    assert report.evidence_mode == "scan"
    assert report.frames_processed == 8
    assert report.scans_completed == 8
    assert report.timeline_integrity_verified
    assert report.initial_roi_area_fraction == 1.0
    assert report.initial_mask_area_fraction == 1.0
    assert report.final_mask_area_fraction < 1.0
    assert report.mean_dx > 0.9


def test_causal_methane_policy_scores_the_actual_focused_sweep() -> None:
    output_video = Path("outputs") / "_test-causal-methane.mp4"
    report_output = Path("outputs") / "_test-causal-methane.json"
    try:
        report = run_video_proof(
            output_video=output_video,
            report_output=report_output,
            scenario="methane",
            max_scans=8,
            render_scale=1,
            vertical_pass_seconds=0.005,
            horizontal_step_seconds=0.0,
            column_mass_fraction=1.0,
            row_mass_fraction=1.0,
            min_sweep_columns=30,
            min_sweep_rows=56,
            max_sweep_columns=320,
        )
    finally:
        output_video.unlink(missing_ok=True)
        report_output.unlink(missing_ok=True)

    assert report.timeline_integrity_verified
    assert report.actual_sweep_success_rate == 1.0
    assert report.mean_actual_sweep_mass_recall > 0.98
    assert report.mean_swept_area_fraction < 0.25
    assert report.initial_scan_duration_seconds < 0.5


def test_full_frame_processing_is_an_explicit_baseline() -> None:
    output_video = Path("outputs") / "_test-full-frame-baseline.mp4"
    report_output = Path("outputs") / "_test-full-frame-baseline.json"
    try:
        report = run_video_proof(
            output_video=output_video,
            report_output=report_output,
            scenario="object",
            evidence_mode="full-frame",
            max_scans=3,
            render_scale=1,
        )
    finally:
        output_video.unlink(missing_ok=True)
        report_output.unlink(missing_ok=True)

    assert report.evidence_mode == "full-frame"
    assert report.plume_mapping_mode == "explicit_full_frame_baseline"
    assert report.motion_source == "full_frame_opencv_px_per_second"
    assert report.scan_duration_seconds == 0.0
    assert report.actual_sweep_evaluated_predictions == 0
    assert report.actual_sweep_successes == 0
    assert report.actual_sweep_success_rate == 0.0
    assert report.sample_variation.full_frame_evidence_mode


def test_slow_scan_reacquires_full_scene_instead_of_capping_uncertainty() -> None:
    output_video = Path("outputs") / "_test-reacquisition.mp4"
    report_output = Path("outputs") / "_test-reacquisition.json"
    try:
        report = run_video_proof(
            output_video=output_video,
            report_output=report_output,
            scenario="methane",
            max_scans=2,
            render_scale=1,
        )
    finally:
        output_video.unlink(missing_ok=True)
        report_output.unlink(missing_ok=True)

    assert report.mean_uncertainty_radius > np.hypot(220, 320)
    assert report.final_swept_area_fraction == 1.0
    assert report.final_scan_duration_seconds > 300.0
    assert any(
        item.next_plan_reacquisition
        for item in report.operational_trace
    )


def test_real_video_stops_when_a_scan_cannot_finish_before_source_end() -> None:
    input_video = Path("outputs") / "_test-short-input.mp4"
    output_video = Path("outputs") / "_test-short-output.mp4"
    report_output = Path("outputs") / "_test-short-output.json"
    writer = cv2.VideoWriter(
        str(input_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (20, 16),
    )
    try:
        for index in range(6):
            writer.write(np.full((16, 20, 3), 20 + index, dtype=np.uint8))
    finally:
        writer.release()

    try:
        report = run_video_proof(
            input_video=input_video,
            output_video=output_video,
            report_output=report_output,
            scenario="object",
            max_scans=3,
            scan_rate_hz=100.0,
            render_scale=1,
        )
    finally:
        input_video.unlink(missing_ok=True)
        output_video.unlink(missing_ok=True)
        report_output.unlink(missing_ok=True)

    assert report.scans_completed == 0
    assert report.termination_reason == "source_exhausted_before_scan"
    assert report.timeline_integrity_verified
    assert report.operational_trace == ()
    assert report.belief_integrity.status == "not_evaluated"
    assert report.sample_variation.status == "not_evaluated"
    assert report.sample_variation.comparisons == 0
