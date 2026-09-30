"""Filesystem and rendering adapter for the headless experiment engine."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from plume_focus.experiment_contracts import (
    DEFAULT_RENDER_SCALE,
    DEFAULT_REPORT_OUTPUT,
    DEFAULT_VIDEO_OUTPUT,
    EvidenceMode,
    ScanMode,
    Scenario,
    VideoProofReport,
)
from plume_focus.experiment_runner import (
    ExperimentRequest,
    run_experiment,
)
from plume_focus.model_profiles import ModelProfile
from plume_focus.plume_dynamics import PlumeDynamicsConfig
from plume_focus.video_rendering import VideoRenderer


def run_video_proof(
    *,
    input_video: Path | None = None,
    output_video: Path = DEFAULT_VIDEO_OUTPUT,
    report_output: Path = DEFAULT_REPORT_OUTPUT,
    max_frames: int = 48,
    max_scans: int | None = None,
    scenario: Scenario = "object",
    evidence_mode: EvidenceMode = "scan",
    scan_mode: ScanMode | None = None,
    scan_rate_hz: float | None = None,
    sample_stride_px: int | None = None,
    vertical_pass_seconds: float | None = None,
    horizontal_step_seconds: float | None = None,
    column_stride_px: int | None = None,
    column_mass_fraction: float | None = None,
    row_mass_fraction: float | None = None,
    min_sweep_columns: int | None = None,
    min_sweep_rows: int | None = None,
    max_sweep_columns: int | None = None,
    methane_wind_dy: float | None = None,
    methane_wind_dx: float | None = None,
    start_full_scene: bool = False,
    source_free: bool = False,
    render_scale: int = DEFAULT_RENDER_SCALE,
    frame_stride: int = 1,
    success_coverage_threshold: float = 0.9,
    model_profile: ModelProfile | None = None,
    plume_dynamics_config: PlumeDynamicsConfig | None = None,
) -> VideoProofReport:
    """Run a headless experiment with an MP4 observer and JSON adapter."""

    request = ExperimentRequest(
        input_video=input_video,
        max_frames=max_frames,
        max_scans=max_scans,
        scenario=scenario,
        evidence_mode=evidence_mode,
        scan_mode=scan_mode,
        scan_rate_hz=scan_rate_hz,
        sample_stride_px=sample_stride_px,
        vertical_pass_seconds=vertical_pass_seconds,
        horizontal_step_seconds=horizontal_step_seconds,
        column_stride_px=column_stride_px,
        column_mass_fraction=column_mass_fraction,
        row_mass_fraction=row_mass_fraction,
        min_sweep_columns=min_sweep_columns,
        min_sweep_rows=min_sweep_rows,
        max_sweep_columns=max_sweep_columns,
        methane_wind_dy=methane_wind_dy,
        methane_wind_dx=methane_wind_dx,
        start_full_scene=start_full_scene,
        source_free=source_free,
        frame_stride=frame_stride,
        success_coverage_threshold=success_coverage_threshold,
        model_profile=model_profile,
        plume_dynamics_config=plume_dynamics_config,
    )
    report_output.parent.mkdir(parents=True, exist_ok=True)
    renderer = VideoRenderer(output_video, render_scale=render_scale)
    result = run_experiment(request, observer=renderer)
    report = replace(result.report, output_video=str(output_video))
    report_output.write_text(
        report.to_json() + "\n",
        encoding="utf-8",
    )
    return report
