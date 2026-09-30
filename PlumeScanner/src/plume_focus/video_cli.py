"""Command-line adapter for annotated experiment proofs."""

from __future__ import annotations

import argparse
from pathlib import Path

from plume_focus.experiment_contracts import (
    DEFAULT_METHANE_REPORT_OUTPUT,
    DEFAULT_METHANE_VIDEO_OUTPUT,
    DEFAULT_PLUME_REPORT_OUTPUT,
    DEFAULT_PLUME_VIDEO_OUTPUT,
    DEFAULT_RENDER_SCALE,
    DEFAULT_REPORT_OUTPUT,
    DEFAULT_VIDEO_OUTPUT,
)
from plume_focus.video_workflow import run_video_proof


def main(argv: list[str] | None = None) -> None:
    """CLI for the OpenCV moving-object proof demo."""

    parser = argparse.ArgumentParser(
        description=(
            "Run a moving-object or plume-like video proof for Plume Focus."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Optional real video path. If omitted, a synthetic video is "
            "generated."
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=["object", "plume", "methane"],
        default="object",
        help="Synthetic/processing scenario to use.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Annotated MP4 output path.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON report path.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=48,
        help="Deprecated compatibility alias for --max-scans.",
    )
    parser.add_argument(
        "--max-scans",
        type=int,
        default=None,
        help="Maximum number of physically completed scan/update cycles.",
    )
    parser.add_argument(
        "--evidence-mode",
        choices=["scan", "full-frame"],
        default="scan",
        help=(
            "Use causal scan evidence by default or an explicit full-frame "
            "baseline."
        ),
    )
    parser.add_argument(
        "--scan-mode",
        choices=["point", "column"],
        default=None,
        help="Use point-footprint sampling or mechanical column sweeps.",
    )
    parser.add_argument(
        "--scan-rate-hz",
        type=float,
        default=None,
        help=(
            "Sequential sample rate. For methane, this is point visits "
            "per second."
        ),
    )
    parser.add_argument(
        "--sample-stride-px",
        type=int,
        default=None,
        help="Grid pixels covered by one point sample footprint.",
    )
    parser.add_argument(
        "--vertical-pass-seconds",
        type=float,
        default=None,
        help=(
            "Column-sweep time for a full-height vertical pass; shorter row "
            "spans scale proportionally."
        ),
    )
    parser.add_argument(
        "--horizontal-step-seconds",
        type=float,
        default=None,
        help="Column-sweep time to step from one vertical pass to the next.",
    )
    parser.add_argument(
        "--column-stride-px",
        type=int,
        default=None,
        help="Column-sweep horizontal stride in image-grid pixels.",
    )
    parser.add_argument(
        "--column-mass-fraction",
        type=float,
        default=None,
        help=(
            "Fraction of belief column mass to sweep before applying the "
            "column budget."
        ),
    )
    parser.add_argument(
        "--row-mass-fraction",
        type=float,
        default=None,
        help="Fraction of belief row mass to keep for vertical sweep trimming.",
    )
    parser.add_argument(
        "--min-sweep-columns",
        type=int,
        default=None,
        help="Minimum number of column axes to sweep in column mode.",
    )
    parser.add_argument(
        "--min-sweep-rows",
        type=int,
        default=None,
        help="Minimum number of vertical rows to keep in column mode.",
    )
    parser.add_argument(
        "--max-sweep-columns",
        type=int,
        default=None,
        help="Maximum number of column axes to sweep in column mode.",
    )
    parser.add_argument(
        "--methane-wind-dy",
        type=float,
        default=None,
        help="Synthetic methane wind row velocity in pixels per frame.",
    )
    parser.add_argument(
        "--methane-wind-dx",
        type=float,
        default=None,
        help="Synthetic methane wind column velocity in pixels per frame.",
    )
    parser.add_argument(
        "--start-full-scene",
        action="store_true",
        help=(
            "For methane, start with a full-scene discovery sweep before "
            "Bayesian focus."
        ),
    )
    parser.add_argument(
        "--source-free",
        action="store_true",
        help=(
            "For methane, infer the plume map from scan evidence and known "
            "wind without using a source point."
        ),
    )
    parser.add_argument(
        "--render-scale",
        type=int,
        default=DEFAULT_RENDER_SCALE,
        help="Video render scale for sharper labels.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help=(
            "Subsample real-video/full-frame refresh; physical scan time "
            "remains seconds."
        ),
    )
    parser.add_argument(
        "--success-coverage",
        type=float,
        default=0.9,
        help=(
            "Minimum completion-time plume probability mass required for "
            "sweep success."
        ),
    )
    args = parser.parse_args(argv)

    if args.output is not None:
        output = args.output
    elif args.scenario == "methane":
        output = DEFAULT_METHANE_VIDEO_OUTPUT
    elif args.scenario == "plume":
        output = DEFAULT_PLUME_VIDEO_OUTPUT
    else:
        output = DEFAULT_VIDEO_OUTPUT

    if args.report is not None:
        report_path = args.report
    elif args.scenario == "methane":
        report_path = DEFAULT_METHANE_REPORT_OUTPUT
    elif args.scenario == "plume":
        report_path = DEFAULT_PLUME_REPORT_OUTPUT
    else:
        report_path = DEFAULT_REPORT_OUTPUT

    report = run_video_proof(
        input_video=args.input,
        output_video=output,
        report_output=report_path,
        max_frames=args.max_frames,
        max_scans=args.max_scans,
        scenario=args.scenario,
        evidence_mode=args.evidence_mode,
        scan_mode=args.scan_mode,
        scan_rate_hz=args.scan_rate_hz,
        sample_stride_px=args.sample_stride_px,
        vertical_pass_seconds=args.vertical_pass_seconds,
        horizontal_step_seconds=args.horizontal_step_seconds,
        column_stride_px=args.column_stride_px,
        column_mass_fraction=args.column_mass_fraction,
        row_mass_fraction=args.row_mass_fraction,
        min_sweep_columns=args.min_sweep_columns,
        min_sweep_rows=args.min_sweep_rows,
        max_sweep_columns=args.max_sweep_columns,
        methane_wind_dy=args.methane_wind_dy,
        methane_wind_dx=args.methane_wind_dx,
        start_full_scene=args.start_full_scene,
        source_free=args.source_free,
        render_scale=args.render_scale,
        frame_stride=args.frame_stride,
        success_coverage_threshold=args.success_coverage,
    )
    print(report.to_json())
    print(f"Wrote annotated proof video to {output}")
    print(f"Wrote report to {report_path}")
