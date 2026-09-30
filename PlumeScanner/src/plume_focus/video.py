"""Compatibility facade and entry point for video proof generation."""

from __future__ import annotations

from plume_focus.experiment_contracts import (
    ATTENTION_HEATMAP_DISPLAY_THRESHOLD,
    ATTENTION_INNER_THRESHOLD,
    ATTENTION_OUTER_THRESHOLD,
    DEFAULT_METHANE_REPORT_OUTPUT,
    DEFAULT_METHANE_VIDEO_OUTPUT,
    DEFAULT_PLUME_REPORT_OUTPUT,
    DEFAULT_PLUME_VIDEO_OUTPUT,
    DEFAULT_RENDER_SCALE,
    DEFAULT_REPORT_OUTPUT,
    DEFAULT_SCAN_RATE_HZ,
    DEFAULT_VIDEO_OUTPUT,
    VIDEO_HEADER_HEIGHT,
    EvidenceMode,
    RasterVideoScan,
    ScanMode,
    Scenario,
    VideoProofReport,
)
from plume_focus.experiment_metrics import (
    estimate_opencv_motion,
    foreground_mask,
    js_divergence,
    kernel_plume_attention_field,
    kl_divergence,
    normalize_distribution,
    plume_truth_distribution,
)
from plume_focus.scan_execution import (
    SampleValuesAt,
    budget_column_sweep_mask,
    column_sweep_scan_moving_frames,
    raster_scan_moving_frames,
)
from plume_focus.video_cli import main
from plume_focus.video_rendering import (
    ATTENTION_INNER_COLOR,
    ATTENTION_OUTER_COLOR,
    NEXT_SWEEP_COLOR,
    TRUE_NEXT_COLOR,
    box_or_none,
    compose_proof_frame,
    draw_attention_contour,
    draw_binary_mask_contour,
    draw_box,
    draw_box_label,
    draw_fitted_text,
    draw_motion_arrow,
    draw_right_panel_key,
    draw_scan_metadata,
    overlay_attention,
    project_single_pixel_scan,
    render_video_header,
    scale_box,
)
from plume_focus.video_sources import (
    SyntheticMethaneLeakConfig,
    SyntheticPlumeVideoConfig,
    SyntheticVideoConfig,
    default_methane_roi,
    generate_synthetic_methane_frame,
    generate_synthetic_methane_frames,
    generate_synthetic_methane_values,
    generate_synthetic_object_frame,
    generate_synthetic_object_values,
    generate_synthetic_plume_frame,
    generate_synthetic_plume_frames,
    generate_synthetic_plume_values,
    generate_synthetic_video_frames,
    load_video_frames,
    methane_scan_roi,
)
from plume_focus.video_workflow import run_video_proof


# Private compatibility alias used by older in-repository callers.
_box_or_none = box_or_none


if __name__ == "__main__":
    main()
