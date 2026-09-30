"""OpenCV rendering adapter for active-sensing experiment steps."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from plume_focus.experiment_contracts import (
    ATTENTION_HEATMAP_DISPLAY_RELATIVE_DENSITY,
    ATTENTION_INNER_RELATIVE_DENSITY,
    ATTENTION_OUTER_RELATIVE_DENSITY,
    RasterVideoScan,
    ScanMode,
    VIDEO_HEADER_HEIGHT,
)
from plume_focus.experiment_runner import (
    ExperimentContext,
    ExperimentStep,
)
from plume_focus.motion import MotionEstimate
from plume_focus.predictor import (
    BoundingBox,
    bounding_box_from_mask,
)


TRUE_NEXT_COLOR = (255, 255, 255)
NEXT_SWEEP_COLOR = (249, 209, 102)
ATTENTION_INNER_COLOR = (34, 197, 94)
ATTENTION_OUTER_COLOR = (249, 115, 22)


class VideoRenderer:
    """Observer that renders experiment steps to an annotated MP4."""

    def __init__(self, output_path: Path, *, render_scale: int = 1) -> None:
        self.output_path = Path(output_path)
        self.render_scale = max(1, int(render_scale))
        self._context: ExperimentContext | None = None
        self._writer: cv2.VideoWriter | None = None

    def start(self, context: ExperimentContext) -> None:
        """Open the video writer after runtime dimensions are known."""

        if self._writer is not None:
            raise RuntimeError("VideoRenderer has already been started.")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        height, width = context.shape
        writer = cv2.VideoWriter(
            str(self.output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            context.render_fps,
            (
                width * 3 * self.render_scale,
                (height + VIDEO_HEADER_HEIGHT) * self.render_scale,
            ),
        )
        if not writer.isOpened():
            writer.release()
            raise ValueError(f"Could not write video: {self.output_path}")
        self._context = context
        self._writer = writer

    def observe(self, step: ExperimentStep) -> None:
        """Render one completed experiment step."""

        if self._writer is None or self._context is None:
            raise RuntimeError("VideoRenderer.observe called before start.")
        motion_for_display = MotionEstimate(
            dy_per_t=(
                step.motion_px_per_second.dy_per_t
                / self._context.render_fps
            ),
            dx_per_t=(
                step.motion_px_per_second.dx_per_t
                / self._context.render_fps
            ),
            method=step.motion_px_per_second.method,
            confidence=step.motion_px_per_second.confidence,
        )
        rendered = compose_proof_frame(
            current=step.current_frame,
            scanned_projection=project_single_pixel_scan(step.scan),
            next_frame=step.target_frame,
            attention=step.attention_field,
            current_box=step.scan.sampled_box,
            next_box=box_or_none(step.target_support),
            next_scan_mask=step.next_scan_mask,
            motion=motion_for_display,
            scan_mode=step.scan.scan_mode,
            scan_rate_hz=step.scan.scan_rate_hz,
            sample_stride_px=step.scan.sample_stride_px,
            vertical_pass_seconds=step.scan.vertical_pass_seconds,
            mean_vertical_pass_seconds=(
                step.scan.mean_vertical_pass_seconds
            ),
            horizontal_step_seconds=step.scan.horizontal_step_seconds,
            scan_duration_seconds=step.scan.scan_duration_seconds,
            scan_area_fraction=step.scan.mechanical_sweep_area_fraction,
            render_scale=self.render_scale,
        )
        self._writer.write(rendered)

    def close(self) -> None:
        """Release the writer; repeated calls are safe."""

        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._context = None


def project_single_pixel_scan(
    scan: RasterVideoScan,
) -> NDArray[np.uint8]:
    """Display measured intensity without painting over scalar evidence.

    Age and path belong on separate panels. In particular, a one-pixel contour
    around a one-pixel sampled column replaces every measured value.
    """

    projected = cv2.cvtColor(scan.values, cv2.COLOR_GRAY2BGR)
    projected[~scan.sampled_mask] = (5, 9, 15)
    return projected


def project_single_pixel_scan_legacy_overlay(
    scan: RasterVideoScan,
) -> NDArray[np.uint8]:
    """Historical rendering, retained only for explicit matched diagnostics."""

    values_bgr = cv2.cvtColor(scan.values, cv2.COLOR_GRAY2BGR)
    age = np.zeros(scan.sampled_at_seconds.shape, dtype=float)
    age[scan.sampled_mask] = (
        scan.completed_at_seconds
        - scan.sampled_at_seconds[scan.sampled_mask]
    )
    sampled_age = age[scan.sampled_mask]
    if sampled_age.size:
        age_norm = np.zeros_like(age)
        age_norm[scan.sampled_mask] = (
            sampled_age - float(sampled_age.min())
        ) / max(float(sampled_age.max() - sampled_age.min()), 1e-9)
    else:
        age_norm = age
    time_color = cv2.applyColorMap(
        (age_norm * 255).astype(np.uint8),
        cv2.COLORMAP_VIRIDIS,
    )
    projected = cv2.addWeighted(values_bgr, 0.62, time_color, 0.48, 0)
    projected[~scan.sampled_mask] = (5, 9, 15)
    if scan.scan_mode == "column" and scan.sampled_box is not None:
        draw_box(
            projected,
            scan.sampled_box,
            NEXT_SWEEP_COLOR,
            "sweep envelope",
        )
    elif scan.roi_box is not None:
        draw_box(
            projected,
            scan.roi_box,
            NEXT_SWEEP_COLOR,
            "scan envelope",
        )
    if np.any(scan.sampled_mask):
        contours, _hierarchy = cv2.findContours(
            scan.sampled_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(projected, contours, -1, (125, 211, 252), 1)
    return projected.astype(np.uint8)


def compose_proof_frame(
    *,
    current: NDArray[np.uint8],
    scanned_projection: NDArray[np.uint8],
    next_frame: NDArray[np.uint8],
    attention: NDArray[np.float64],
    current_box: BoundingBox | None,
    next_box: BoundingBox | None,
    next_scan_mask: NDArray[np.bool_] | None,
    motion: MotionEstimate,
    scan_mode: ScanMode,
    scan_rate_hz: float,
    sample_stride_px: int,
    vertical_pass_seconds: float,
    mean_vertical_pass_seconds: float,
    horizontal_step_seconds: float,
    scan_duration_seconds: float,
    scan_area_fraction: float,
    render_scale: int = 1,
) -> NDArray[np.uint8]:
    """Compose a side-by-side annotated frame for the proof video."""

    render_scale = max(1, int(render_scale))
    target_size = (
        current.shape[1] * render_scale,
        current.shape[0] * render_scale,
    )
    left = cv2.resize(
        cv2.cvtColor(current, cv2.COLOR_GRAY2BGR),
        target_size,
        interpolation=cv2.INTER_CUBIC,
    )
    middle = cv2.resize(
        scanned_projection,
        target_size,
        interpolation=cv2.INTER_CUBIC,
    )
    resized_next = cv2.resize(
        cv2.cvtColor(next_frame, cv2.COLOR_GRAY2BGR),
        target_size,
        interpolation=cv2.INTER_CUBIC,
    )
    resized_attention = cv2.resize(
        attention,
        target_size,
        interpolation=cv2.INTER_LINEAR,
    )
    right = overlay_attention(resized_next, resized_attention)

    if current_box is not None:
        scaled_current_box = scale_box(current_box, render_scale)
        draw_box(
            left,
            scaled_current_box,
            (125, 211, 252),
            "sampled envelope",
            render_scale=render_scale,
        )
        draw_motion_arrow(
            left,
            scaled_current_box,
            motion,
            render_scale=render_scale,
        )
    if next_box is not None:
        draw_box(
            right,
            scale_box(next_box, render_scale),
            TRUE_NEXT_COLOR,
            "true plume env",
            render_scale=render_scale,
            label_position="bottom",
        )
    if next_scan_mask is not None and np.any(next_scan_mask):
        draw_binary_mask_contour(
            right,
            next_scan_mask,
            NEXT_SWEEP_COLOR,
            "next sweep mask",
            render_scale=render_scale,
        )

    draw_scan_metadata(
        middle,
        scan_mode=scan_mode,
        scan_rate_hz=scan_rate_hz,
        sample_stride_px=sample_stride_px,
        vertical_pass_seconds=vertical_pass_seconds,
        mean_vertical_pass_seconds=mean_vertical_pass_seconds,
        horizontal_step_seconds=horizontal_step_seconds,
        scan_duration_seconds=scan_duration_seconds,
        scan_area_fraction=scan_area_fraction,
        render_scale=render_scale,
    )
    draw_attention_contour(
        right,
        resized_attention,
        render_scale=render_scale,
    )
    draw_right_panel_key(right, render_scale=render_scale, truth_available=next_box is not None)
    panel = np.concatenate([left, middle, right], axis=1)
    header = render_video_header(
        current.shape[1] * render_scale,
        panel.shape[1],
        scan_mode=scan_mode,
        render_scale=render_scale,
    )
    return np.vstack([header, panel])


def render_video_header(
    panel_width: int,
    total_width: int,
    *,
    scan_mode: ScanMode = "point",
    render_scale: int = 1,
) -> NDArray[np.uint8]:
    """Render a clean two-line header for the three proof panels."""

    header = np.zeros(
        (VIDEO_HEADER_HEIGHT * render_scale, total_width, 3),
        dtype=np.uint8,
    )
    header[:] = (13, 22, 37)
    scan_title = (
        "Current Column Sweep"
        if scan_mode == "column"
        else "Current Point Scan"
    )
    scan_subtitle = (
        "completed time-compressed sweep"
        if scan_mode == "column"
        else "completed scan summary"
    )
    labels = [
        ("Current Frame", "motion / wind vector"),
        (scan_title, scan_subtitle),
        ("Next Scan Completion", "relative density + planned sweep"),
    ]

    for idx, (title, subtitle) in enumerate(labels):
        x0 = idx * panel_width
        x1 = x0 + panel_width
        if idx > 0:
            cv2.line(
                header,
                (x0, 8 * render_scale),
                (x0, (VIDEO_HEADER_HEIGHT - 8) * render_scale),
                (38, 58, 84),
                render_scale,
                cv2.LINE_AA,
            )
        cv2.rectangle(
            header,
            (x0 + 5 * render_scale, 6 * render_scale),
            (x1 - 6 * render_scale, (VIDEO_HEADER_HEIGHT - 7) * render_scale),
            (18, 31, 49),
            render_scale,
        )
        draw_fitted_text(
            header,
            title,
            x0 + 10 * render_scale,
            24 * render_scale,
            panel_width - 20 * render_scale,
            (226, 242, 255),
            base_scale=0.46 * render_scale,
            thickness=render_scale,
        )
        draw_fitted_text(
            header,
            subtitle,
            x0 + 10 * render_scale,
            42 * render_scale,
            panel_width - 20 * render_scale,
            (148, 170, 196),
            base_scale=0.34 * render_scale,
            thickness=max(1, render_scale),
        )

    return header


def draw_fitted_text(
    image: NDArray[np.uint8],
    text: str,
    x: int,
    y: int,
    max_width: int,
    color: tuple[int, int, int],
    *,
    base_scale: float,
    thickness: int,
) -> None:
    """Draw text scaled down just enough to fit its panel."""

    scale = base_scale
    while scale > 0.22:
        size, _baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            thickness,
        )
        if size[0] <= max_width:
            break
        scale -= 0.02
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def overlay_attention(
    frame_bgr: NDArray[np.uint8],
    attention: NDArray[np.float64],
) -> NDArray[np.uint8]:
    """Overlay a predictive attention field on a BGR frame."""

    normalized = np.clip(attention, 0.0, 1.0)
    if np.ptp(normalized) <= 1e-12:
        return frame_bgr.copy()
    heatmap = cv2.applyColorMap(
        (normalized * 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    mask = (
        normalized[..., None]
        > ATTENTION_HEATMAP_DISPLAY_RELATIVE_DENSITY
    )
    blended = cv2.addWeighted(frame_bgr, 0.72, heatmap, 0.55, 0)
    return np.where(mask, blended, frame_bgr).astype(np.uint8)


def draw_attention_contour(
    frame_bgr: NDArray[np.uint8],
    attention: NDArray[np.float64],
    *,
    render_scale: int = 1,
) -> None:
    """Draw smooth-bound contours at two relative-density levels."""

    for threshold, color in [
        (ATTENTION_INNER_RELATIVE_DENSITY, ATTENTION_INNER_COLOR),
        (ATTENTION_OUTER_RELATIVE_DENSITY, ATTENTION_OUTER_COLOR),
    ]:
        mask = (attention >= threshold).astype(np.uint8)
        contours, _hierarchy = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(
            frame_bgr,
            contours,
            -1,
            color,
            max(1, render_scale),
        )


def draw_right_panel_key(
    frame_bgr: NDArray[np.uint8],
    *,
    render_scale: int = 1,
    truth_available: bool = True,
) -> None:
    """Draw a compact key for the right-panel overlays."""

    entries = [
        ("heat: relative density", None),
        ("cyan: next sweep mask", NEXT_SWEEP_COLOR),
        ("green: high-belief contour", ATTENTION_INNER_COLOR),
        ("blue: conservative contour", ATTENTION_OUTER_COLOR),
    ]
    if truth_available:
        entries.insert(1, ("white: true plume envelope", TRUE_NEXT_COLOR))
    margin = 6 * render_scale
    line_h = 12 * render_scale
    width = min(
        frame_bgr.shape[1] - 2 * margin,
        128 * render_scale,
    )
    height = len(entries) * line_h + 12 * render_scale
    x0 = margin
    y0 = margin
    x1 = x0 + width
    y1 = y0 + height
    cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (9, 14, 24), -1)
    cv2.rectangle(
        frame_bgr,
        (x0, y0),
        (x1, y1),
        (38, 58, 84),
        max(1, render_scale),
    )

    for idx, (label, color) in enumerate(entries):
        y = y0 + (idx + 1) * line_h
        swatch_x = x0 + 6 * render_scale
        text_x = x0 + 22 * render_scale
        if color is None:
            heat = np.linspace(
                0,
                255,
                10 * render_scale,
                dtype=np.uint8,
            )[None, :]
            heat = np.repeat(heat, 5 * render_scale, axis=0)
            heat_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
            sy0 = max(0, y - 7 * render_scale)
            sy1 = min(
                frame_bgr.shape[0],
                sy0 + heat_bgr.shape[0],
            )
            sx1 = min(
                frame_bgr.shape[1],
                swatch_x + heat_bgr.shape[1],
            )
            frame_bgr[sy0:sy1, swatch_x:sx1] = heat_bgr[
                : sy1 - sy0,
                : sx1 - swatch_x,
            ]
        else:
            cv2.line(
                frame_bgr,
                (swatch_x, y - 3 * render_scale),
                (swatch_x + 10 * render_scale, y - 3 * render_scale),
                color,
                max(2, render_scale),
                cv2.LINE_AA,
            )
        draw_fitted_text(
            frame_bgr,
            label,
            text_x,
            y,
            width - 28 * render_scale,
            (226, 242, 255),
            base_scale=0.27 * render_scale,
            thickness=max(1, render_scale),
        )


def draw_binary_mask_contour(
    frame_bgr: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    color: tuple[int, int, int],
    label: str,
    *,
    render_scale: int = 1,
) -> None:
    """Draw an outline and label for a boolean scan/ROI mask."""

    scaled_mask = cv2.resize(
        mask.astype(np.uint8),
        (frame_bgr.shape[1], frame_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    contours, _hierarchy = cv2.findContours(
        scaled_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(
        frame_bgr,
        contours,
        -1,
        color,
        max(1, render_scale),
    )
    if np.any(scaled_mask):
        draw_box_label(
            frame_bgr,
            bounding_box_from_mask(scaled_mask.astype(bool)),
            color,
            label,
            render_scale=render_scale,
        )


def draw_box(
    frame_bgr: NDArray[np.uint8],
    box: BoundingBox,
    color: tuple[int, int, int],
    label: str,
    *,
    render_scale: int = 1,
    label_position: Literal["auto", "bottom"] = "auto",
) -> None:
    """Draw an image-coordinate bounding box."""

    cv2.rectangle(
        frame_bgr,
        (box.min_col, box.min_row),
        (box.max_col, box.max_row),
        color,
        max(1, render_scale),
    )
    draw_box_label(
        frame_bgr,
        box,
        color,
        label,
        render_scale=render_scale,
        label_position=label_position,
    )


def draw_box_label(
    frame_bgr: NDArray[np.uint8],
    box: BoundingBox,
    color: tuple[int, int, int],
    label: str,
    *,
    render_scale: int = 1,
    label_position: Literal["auto", "bottom"] = "auto",
) -> None:
    """Draw a label near a box without letting it clip at frame edges."""

    scale = 0.34 * render_scale
    thickness = max(1, render_scale)
    text_size, baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )
    text_w, text_h = text_size
    frame_h, frame_w = frame_bgr.shape[:2]

    x = int(
        np.clip(
            box.min_col + 2 * render_scale,
            render_scale,
            max(
                render_scale,
                frame_w - text_w - 3 * render_scale,
            ),
        )
    )
    if label_position == "bottom":
        y = min(
            frame_h - baseline - 3 * render_scale,
            box.max_row + text_h + 5 * render_scale,
        )
    elif box.min_row - text_h - 7 * render_scale >= 0:
        y = box.min_row - 4 * render_scale
    else:
        y = min(
            frame_h - baseline - 3 * render_scale,
            box.min_row + text_h + 5 * render_scale,
        )

    bg_top_left = (
        max(0, x - 2 * render_scale),
        max(0, y - text_h - 3 * render_scale),
    )
    bg_bottom_right = (
        min(frame_w - 1, x + text_w + 2 * render_scale),
        min(frame_h - 1, y + baseline + 2 * render_scale),
    )
    cv2.rectangle(
        frame_bgr,
        bg_top_left,
        bg_bottom_right,
        (9, 14, 24),
        -1,
    )
    cv2.putText(
        frame_bgr,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_motion_arrow(
    frame_bgr: NDArray[np.uint8],
    box: BoundingBox,
    motion: MotionEstimate,
    *,
    render_scale: int = 1,
) -> None:
    """Draw the estimated immediate motion vector."""

    center_x = int(round((box.min_col + box.max_col) / 2.0))
    center_y = int(round((box.min_row + box.max_row) / 2.0))
    end_x = int(
        round(center_x + motion.dx_per_t * 5.0 * render_scale)
    )
    end_y = int(
        round(center_y + motion.dy_per_t * 5.0 * render_scale)
    )
    cv2.arrowedLine(
        frame_bgr,
        (center_x, center_y),
        (end_x, end_y),
        (34, 197, 94),
        max(2, 2 * render_scale),
        cv2.LINE_AA,
        tipLength=0.25,
    )


def draw_scan_metadata(
    frame_bgr: NDArray[np.uint8],
    *,
    scan_mode: ScanMode,
    scan_rate_hz: float,
    sample_stride_px: int,
    vertical_pass_seconds: float,
    mean_vertical_pass_seconds: float,
    horizontal_step_seconds: float,
    scan_duration_seconds: float,
    scan_area_fraction: float,
    render_scale: int = 1,
) -> None:
    """Annotate the scan with its physical sampling assumptions."""

    scan_name = "ROI scan" if scan_area_fraction < 0.99 else "full scan"
    if scan_mode == "column":
        lines = [
            "time-compressed scan summary",
            (
                f"{scan_duration_seconds:.1f}s {scan_name}; "
                f"{mean_vertical_pass_seconds:.2f}s avg v + "
                f"{horizontal_step_seconds:.2f}s step"
            ),
        ]
    elif sample_stride_px > 1:
        lines = [
            f"{scan_rate_hz:.0f} pts/s, {sample_stride_px}px footprint",
            f"{scan_duration_seconds:.1f}s {scan_name}",
        ]
    else:
        lines = [
            f"{scan_rate_hz:.0f} px/s",
            f"{scan_duration_seconds:.1f}s {scan_name}",
        ]
    x = 6 * render_scale
    band_height = 32 * render_scale
    y0 = frame_bgr.shape[0] - band_height - 4 * render_scale
    y1 = frame_bgr.shape[0] - 4 * render_scale
    cv2.rectangle(
        frame_bgr,
        (x - 3 * render_scale, y0),
        (frame_bgr.shape[1] - 6 * render_scale, y1),
        (9, 14, 24),
        -1,
    )
    draw_fitted_text(
        frame_bgr,
        lines[0],
        x,
        y0 + 12 * render_scale,
        frame_bgr.shape[1] - x - 12 * render_scale,
        (226, 242, 255),
        base_scale=0.34 * render_scale,
        thickness=max(1, render_scale),
    )
    draw_fitted_text(
        frame_bgr,
        lines[1],
        x,
        y0 + 26 * render_scale,
        frame_bgr.shape[1] - x - 12 * render_scale,
        (148, 170, 196),
        base_scale=0.30 * render_scale,
        thickness=max(1, render_scale),
    )


def scale_box(box: BoundingBox, scale: int) -> BoundingBox:
    """Scale an inclusive pixel box into a resized image."""

    return BoundingBox(
        min_row=box.min_row * scale,
        min_col=box.min_col * scale,
        max_row=(box.max_row + 1) * scale - 1,
        max_col=(box.max_col + 1) * scale - 1,
    )


def box_or_none(
    mask: NDArray[np.bool_],
) -> BoundingBox | None:
    """Return a box for a non-empty mask."""

    if not np.any(mask):
        return None
    return bounding_box_from_mask(mask)
