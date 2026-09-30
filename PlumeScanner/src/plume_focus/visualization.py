"""SVG visualization for the Plume Focus demo.

The visual output is intentionally simple and dependency-free. It shows the
boundary-analysis idea directly: reduced sensor readings, inferred motion, and
the conservative attention boxes used to focus the next scan.
"""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from plume_focus.motion import MotionEstimate, estimate_centroid_motion
from plume_focus.predictor import BoundConfig, BoundingBox, bounding_box_from_mask, predict_attention_area
from plume_focus.scanner import SinglePixelScanner
from plume_focus.synthetic import moving_gaussian_frame


DEFAULT_OUTPUT = Path("outputs") / "plume-focus-demo.svg"


def build_visual_demo(output_path: Path = DEFAULT_OUTPUT) -> Path:
    """Create an SVG visual demo and return the written path."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shape = (64, 80)
    start_center = (25.0, 25.0)
    true_motion = MotionEstimate(dy_per_t=1.25, dx_per_t=2.0, method="truth")

    frame_0 = moving_gaussian_frame(shape, start_center, true_motion, time=0.0)
    frame_1 = moving_gaussian_frame(shape, start_center, true_motion, time=1.0)
    frame_2 = moving_gaussian_frame(shape, start_center, true_motion, time=2.0)

    measured_motion = estimate_centroid_motion(frame_0, frame_1)
    average_wind_motion = MotionEstimate(dy_per_t=1.0, dx_per_t=1.7, method="average_wind_proxy", confidence=0.6)

    scanner = SinglePixelScanner(shape, dwell_time=0.00005, stabilization_time=0.00001)
    scan = scanner.scan(lambda t: moving_gaussian_frame(shape, start_center, true_motion, time=1.0 + t))

    motion_prediction = predict_attention_area(
        frame_1,
        measured_motion,
        bounds=BoundConfig(wind_error_speed=0.75, model_error_radius=1.0, scan_age=scan.max_age),
    )
    wind_prediction = predict_attention_area(
        frame_1,
        average_wind_motion,
        bounds=BoundConfig(wind_error_speed=1.5, model_error_radius=2.0, scan_age=scan.max_age),
    )
    true_next_box = bounding_box_from_mask(frame_2 >= frame_2.max() * 0.25)

    svg = _render_svg(
        panels=[
            _Panel("Initial plume, t=0", frame_0, []),
            _Panel("Single-pixel scan, t~1", scan.values, [("scan source", motion_prediction.source_box, "#7dd3fc")]),
            _Panel("True next plume, t=2", frame_2, [("true next", true_next_box, "#ffffff")]),
            _Panel(
                "Predicted attention bounds",
                frame_2,
                [
                    ("motion vector bound", motion_prediction.attention_box, "#22c55e"),
                    ("average wind bound", wind_prediction.attention_box, "#f97316"),
                    ("true next", true_next_box, "#ffffff"),
                ],
            ),
        ],
        summary=[
            f"True motion: dy={true_motion.dy_per_t:.2f}, dx={true_motion.dx_per_t:.2f} px/t",
            f"Measured motion: dy={measured_motion.dy_per_t:.2f}, dx={measured_motion.dx_per_t:.2f} px/t",
            f"Scan duration: {scan.scan_duration:.4f} t; oldest pixel age: {scan.max_age:.4f} t",
            f"Motion bound radius: {motion_prediction.uncertainty_radius:.2f} px",
            f"Average-wind bound radius: {wind_prediction.uncertainty_radius:.2f} px",
        ],
    )
    output_path.write_text(svg, encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the visual demo."""

    parser = argparse.ArgumentParser(description="Write a visual SVG demo for Plume Focus.")
    parser.add_argument(
        "output",
        nargs="?",
        default=str(DEFAULT_OUTPUT),
        help=f"SVG output path. Defaults to {DEFAULT_OUTPUT.as_posix()}.",
    )
    args = parser.parse_args(argv)

    written = build_visual_demo(Path(args.output))
    print(f"Wrote visual demo to {written}")


class _Panel:
    def __init__(
        self,
        title: str,
        frame: NDArray[np.float64],
        boxes: list[tuple[str, BoundingBox, str]],
    ) -> None:
        self.title = title
        self.frame = frame
        self.boxes = boxes


def _render_svg(panels: list[_Panel], summary: list[str]) -> str:
    cell = 5
    panel_w = 80 * cell
    panel_h = 64 * cell
    gap = 42
    margin = 34
    title_h = 34
    summary_h = 110
    width = margin * 2 + panel_w * 2 + gap
    height = margin * 2 + title_h * 2 + panel_h * 2 + gap + summary_h

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        "<filter id=\"soft-shadow\" x=\"-10%\" y=\"-10%\" width=\"120%\" height=\"120%\">",
        "<feDropShadow dx=\"0\" dy=\"8\" stdDeviation=\"8\" flood-color=\"#020617\" flood-opacity=\"0.32\"/>",
        "</filter>",
        "</defs>",
        f'<rect width="{width}" height="{height}" fill="#08111f"/>',
        f'<text x="{margin}" y="32" fill="#e5f2ff" font-family="Segoe UI, sans-serif" font-size="22" font-weight="700">Plume Focus Visual Demo</text>',
        f'<text x="{margin}" y="58" fill="#a7bdd6" font-family="Segoe UI, sans-serif" font-size="13">Boundary prediction from reduced 2-D observations: last area + motion vector + uncertainty bound.</text>',
    ]

    panel_positions = [
        (margin, 86),
        (margin + panel_w + gap, 86),
        (margin, 86 + panel_h + title_h + gap),
        (margin + panel_w + gap, 86 + panel_h + title_h + gap),
    ]

    for panel, (x, y) in zip(panels, panel_positions):
        parts.append(_render_panel(panel, x, y, cell, panel_w, panel_h))

    summary_y = height - summary_h + 18
    parts.append(f'<rect x="{margin}" y="{summary_y - 20}" width="{width - margin * 2}" height="{summary_h - 22}" rx="18" fill="#0f1c2e" stroke="#26415f"/>')
    parts.append(f'<text x="{margin + 18}" y="{summary_y + 2}" fill="#d8ebff" font-family="Segoe UI, sans-serif" font-size="15" font-weight="700">Run Summary</text>')
    for index, line in enumerate(summary):
        parts.append(
            f'<text x="{margin + 18}" y="{summary_y + 28 + index * 16}" fill="#b7c9dc" '
            f'font-family="Consolas, monospace" font-size="12">{escape(line)}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def _render_panel(panel: _Panel, x: int, y: int, cell: int, panel_w: int, panel_h: int) -> str:
    parts = [
        f'<text x="{x}" y="{y - 12}" fill="#d8ebff" font-family="Segoe UI, sans-serif" font-size="14" font-weight="700">{escape(panel.title)}</text>',
        f'<g filter="url(#soft-shadow)">',
        f'<rect x="{x - 1}" y="{y - 1}" width="{panel_w + 2}" height="{panel_h + 2}" rx="14" fill="#0b1524" stroke="#23364f"/>',
        f'<g clip-path="inset(0 round 13px)">',
        _heatmap_rects(panel.frame, x, y, cell),
        "</g>",
    ]

    for label, box, color in panel.boxes:
        parts.append(_box_rect(box, x, y, cell, color))
        parts.append(
            f'<text x="{x + box.min_col * cell + 4}" y="{y + max(14, box.min_row * cell - 5)}" '
            f'fill="{color}" font-family="Segoe UI, sans-serif" font-size="11" font-weight="700">{escape(label)}</text>'
        )

    parts.extend(["</g>", _legend(panel.boxes, x, y + panel_h + 22)])
    return "\n".join(parts)


def _heatmap_rects(frame: NDArray[np.float64], x: int, y: int, cell: int) -> str:
    arr = np.asarray(frame, dtype=float)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    scale = 1.0 / max(hi - lo, 1e-12)

    rects: list[str] = []
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            value = (float(arr[row, col]) - lo) * scale
            color = _plume_color(value)
            rects.append(f'<rect x="{x + col * cell}" y="{y + row * cell}" width="{cell}" height="{cell}" fill="{color}"/>')
    return "\n".join(rects)


def _plume_color(value: float) -> str:
    """Map normalized intensity to a dark-blue-to-gold plume palette."""

    value = float(np.clip(value, 0.0, 1.0))
    stops = [
        (8, 17, 34),
        (18, 52, 86),
        (17, 100, 135),
        (74, 180, 146),
        (249, 209, 102),
        (255, 245, 200),
    ]
    position = value * (len(stops) - 1)
    low = int(np.floor(position))
    high = min(low + 1, len(stops) - 1)
    frac = position - low
    rgb = tuple(round(stops[low][i] * (1.0 - frac) + stops[high][i] * frac) for i in range(3))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _box_rect(box: BoundingBox, x: int, y: int, cell: int, color: str) -> str:
    rect_x = x + box.min_col * cell
    rect_y = y + box.min_row * cell
    width = box.width * cell
    height = box.height * cell
    return (
        f'<rect x="{rect_x}" y="{rect_y}" width="{width}" height="{height}" '
        f'fill="none" stroke="{color}" stroke-width="3" stroke-dasharray="8 5"/>'
    )


def _legend(boxes: list[tuple[str, BoundingBox, str]], x: int, y: int) -> str:
    if not boxes:
        return ""

    parts = []
    cursor = x
    for label, _box, color in boxes:
        parts.append(f'<rect x="{cursor}" y="{y - 10}" width="12" height="12" fill="none" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{cursor + 18}" y="{y}" fill="#a7bdd6" font-family="Segoe UI, sans-serif" font-size="11">{escape(label)}</text>')
        cursor += 18 + len(label) * 7 + 24
    return "\n".join(parts)
