"""Mechanical point and column scan execution for time-addressable sources."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from plume_focus.experiment import TimelineExhausted
from plume_focus.experiment_contracts import (
    DEFAULT_SCAN_RATE_HZ,
    RasterVideoScan,
)
from plume_focus.predictor import BoundingBox
from plume_focus.scanner import SinglePixelScanner


SampleValuesAt = Callable[
    [NDArray[np.float64], NDArray[np.int_], NDArray[np.int_]],
    NDArray[np.float64],
]


def raster_scan_moving_frames(
    frames: list[NDArray[np.uint8]],
    start_index: int,
    *,
    fps: float,
    scan_rate_hz: float = DEFAULT_SCAN_RATE_HZ,
    sample_stride_px: int = 1,
    order: str = "snake",
    sample_values_at: SampleValuesAt | None = None,
    sample_values_at_seconds: SampleValuesAt | None = None,
    started_at_seconds: float | None = None,
    roi_box: BoundingBox | None = None,
    roi_mask: NDArray[np.bool_] | None = None,
) -> RasterVideoScan:
    """Assemble one frame by sequentially scanning points while the field moves."""

    if scan_rate_hz <= 0:
        raise ValueError("scan_rate_hz must be positive.")
    if fps <= 0:
        raise ValueError("fps must be positive.")
    if sample_stride_px <= 0:
        raise ValueError("sample_stride_px must be positive.")
    if sample_values_at is not None and sample_values_at_seconds is not None:
        raise ValueError(
            "Provide only one of sample_values_at or sample_values_at_seconds."
        )

    shape = frames[start_index].shape
    if roi_mask is not None and roi_mask.shape != shape:
        raise ValueError("roi_mask must match frame shape.")

    sample_stride_px = int(sample_stride_px)
    scan_shape = shape if roi_box is None else (roi_box.height, roi_box.width)
    coarse_shape = (
        int(np.ceil(scan_shape[0] / sample_stride_px)),
        int(np.ceil(scan_shape[1] / sample_stride_px)),
    )
    coarse_positions = np.asarray(
        list(
            SinglePixelScanner(
                coarse_shape,
                dwell_time=1.0,
                order=order,
            )._positions()
        ),
        dtype=int,
    )
    local_positions = (
        coarse_positions * sample_stride_px + sample_stride_px // 2
    )
    local_positions[:, 0] = np.clip(
        local_positions[:, 0],
        0,
        scan_shape[0] - 1,
    )
    local_positions[:, 1] = np.clip(
        local_positions[:, 1],
        0,
        scan_shape[1] - 1,
    )
    if roi_box is None:
        positions = local_positions
    else:
        positions = local_positions + np.array(
            [roi_box.min_row, roi_box.min_col],
            dtype=int,
        )
    if roi_mask is not None:
        keep = roi_mask[positions[:, 0], positions[:, 1]]
        positions = positions[keep]
    rows = positions[:, 0]
    cols = positions[:, 1]
    sample_offsets_seconds = (
        np.arange(positions.shape[0], dtype=float) / scan_rate_hz
    )
    started_at_seconds = (
        start_index / fps
        if started_at_seconds is None
        else float(started_at_seconds)
    )
    sampled_at_flat = started_at_seconds + sample_offsets_seconds
    frame_times = sampled_at_flat * fps

    if sample_values_at_seconds is not None:
        flat_values = sample_values_at_seconds(sampled_at_flat, rows, cols)
    elif sample_values_at is None:
        source_indices = np.rint(frame_times).astype(int)
        if np.any(source_indices < 0) or np.any(
            source_indices >= len(frames)
        ):
            raise TimelineExhausted(
                "Point scan requests samples outside the available frame timeline."
            )
        flat_values = np.zeros(positions.shape[0], dtype=float)
        for source_index in np.unique(source_indices):
            source_mask = source_indices == source_index
            flat_values[source_mask] = frames[int(source_index)][
                rows[source_mask],
                cols[source_mask],
            ]
    else:
        flat_values = sample_values_at(frame_times, rows, cols)

    values = np.zeros(shape, dtype=np.uint8)
    sampled_at = np.full(shape, np.nan, dtype=float)
    sampled_mask = np.zeros(shape, dtype=bool)
    half_low = sample_stride_px // 2
    half_high = sample_stride_px - half_low
    for row, col, value, sampled_time in zip(
        rows,
        cols,
        flat_values,
        sampled_at_flat,
    ):
        min_row = max(0, int(row) - half_low)
        max_row = min(shape[0], int(row) + half_high)
        min_col = max(0, int(col) - half_low)
        max_col = min(shape[1], int(col) + half_high)
        footprint = np.ones(
            (max_row - min_row, max_col - min_col),
            dtype=bool,
        )
        if roi_box is not None:
            row_grid, col_grid = np.indices(footprint.shape)
            footprint &= (
                (row_grid + min_row >= roi_box.min_row)
                & (row_grid + min_row <= roi_box.max_row)
                & (col_grid + min_col >= roi_box.min_col)
                & (col_grid + min_col <= roi_box.max_col)
            )
        if roi_mask is not None:
            footprint &= roi_mask[min_row:max_row, min_col:max_col]
        value_block = values[min_row:max_row, min_col:max_col]
        sampled_at_block = sampled_at[min_row:max_row, min_col:max_col]
        sampled_mask_block = sampled_mask[min_row:max_row, min_col:max_col]
        value_block[footprint] = np.clip(value, 0, 255).astype(np.uint8)
        sampled_at_block[footprint] = sampled_time
        sampled_mask_block[footprint] = True
    completed_at_seconds = (
        started_at_seconds + positions.shape[0] / scan_rate_hz
    )
    requested_mask = np.zeros(shape, dtype=bool)
    if roi_mask is not None:
        requested_mask |= roi_mask
    elif roi_box is not None:
        requested_mask[roi_box.to_slices()] = True
    else:
        requested_mask[:] = True

    return RasterVideoScan(
        values=values,
        sampled_at_seconds=sampled_at,
        scan_rate_hz=scan_rate_hz,
        started_at_seconds=started_at_seconds,
        completed_at_seconds=completed_at_seconds,
        sampled_mask=sampled_mask,
        requested_mask=requested_mask,
        mechanical_sweep_mask=sampled_mask.copy(),
        roi_box=roi_box,
        sample_points=int(positions.shape[0]),
        sample_stride_px=sample_stride_px,
        scan_mode="point",
    )


def column_sweep_scan_moving_frames(
    frames: list[NDArray[np.uint8]],
    start_index: int,
    *,
    fps: float,
    vertical_pass_seconds: float,
    horizontal_step_seconds: float = 0.0,
    column_stride_px: int = 1,
    sample_values_at: SampleValuesAt | None = None,
    sample_values_at_seconds: SampleValuesAt | None = None,
    started_at_seconds: float | None = None,
    roi_box: BoundingBox | None = None,
    roi_mask: NDArray[np.bool_] | None = None,
) -> RasterVideoScan:
    """Scan columns with vertical sweeps and horizontal positioning steps."""

    if fps <= 0:
        raise ValueError("fps must be positive.")
    if vertical_pass_seconds <= 0:
        raise ValueError("vertical_pass_seconds must be positive.")
    if horizontal_step_seconds < 0:
        raise ValueError("horizontal_step_seconds must be non-negative.")
    if column_stride_px <= 0:
        raise ValueError("column_stride_px must be positive.")
    if sample_values_at is not None and sample_values_at_seconds is not None:
        raise ValueError(
            "Provide only one of sample_values_at or sample_values_at_seconds."
        )

    shape = frames[start_index].shape
    if roi_mask is not None and roi_mask.shape != shape:
        raise ValueError("roi_mask must match frame shape.")

    roi = (
        roi_box
        if roi_box is not None
        else BoundingBox(0, 0, shape[0] - 1, shape[1] - 1)
    )
    roi = roi.clipped(shape)
    global_columns = np.arange(
        0,
        shape[1],
        int(column_stride_px),
        dtype=int,
    )
    columns = global_columns[
        (global_columns >= roi.min_col)
        & (global_columns <= roi.max_col)
    ]
    if columns.size == 0:
        center_col = int(round((roi.min_col + roi.max_col) / 2.0))
        columns = np.asarray(
            [int(np.clip(center_col, roi.min_col, roi.max_col))],
            dtype=int,
        )
    if roi_mask is not None:
        columns = np.asarray(
            [
                col
                for col in columns
                if np.any(
                    roi_mask[roi.min_row : roi.max_row + 1, col]
                )
            ],
            dtype=int,
        )

    values = np.zeros(shape, dtype=np.uint8)
    sampled_at = np.full(shape, np.nan, dtype=float)
    sampled_mask = np.zeros(shape, dtype=bool)
    started_at_seconds = (
        start_index / fps
        if started_at_seconds is None
        else float(started_at_seconds)
    )
    current_time = started_at_seconds
    columns_scanned = 0
    vertical_reference_rows = max(1, shape[0])
    vertical_travel_rows: list[int] = []
    vertical_pass_durations: list[float] = []
    row_chunks: list[NDArray[np.int_]] = []
    col_chunks: list[NDArray[np.int_]] = []
    time_chunks: list[NDArray[np.float64]] = []

    for column_index, col in enumerate(columns):
        if roi_mask is None:
            record_rows = np.arange(
                roi.min_row,
                roi.max_row + 1,
                dtype=int,
            )
            segment_min = roi.min_row
            segment_max = roi.max_row
        else:
            active_rows = (
                np.where(
                    roi_mask[roi.min_row : roi.max_row + 1, col]
                )[0]
                + roi.min_row
            )
            if active_rows.size == 0:
                continue
            segment_min = int(active_rows.min())
            segment_max = int(active_rows.max())
            record_rows = np.arange(
                segment_min,
                segment_max + 1,
                dtype=int,
            )

        travel_rows = segment_max - segment_min + 1
        vertical_duration = (
            vertical_pass_seconds
            * travel_rows
            / vertical_reference_rows
        )
        if column_index % 2 == 1:
            ordered_rows = record_rows[::-1]
            sweep_offsets = (
                segment_max - ordered_rows
            ) / max(1, segment_max - segment_min)
        else:
            ordered_rows = record_rows
            sweep_offsets = (
                ordered_rows - segment_min
            ) / max(1, segment_max - segment_min)

        sample_times = (
            current_time
            + sweep_offsets.astype(float) * vertical_duration
        )
        row_chunks.append(ordered_rows.astype(int))
        col_chunks.append(
            np.full(ordered_rows.shape, col, dtype=int)
        )
        time_chunks.append(sample_times.astype(float))
        vertical_travel_rows.append(int(travel_rows))
        vertical_pass_durations.append(float(vertical_duration))
        columns_scanned += 1
        current_time += vertical_duration
        if columns_scanned < columns.size:
            current_time += horizontal_step_seconds

    if row_chunks:
        rows = np.concatenate(row_chunks)
        cols = np.concatenate(col_chunks)
        sample_times = np.concatenate(time_chunks)
        frame_times = sample_times * fps
        if sample_values_at_seconds is not None:
            sample_values = sample_values_at_seconds(
                sample_times,
                rows,
                cols,
            )
        elif sample_values_at is None:
            source_indices = np.rint(frame_times).astype(int)
            if np.any(source_indices < 0) or np.any(
                source_indices >= len(frames)
            ):
                raise TimelineExhausted(
                    "Column scan requests samples outside the available frame timeline."
                )
            sample_values = np.zeros(rows.shape[0], dtype=float)
            for source_index in np.unique(source_indices):
                source_mask = source_indices == source_index
                sample_values[source_mask] = frames[int(source_index)][
                    rows[source_mask],
                    cols[source_mask],
                ]
        else:
            sample_values = sample_values_at(frame_times, rows, cols)

        values[rows, cols] = np.clip(
            sample_values,
            0,
            255,
        ).astype(np.uint8)
        sampled_at[rows, cols] = sample_times
        sampled_mask[rows, cols] = True

    requested_mask = np.zeros(shape, dtype=bool)
    if roi_mask is not None:
        requested_mask |= roi_mask
    elif roi_box is not None:
        requested_mask[roi.to_slices()] = True
    else:
        requested_mask[:] = True

    return RasterVideoScan(
        values=values,
        sampled_at_seconds=sampled_at,
        scan_rate_hz=0.0,
        started_at_seconds=started_at_seconds,
        completed_at_seconds=current_time,
        sampled_mask=sampled_mask,
        requested_mask=requested_mask,
        mechanical_sweep_mask=sampled_mask.copy(),
        roi_box=roi_box,
        sample_points=int(columns_scanned),
        sample_stride_px=int(column_stride_px),
        scan_mode="column",
        vertical_pass_seconds=float(vertical_pass_seconds),
        horizontal_step_seconds=float(horizontal_step_seconds),
        vertical_reference_rows=vertical_reference_rows,
        mean_vertical_travel_rows=(
            float(np.mean(vertical_travel_rows))
            if vertical_travel_rows
            else 0.0
        ),
        mean_vertical_pass_seconds=(
            float(np.mean(vertical_pass_durations))
            if vertical_pass_durations
            else 0.0
        ),
    )


def budget_column_sweep_mask(
    base_mask: NDArray[np.bool_],
    belief: NDArray[np.float64],
    *,
    mass_fraction: float,
    min_columns: int,
    max_columns: int | None,
    include_col: int | None = None,
    row_mass_fraction: float = 1.0,
    min_rows: int | None = None,
    include_row: int | None = None,
) -> NDArray[np.bool_]:
    """Keep the most informative sweep axes under column and row budgets."""

    if base_mask.shape != belief.shape:
        raise ValueError("base_mask and belief must have matching shapes.")
    if not 0.0 < mass_fraction <= 1.0:
        raise ValueError("mass_fraction must be in (0, 1].")
    if not 0.0 < row_mass_fraction <= 1.0:
        raise ValueError("row_mass_fraction must be in (0, 1].")
    if min_columns <= 0:
        raise ValueError("min_columns must be positive.")
    if min_rows is not None and min_rows <= 0:
        raise ValueError("min_rows must be positive when provided.")
    if max_columns is not None and max_columns <= 0:
        raise ValueError("max_columns must be positive when provided.")

    active_columns = np.where(np.any(base_mask, axis=0))[0]
    if active_columns.size == 0:
        return base_mask.copy()

    column_scores = np.where(base_mask, belief, 0.0).sum(axis=0)[
        active_columns
    ]
    order = np.argsort(column_scores)[::-1]
    sorted_scores = column_scores[order]
    total_score = float(sorted_scores.sum())
    if total_score > 0:
        cumulative = np.cumsum(sorted_scores) / total_score
        selected_count = (
            int(
                np.searchsorted(
                    cumulative,
                    mass_fraction,
                    side="left",
                )
            )
            + 1
        )
    else:
        selected_count = active_columns.size

    selected_count = max(min_columns, selected_count)
    if max_columns is not None:
        selected_count = min(max_columns, selected_count)
    selected_count = min(active_columns.size, selected_count)

    selected_columns = set(
        active_columns[order[:selected_count]].tolist()
    )
    if include_col is not None:
        selected_columns.add(
            int(np.clip(include_col, 0, base_mask.shape[1] - 1))
        )

    budgeted = np.zeros_like(base_mask)
    columns = np.asarray(sorted(selected_columns), dtype=int)
    budgeted[:, columns] = base_mask[:, columns]

    if (
        row_mass_fraction < 1.0
        or min_rows is not None
        or include_row is not None
    ):
        active_rows = np.where(np.any(budgeted, axis=1))[0]
        if active_rows.size == 0:
            return budgeted

        row_scores = np.where(budgeted, belief, 0.0).sum(axis=1)
        min_row, max_row = _smallest_contiguous_mass_band(
            row_scores,
            mass_fraction=row_mass_fraction,
            minimum_rows=min_rows or 1,
            include_row=include_row,
        )
        trimmed = np.zeros_like(base_mask)
        trimmed[min_row : max_row + 1, :] = budgeted[
            min_row : max_row + 1,
            :,
        ]
        budgeted = trimmed

    return budgeted


def _smallest_contiguous_mass_band(
    row_scores: NDArray[np.float64],
    *,
    mass_fraction: float,
    minimum_rows: int,
    include_row: int | None,
) -> tuple[int, int]:
    """Return the shortest contiguous row band satisfying the mass budget."""

    scores = np.maximum(np.asarray(row_scores, dtype=float), 0.0)
    row_count = scores.size
    if row_count == 0:
        raise ValueError("row_scores must not be empty.")

    total = float(scores.sum())
    if total <= 0.0:
        peak = row_count // 2
        best_start = peak
        best_end = peak
    else:
        target = mass_fraction * total
        best_start = 0
        best_end = row_count - 1
        running = 0.0
        left = 0
        for right, score in enumerate(scores):
            running += float(score)
            while (
                left <= right
                and running - float(scores[left]) >= target
            ):
                running -= float(scores[left])
                left += 1
            if (
                running >= target
                and right - left < best_end - best_start
            ):
                best_start, best_end = left, right

    desired_rows = min(row_count, max(1, int(minimum_rows)))
    while best_end - best_start + 1 < desired_rows:
        if best_start > 0:
            best_start -= 1
        if (
            best_end - best_start + 1 < desired_rows
            and best_end < row_count - 1
        ):
            best_end += 1

    if include_row is not None:
        included = int(np.clip(include_row, 0, row_count - 1))
        best_start = min(best_start, included)
        best_end = max(best_end, included)

    return best_start, best_end
