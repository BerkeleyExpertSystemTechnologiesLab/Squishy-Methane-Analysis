import numpy as np

from plume_focus.predictor import BoundingBox
from plume_focus.scan_execution import (
    budget_column_sweep_mask,
    column_sweep_scan_moving_frames,
    raster_scan_moving_frames,
)
from plume_focus.video_sources import (
    SyntheticMethaneLeakConfig,
    SyntheticVideoConfig,
    generate_synthetic_methane_frames,
    generate_synthetic_methane_values,
    generate_synthetic_object_values,
    generate_synthetic_video_frames,
    methane_scan_roi,
)


def test_raster_scan_over_moving_frames_is_not_static_snapshot() -> None:
    config = SyntheticVideoConfig(frames=4, velocity=(2.0, 3.0))
    frames = generate_synthetic_video_frames(config)

    scan = raster_scan_moving_frames(
        frames,
        1,
        fps=config.fps,
        scan_rate_hz=60.0,
        sample_values_at_seconds=lambda times, rows, cols: generate_synthetic_object_values(
            config, times * config.fps, rows, cols
        ),
    )

    assert scan.values.shape == frames[1].shape
    assert not np.array_equal(scan.values, frames[1])
    assert np.isclose(scan.scan_duration_seconds, frames[1].size / 60.0)


def test_point_scan_stride_uses_footprints_not_every_pixel_visit() -> None:
    config = SyntheticVideoConfig(frames=4, shape=(12, 16), velocity=(2.0, 3.0))
    frames = generate_synthetic_video_frames(config)

    scan = raster_scan_moving_frames(
        frames,
        1,
        fps=config.fps,
        scan_rate_hz=20.0,
        sample_stride_px=4,
        sample_values_at_seconds=lambda times, rows, cols: generate_synthetic_object_values(
            config, times * config.fps, rows, cols
        ),
    )

    assert scan.sample_stride_px == 4
    assert scan.effective_sample_points == 12
    assert scan.sampled_pixels == frames[1].size
    assert np.isclose(scan.scan_duration_seconds, scan.effective_sample_points / 20.0)


def test_column_sweep_time_scales_by_columns_not_pixels() -> None:
    config = SyntheticVideoConfig(frames=4, shape=(10, 6), velocity=(0.0, 0.0))
    frames = generate_synthetic_video_frames(config)

    scan = column_sweep_scan_moving_frames(
        frames,
        1,
        fps=config.fps,
        vertical_pass_seconds=0.5,
        horizontal_step_seconds=0.1,
        column_stride_px=1,
        sample_values_at_seconds=lambda times, rows, cols: generate_synthetic_object_values(
            config, times * config.fps, rows, cols
        ),
    )

    assert scan.scan_mode == "column"
    assert scan.effective_sample_points == 6
    assert scan.sampled_pixels == frames[1].size
    assert np.isclose(scan.scan_duration_seconds, 6 * 0.5 + 5 * 0.1)


def test_column_sweep_can_keep_small_pixels_with_horizontal_stride() -> None:
    config = SyntheticVideoConfig(frames=4, shape=(10, 6), velocity=(0.0, 0.0))
    frames = generate_synthetic_video_frames(config)

    scan = column_sweep_scan_moving_frames(
        frames,
        1,
        fps=config.fps,
        vertical_pass_seconds=0.5,
        horizontal_step_seconds=0.1,
        column_stride_px=2,
        sample_values_at_seconds=lambda times, rows, cols: generate_synthetic_object_values(
            config, times * config.fps, rows, cols
        ),
    )

    assert scan.effective_sample_points == 3
    assert scan.sampled_pixels == 30
    assert np.isclose(scan.scan_duration_seconds, 3 * 0.5 + 2 * 0.1)


def test_column_sweep_vertical_time_scales_by_travel_length() -> None:
    config = SyntheticVideoConfig(frames=4, shape=(10, 6), velocity=(0.0, 0.0))
    frames = generate_synthetic_video_frames(config)

    scan = column_sweep_scan_moving_frames(
        frames,
        1,
        fps=config.fps,
        vertical_pass_seconds=0.5,
        horizontal_step_seconds=0.1,
        column_stride_px=1,
        sample_values_at_seconds=lambda times, rows, cols: generate_synthetic_object_values(
            config, times * config.fps, rows, cols
        ),
        roi_box=BoundingBox(2, 0, 6, 5),
    )

    assert scan.sampled_pixels == 30
    assert scan.vertical_reference_rows == 10
    assert np.isclose(scan.mean_vertical_travel_rows, 5.0)
    assert np.isclose(scan.mean_vertical_pass_seconds, 0.25)
    assert np.isclose(scan.scan_duration_seconds, 6 * 0.25 + 5 * 0.1)


def test_column_sweep_stride_is_anchored_to_image_grid_not_roi_edge() -> None:
    config = SyntheticVideoConfig(frames=4, shape=(10, 12), velocity=(0.0, 0.0))
    frames = generate_synthetic_video_frames(config)

    first = column_sweep_scan_moving_frames(
        frames,
        1,
        fps=config.fps,
        vertical_pass_seconds=0.5,
        horizontal_step_seconds=0.1,
        column_stride_px=3,
        sample_values_at_seconds=lambda times, rows, cols: generate_synthetic_object_values(
            config, times * config.fps, rows, cols
        ),
        roi_box=BoundingBox(0, 1, 9, 7),
    )
    shifted = column_sweep_scan_moving_frames(
        frames,
        1,
        fps=config.fps,
        vertical_pass_seconds=0.5,
        horizontal_step_seconds=0.1,
        column_stride_px=3,
        sample_values_at_seconds=lambda times, rows, cols: generate_synthetic_object_values(
            config, times * config.fps, rows, cols
        ),
        roi_box=BoundingBox(0, 2, 9, 8),
    )

    assert np.unique(np.where(first.sampled_mask)[1]).tolist() == [3, 6]
    assert np.unique(np.where(shifted.sampled_mask)[1]).tolist() == [3, 6]


def test_column_sweep_samples_the_complete_mechanical_row_band() -> None:
    config = SyntheticVideoConfig(frames=2, shape=(10, 4), velocity=(0.0, 0.0))
    frames = generate_synthetic_video_frames(config)
    requested = np.zeros(config.shape, dtype=bool)
    requested[1, 1] = True
    requested[4, 1] = True

    scan = column_sweep_scan_moving_frames(
        frames,
        0,
        fps=config.fps,
        vertical_pass_seconds=0.5,
        column_stride_px=1,
        sample_values_at_seconds=lambda times, rows, cols: generate_synthetic_object_values(
            config, times * config.fps, rows, cols
        ),
        roi_box=BoundingBox(0, 0, 9, 3),
        roi_mask=requested,
    )

    assert np.where(scan.requested_mask[:, 1])[0].tolist() == [1, 4]
    assert np.where(scan.mechanical_sweep_mask[:, 1])[0].tolist() == [1, 2, 3, 4]
    assert np.isclose(scan.scan_duration_seconds, 0.2)


def test_budget_column_sweep_mask_keeps_high_belief_columns() -> None:
    base_mask = np.ones((6, 10), dtype=bool)
    belief = np.zeros((6, 10), dtype=float)
    belief[:, 7] = 10.0
    belief[:, 8] = 6.0
    belief[:, 1] = 1.0

    budgeted = budget_column_sweep_mask(
        base_mask,
        belief,
        mass_fraction=0.8,
        min_columns=2,
        max_columns=3,
        include_col=1,
    )

    assert np.unique(np.where(budgeted)[1]).tolist() == [1, 7, 8]


def test_budget_column_sweep_mask_can_trim_vertical_rows() -> None:
    base_mask = np.ones((6, 5), dtype=bool)
    belief = np.zeros((6, 5), dtype=float)
    belief[4, :] = 10.0
    belief[3, :] = 5.0
    belief[2, :] = 1.0

    budgeted = budget_column_sweep_mask(
        base_mask,
        belief,
        mass_fraction=1.0,
        min_columns=1,
        max_columns=None,
        row_mass_fraction=0.8,
        min_rows=2,
        include_row=1,
    )

    assert np.unique(np.where(budgeted)[0]).tolist() == [1, 2, 3, 4]
    assert np.unique(np.where(budgeted)[1]).tolist() == [0, 1, 2, 3, 4]


def test_methane_sampling_handles_long_scan_times_without_divide_by_zero() -> None:
    config = SyntheticMethaneLeakConfig(frames=10)
    frame_times = np.array([0.0, 120.0, 720.0, 2800.0])
    rows = np.array([120, 121, 122, 123])
    cols = np.array([48, 90, 140, 200])

    with np.errstate(divide="raise", invalid="raise"):
        values = generate_synthetic_methane_values(config, frame_times, rows, cols)

    assert values.shape == frame_times.shape
    assert np.all(np.isfinite(values))


def test_methane_roi_scan_is_smaller_than_full_frame() -> None:
    config = SyntheticMethaneLeakConfig(frames=10)
    frames = generate_synthetic_methane_frames(config)
    roi = methane_scan_roi(config)

    scan = raster_scan_moving_frames(
        frames,
        2,
        fps=config.fps,
        scan_rate_hz=config.scan_rate_hz,
        sample_values_at_seconds=lambda times, rows, cols: generate_synthetic_methane_values(
            config, times * config.fps, rows, cols
        ),
        roi_box=roi,
    )

    assert scan.sampled_pixels == roi.area
    assert scan.sampled_area_fraction < 1.0
    assert scan.scan_duration_seconds < frames[2].size / config.scan_rate_hz
