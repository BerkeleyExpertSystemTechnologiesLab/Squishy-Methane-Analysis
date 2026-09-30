import numpy as np

from plume_focus.experiment_metrics import foreground_mask
from plume_focus.video_sources import (
    SyntheticMethaneLeakConfig,
    SyntheticPlumeVideoConfig,
    generate_synthetic_methane_frame,
    generate_synthetic_methane_frames,
    generate_synthetic_methane_values,
    generate_synthetic_plume_frames,
)


def test_plume_frames_are_diffuse_and_advect_rightward() -> None:
    config = SyntheticPlumeVideoConfig(frames=8)
    frames = generate_synthetic_plume_frames(config)

    early_mask = foreground_mask(frames[1], threshold_ratio=0.24)
    later_mask = foreground_mask(frames[5], threshold_ratio=0.24)
    early_cols = np.where(early_mask)[1]
    later_cols = np.where(later_mask)[1]

    assert later_cols.mean() > early_cols.mean()
    assert later_mask.sum() > 50


def test_methane_leak_keeps_source_fixed_and_advects_downwind() -> None:
    config = SyntheticMethaneLeakConfig(frames=10)
    frames = generate_synthetic_methane_frames(config)

    source_row, source_col = map(round, config.source)
    source_values = [frame[source_row, source_col] for frame in frames[:5]]
    later_mask = foreground_mask(frames[7], threshold_ratio=0.22)
    later_cols = np.where(later_mask)[1]

    assert min(source_values) > 70
    assert later_cols.mean() > source_col + 15


def test_methane_leak_supports_tilted_wind_vector() -> None:
    config = SyntheticMethaneLeakConfig(frames=12, wind_velocity=(-0.55, 1.0))
    frames = generate_synthetic_methane_frames(config)

    source_row, source_col = config.source
    later_mask = foreground_mask(frames[10], threshold_ratio=0.22)
    later_rows, later_cols = np.where(later_mask)

    assert later_cols.mean() > source_col + 15
    assert later_rows.mean() < source_row - 3


def test_randomized_methane_phases_preserve_frame_sample_equivalence() -> None:
    config = SyntheticMethaneLeakConfig(
        lateral_phase=0.7,
        amplitude_phase=-1.1,
        shimmer_phase=2.2,
    )
    time = 17.25
    rows = np.asarray([10, 80, 120, 205], dtype=int)
    cols = np.asarray([5, 75, 190, 310], dtype=int)

    frame = generate_synthetic_methane_frame(config, time)
    values = generate_synthetic_methane_values(
        config,
        np.full(rows.shape, time),
        rows,
        cols,
    )

    assert np.array_equal(frame[rows, cols], values.astype(np.uint8))


def test_cached_methane_truth_frames_are_returned_as_independent_arrays() -> None:
    config = SyntheticMethaneLeakConfig()
    first = generate_synthetic_methane_frame(config, 4.5)
    expected = int(first[0, 0])

    first[0, 0] = (expected + 1) % 255
    second = generate_synthetic_methane_frame(config, 4.5)

    assert int(second[0, 0]) == expected
