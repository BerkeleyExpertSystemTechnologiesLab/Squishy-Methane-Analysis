import numpy as np
import pytest

from plume_focus.experiment import (
    ArrayVideoTimeline,
    ScanEvent,
    TimestampedArrayTimeline,
    TimelineExhausted,
    distribution_centroid_motion,
    evaluate_scan_mask,
)


def test_finite_video_timeline_never_clamps_future_samples() -> None:
    frames = tuple(np.full((3, 4), index, dtype=np.uint8) for index in range(3))
    timeline = ArrayVideoTimeline(frames, fps=2.0)

    assert timeline.frame_at_seconds(1.0)[0, 0] == 2
    with pytest.raises(TimelineExhausted):
        timeline.samples_at_seconds(
            np.asarray([1.01]),
            np.asarray([0]),
            np.asarray([0]),
        )


def test_timestamped_timeline_uses_latest_completed_capture() -> None:
    frames = tuple(
        np.full((2, 3), value, dtype=np.uint8)
        for value in (10, 20, 30)
    )
    timeline = TimestampedArrayTimeline(
        frames,
        timestamps_seconds=(0.0, 0.4, 1.1),
        fps=2.0,
    )

    assert timeline.frame_at_seconds(0.39)[0, 0] == 10
    assert timeline.frame_at_seconds(0.4)[0, 0] == 20
    assert timeline.samples_at_seconds(
        np.asarray([0.8, 1.1]),
        np.asarray([0, 1]),
        np.asarray([1, 2]),
    ).tolist() == [20.0, 30.0]

    with pytest.raises(TimelineExhausted):
        timeline.frame_at_seconds(1.100001)


def test_scan_events_reject_overlap_and_future_evidence() -> None:
    first = ScanEvent(0, 0.0, 2.0, 2.0, 4.0)
    first.validate()

    with pytest.raises(ValueError, match="overlap"):
        ScanEvent(1, 1.5, 3.0, 3.0, 5.0).validate(first)
    with pytest.raises(ValueError, match="before it completes"):
        ScanEvent(1, 2.0, 3.0, 2.5, 4.0).validate(first)


def test_scan_mask_metrics_use_probability_mass_and_actual_mask() -> None:
    truth = np.zeros((4, 5), dtype=float)
    truth[1:3, 2:4] = np.asarray([[1.0, 3.0], [1.0, 5.0]])
    support = truth > 0
    scan_mask = np.zeros_like(support)
    scan_mask[:, 3] = True

    metrics = evaluate_scan_mask(scan_mask, truth, support)

    assert metrics.support_recall == 0.5
    assert metrics.probability_mass_recall == 0.8
    assert metrics.mask_area_fraction == 0.2


def test_motion_is_estimated_from_completed_scan_distributions() -> None:
    previous = np.zeros((10, 12), dtype=float)
    current = np.zeros_like(previous)
    previous[4, 3] = 1.0
    current[6, 7] = 1.0

    motion = distribution_centroid_motion(previous, current, elapsed_seconds=2.0)

    assert motion.dy_per_t == 1.0
    assert motion.dx_per_t == 2.0
    assert motion.method == "scan_posterior_centroid_px_per_second"
