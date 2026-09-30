import numpy as np
import pytest

from plume_focus.experiment import TimestampedArrayTimeline, TimelineExhausted
from plume_focus.sensor_emulation import (
    FixedLeakOffCalibration,
    SensorEmulationProfile,
    SensorEmulationTimeline,
    emulate_sensor_frame,
    fit_fixed_leak_off_calibration,
)


def _profile(
    residual_mode: str = "two_sided_magnitude",
    *,
    smoothing_time_constant_seconds: float | None = None,
) -> SensorEmulationProfile:
    return SensorEmulationProfile(
        profile_id="test_sensor_emulation",
        residual_mode=residual_mode,  # type: ignore[arg-type]
        provenance="test fixture",
        noise_floor_raw_units=1.0,
        output_units_per_noise_sigma=10.0,
        smoothing_time_constant_seconds=smoothing_time_constant_seconds,
    )


def test_fixed_background_removes_scene_structure_without_global_scaling() -> None:
    background = np.asarray(
        [[20, 80, 140], [35, 95, 200]],
        dtype=np.uint8,
    )
    leak_off = np.stack([background, background, background])
    calibration = fit_fixed_leak_off_calibration(
        leak_off,
        profile=_profile(),
    )
    observed = background.copy()
    observed[0, 1] += 5

    signal = emulate_sensor_frame(
        observed,
        calibration=calibration,
        profile=_profile(),
    )

    expected = np.zeros(background.shape, dtype=float)
    expected[0, 1] = 50.0
    np.testing.assert_array_equal(signal, expected)


def test_per_pixel_mad_downweights_a_noisy_pixel() -> None:
    leak_off = np.asarray(
        [
            [[100, 90]],
            [[100, 95]],
            [[100, 100]],
            [[100, 105]],
            [[100, 110]],
        ],
        dtype=np.uint8,
    )
    profile = _profile()
    calibration = fit_fixed_leak_off_calibration(
        leak_off,
        profile=profile,
    )

    signal = emulate_sensor_frame(
        np.asarray([[110, 110]], dtype=np.uint8),
        calibration=calibration,
        profile=profile,
    )

    assert calibration.noise_sigma_raw_units[0, 1] > (
        calibration.noise_sigma_raw_units[0, 0]
    )
    assert signal[0, 0] > signal[0, 1]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("bright", [[100.0, 0.0]]),
        ("dark", [[0.0, 100.0]]),
        ("two_sided_magnitude", [[100.0, 100.0]]),
    ],
)
def test_residual_modes_are_explicit_and_nonnegative(
    mode: str,
    expected: list[list[float]],
) -> None:
    calibration = FixedLeakOffCalibration(
        background_raw_units=np.asarray([[100.0, 100.0]]),
        noise_sigma_raw_units=np.asarray([[1.0, 1.0]]),
        sample_count=3,
    )

    signal = emulate_sensor_frame(
        np.asarray([[110, 90]], dtype=np.uint8),
        calibration=calibration,
        profile=_profile(mode),
    )

    np.testing.assert_array_equal(signal, expected)


def test_timeline_samples_only_requested_coordinates() -> None:
    frames = (
        np.asarray([[10, 20], [30, 40]], dtype=np.uint8),
        np.asarray([[11, 22], [33, 44]], dtype=np.uint8),
    )
    source = TimestampedArrayTimeline(
        frames=frames,
        timestamps_seconds=(0.0, 1.0),
        fps=1.0,
    )
    calibration = FixedLeakOffCalibration(
        background_raw_units=np.asarray([[10, 20], [30, 40]], dtype=float),
        noise_sigma_raw_units=np.ones((2, 2), dtype=float),
        sample_count=2,
    )
    timeline = SensorEmulationTimeline(source, calibration, _profile("bright"))

    values = timeline.samples_at_seconds(
        np.asarray([1.0, 1.0]),
        np.asarray([0, 1]),
        np.asarray([1, 0]),
    )

    assert values.tolist() == [20.0, 30.0]
    assert values.shape == (2,)


def test_requested_samples_ignore_unrequested_and_future_values() -> None:
    calibration = FixedLeakOffCalibration(
        background_raw_units=np.zeros((2, 2), dtype=float),
        noise_sigma_raw_units=np.ones((2, 2), dtype=float),
        sample_count=2,
    )
    profile = SensorEmulationProfile(
        profile_id="causal_sampling_test",
        residual_mode="bright",
        provenance="test fixture",
        output_units_per_noise_sigma=1.0,
    )

    def make_timeline(
        *,
        unrequested_value: int,
        future_value: int,
    ) -> SensorEmulationTimeline:
        source = TimestampedArrayTimeline(
            frames=(
                np.asarray(
                    [[0, 0], [0, unrequested_value]],
                    dtype=np.uint8,
                ),
                np.asarray(
                    [[100, 0], [0, unrequested_value]],
                    dtype=np.uint8,
                ),
                np.asarray(
                    [[future_value, 0], [0, unrequested_value]],
                    dtype=np.uint8,
                ),
            ),
            timestamps_seconds=(0.0, 1.0, 2.0),
            fps=1.0,
        )
        return SensorEmulationTimeline(source, calibration, profile)

    low_unrequested_and_future = make_timeline(
        unrequested_value=0,
        future_value=0,
    )
    high_unrequested_and_future = make_timeline(
        unrequested_value=255,
        future_value=255,
    )
    requested = (
        np.asarray([0.5, 1.5]),
        np.asarray([0, 0]),
        np.asarray([0, 0]),
    )

    low_value = low_unrequested_and_future.samples_at_seconds(*requested)
    high_value = high_unrequested_and_future.samples_at_seconds(*requested)

    np.testing.assert_allclose(low_value, high_value)
    np.testing.assert_allclose(low_value, [0.0, 100.0])
    with pytest.raises(TimelineExhausted):
        low_unrequested_and_future.samples_at_seconds(
            np.asarray([2.000001]),
            np.asarray([0]),
            np.asarray([0]),
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"profile_id": ""},
        {"residual_mode": "automatic"},
        {"provenance": ""},
        {"noise_floor_raw_units": 0.0},
        {"output_units_per_noise_sigma": float("nan")},
        {"smoothing_time_constant_seconds": 1.0},
        {"profile_schema_version": 0},
        {"profile_schema_version": 1.5},
        {"profile_schema_version": 2},
    ],
)
def test_profile_rejects_invalid_behavior_values(
    changes: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "profile_id": "valid",
        "residual_mode": "bright",
        "provenance": "declared test",
    }
    values.update(changes)

    with pytest.raises(ValueError):
        SensorEmulationProfile(**values)  # type: ignore[arg-type]


def test_calibration_and_emulation_do_not_mutate_inputs() -> None:
    leak_off = np.asarray(
        [
            [[9, 19], [29, 39]],
            [[10, 20], [30, 40]],
            [[11, 21], [31, 41]],
        ],
        dtype=np.uint8,
    )
    original_leak_off = leak_off.copy()
    calibration = fit_fixed_leak_off_calibration(
        leak_off,
        profile=_profile(),
    )
    observed = np.asarray([[12, 18], [35, 35]], dtype=np.uint8)
    original_observed = observed.copy()

    emulate_sensor_frame(
        observed,
        calibration=calibration,
        profile=_profile(),
    )

    np.testing.assert_array_equal(leak_off, original_leak_off)
    np.testing.assert_array_equal(observed, original_observed)
    assert not calibration.background_raw_units.flags.writeable
    assert not calibration.noise_sigma_raw_units.flags.writeable
    with pytest.raises(ValueError):
        calibration.background_raw_units[0, 0] = 0.0
