import numpy as np

from plume_focus.scanner import SinglePixelScanner


def test_single_pixel_scanner_records_temporal_age() -> None:
    scanner = SinglePixelScanner((2, 3), dwell_time=0.1, stabilization_time=0.0)

    result = scanner.scan(lambda t: np.full((2, 3), t))

    assert np.isclose(result.scan_duration, 0.6)
    assert np.isclose(result.max_age, 0.6)
    assert result.values[0, 0] == 0.0
    assert np.isclose(result.values[-1, -1], 0.5)


def test_snake_order_reverses_every_other_row() -> None:
    scanner = SinglePixelScanner((2, 3), dwell_time=1.0, order="snake")

    result = scanner.scan(lambda t: np.full((2, 3), t))

    assert result.sampled_at[0].tolist() == [0.0, 1.0, 2.0]
    assert result.sampled_at[1].tolist() == [5.0, 4.0, 3.0]
