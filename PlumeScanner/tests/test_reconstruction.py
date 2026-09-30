import numpy as np

from plume_focus.reconstruction import grid_sample_points, reconstruct_idw, sample_frame
from plume_focus.synthetic import gaussian_plume


def test_grid_sampling_reads_expected_values() -> None:
    frame = np.arange(16).reshape(4, 4)
    points = grid_sample_points(frame.shape, spacing=2)

    values = sample_frame(frame, points)

    assert points.tolist() == [[0, 0], [0, 2], [2, 0], [2, 2]]
    assert values.tolist() == [0, 2, 8, 10]


def test_idw_reconstruction_preserves_sample_points() -> None:
    frame = gaussian_plume((12, 12), center=(6.0, 6.0))
    points = grid_sample_points(frame.shape, spacing=3)
    values = sample_frame(frame, points)

    reconstructed = reconstruct_idw(points, values, frame.shape)
    reconstructed_values = sample_frame(reconstructed, points)

    assert np.allclose(reconstructed_values, values)
