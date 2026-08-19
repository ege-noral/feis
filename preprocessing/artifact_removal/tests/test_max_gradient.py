import numpy as np

from preprocessing.artifact_removal.features.max_gradient import compute_max_gradient


def test_max_gradient_detects_sudden_change():
    signal = np.array([
        1, 2, 3, 4, 20, 21
    ], dtype=float)

    result = compute_max_gradient(signal)

    expected = 16.0

    assert np.isclose(result, expected)