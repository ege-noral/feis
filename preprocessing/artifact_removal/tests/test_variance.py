import numpy as np

from preprocessing.artifact_removal.features.variance import compute_variance


def test_variance():
    # Mock EEG signal
    signal = np.array([-2, 1, 3, 7, -5], dtype=float)

    result = compute_variance(signal)

    # Mean = 0.8
    # Variance = 16.96
    expected = 16.96

    assert np.isclose(result, expected)