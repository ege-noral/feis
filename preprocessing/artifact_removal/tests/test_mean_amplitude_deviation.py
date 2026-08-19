import numpy as np

from preprocessing.artifact_removal.features.mean_amplitude_deviation import (
    compute_mean_amplitude_deviation
)


def test_mean_amplitude_deviation():
    # Mock epoch:
    # mean = 10
    signal = np.array([
        8, 10, 12, 10
    ], dtype=float)

    # Aynı kanalın normal/reference ortalaması
    reference_mean = 2.0

    result = compute_mean_amplitude_deviation(
        signal,
        reference_mean
    )

    # 10 - 2 = 8
    expected = 8.0

    assert np.isclose(result, expected)