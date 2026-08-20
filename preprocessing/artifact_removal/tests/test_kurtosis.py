import numpy as np

from preprocessing.artifact_removal.features.kurtosis import compute_kurtosis


def test_kurtosis_is_higher_for_spiky_signal():
    normal_signal = np.array([
        -1, 0, 1, -1, 0, 1,
        -1, 0, 1, -1, 0, 1
    ], dtype=float)

    spiky_signal = np.array([
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 10
    ], dtype=float)

    normal_kurtosis = compute_kurtosis(normal_signal)
    spiky_kurtosis = compute_kurtosis(spiky_signal)

    assert spiky_kurtosis > normal_kurtosis