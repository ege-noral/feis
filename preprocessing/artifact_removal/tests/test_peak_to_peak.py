import numpy as np

from preprocessing.artifact_removal.features.peak_to_peak import compute_peak_to_peak


def test_peak_to_peak():
    signal = np.array([-2, 1, 3, 7, -5], dtype=float)

    result = compute_peak_to_peak(signal)

    expected = 12.0

    assert np.isclose(result, expected)