import numpy as np

from preprocessing.artifact_removal.feature_vector import (
    artifact_feature_vector
)


def test_feature_vector():
    fs = 256
    duration = 5

    t = np.arange(0, duration, 1 / fs)

    # Basit mock EEG signal
    signal = (
        np.sin(2 * np.pi * 10 * t)
        + 0.2 * np.sin(2 * np.pi * 30 * t)
    )

    reference_mean = 0.0

    features = artifact_feature_vector(
        signal,
        reference_mean=reference_mean,
        fs=fs,
    )

    expected_features = {
        "peak_to_peak",
        "variance",
        "kurtosis",
        "max_gradient",
        "hf_power_ratio",
        "mean_amplitude_deviation",
    }

    assert set(features.keys()) == expected_features

    for value in features.values():
        assert np.isfinite(value)