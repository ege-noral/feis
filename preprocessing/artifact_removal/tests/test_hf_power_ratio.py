import numpy as np

from preprocessing.artifact_removal.features.hf_power_ratio import (
    compute_hf_power_ratio
)


def test_hf_power_ratio_is_higher_for_high_frequency_signal():
    fs = 256
    duration = 5

    t = np.arange(0, duration, 1 / fs)

    # Normal EEG benzeri düşük frekans
    low_frequency_signal = np.sin(
        2 * np.pi * 5 * t
    )

    # Kas artefaktına benzer yüksek frekans bileşeni
    high_frequency_signal = np.sin(
        2 * np.pi * 30 * t
    )

    low_ratio = compute_hf_power_ratio(
        low_frequency_signal,
        fs=fs
    )

    high_ratio = compute_hf_power_ratio(
        high_frequency_signal,
        fs=fs
    )

    assert high_ratio > low_ratio