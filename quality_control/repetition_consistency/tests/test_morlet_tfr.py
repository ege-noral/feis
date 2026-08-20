import numpy as np

from quality_control.repetition_consistency.morlet_tfr import (
    compute_morlet_tfr
)


def test_morlet_tfr_detects_frequency_change():
    fs = 256
    duration = 5.0

    t = np.arange(0, duration, 1 / fs)

    # İlk 2.5 saniye: 10 Hz
    # Son 2.5 saniye: 30 Hz
    signal = np.where(
        t < 2.5,
        np.sin(2 * np.pi * 10 * t),
        np.sin(2 * np.pi * 30 * t),
    )

    freqs, power = compute_morlet_tfr(
        signal,
        fs=fs,
        freqs=np.array([10.0, 30.0]),
        n_cycles=6.0,
    )

    # --------------------------------------------------
    # Shape kontrolü
    # --------------------------------------------------

    assert power.shape == (
        2,
        len(signal),
    )

    # 10 ve 30 Hz'in satırlarını bul
    idx_10 = np.where(freqs == 10.0)[0][0]
    idx_30 = np.where(freqs == 30.0)[0][0]

    # --------------------------------------------------
    # Edge ve geçiş bölgelerinden uzak alanları kullan
    # --------------------------------------------------

    first_region = (
        (t >= 0.5)
        & (t <= 2.0)
    )

    second_region = (
        (t >= 3.0)
        & (t <= 4.5)
    )

    # İlk bölümde 10 Hz baskın olmalı
    power_10_first = np.mean(
        power[idx_10, first_region]
    )

    power_30_first = np.mean(
        power[idx_30, first_region]
    )

    assert power_10_first > power_30_first

    # İkinci bölümde 30 Hz baskın olmalı
    power_30_second = np.mean(
        power[idx_30, second_region]
    )

    power_10_second = np.mean(
        power[idx_10, second_region]
    )

    assert power_30_second > power_10_second