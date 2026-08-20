import numpy as np

from quality_control.repetition_consistency.self_similarity import (
    compute_tfr_self_similarity
)


def test_self_similarity_detects_repeated_patterns():
    fs = 100

    # 5 frequency × 500 time samples
    power = np.zeros(
        (5, 500),
        dtype=float
    )

    # Tekrarlayan 2D motif
    motif = np.array([
        [0, 1, 2, 1, 0],
        [1, 3, 5, 3, 1],
        [0, 1, 2, 1, 0],
    ], dtype=float)

    # Aynı motifi 5 farklı zamana koy
    positions = [
        50,
        150,
        250,
        350,
        450,
    ]

    for start in positions:
        power[
            1:4,
            start:start + motif.shape[1]
        ] += motif

    window_seconds = 0.20
    step_seconds = 0.05

    similarity_matrix, centers = (
        compute_tfr_self_similarity(
            power,
            fs=fs,
            window_seconds=window_seconds,
            step_seconds=step_seconds,
        )
    )

    # --------------------------------------------------
    # 1. Matrix kare olmalı
    # --------------------------------------------------

    assert similarity_matrix.shape[0] == (
        similarity_matrix.shape[1]
    )

    # --------------------------------------------------
    # 2. Motiflerin başladığı pencereleri bul
    # --------------------------------------------------

    step_samples = int(
        round(step_seconds * fs)
    )

    motif_window_indices = [
        position // step_samples
        for position in positions
    ]

    # --------------------------------------------------
    # 3. Bu 5 pencerenin birbirleriyle similarity'si
    #    yüksek olmalı
    # --------------------------------------------------

    repeated_similarity = similarity_matrix[
        np.ix_(
            motif_window_indices,
            motif_window_indices
        )
    ]

    assert np.all(
        repeated_similarity > 0.9
    )