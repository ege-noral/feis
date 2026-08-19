import numpy as np

from preprocessing.artifact_removal.outlier_detection.robust_zscore import (
    compute_robust_zscore
)


def test_robust_zscore_detects_outlier():
    values = np.array([
        -1, 0, 1,
        -1, 0, 1,
        -1, 0, 1,
        -1, 0, 20
    ], dtype=float)

    result = compute_robust_zscore(values)

    # Son değer en güçlü outlier olmalı
    assert np.abs(result[-1]) == np.max(np.abs(result))

    # Ve açık biçimde outlier seviyesinde olmalı
    assert np.abs(result[-1]) > 3.5