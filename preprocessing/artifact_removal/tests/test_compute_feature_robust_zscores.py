import numpy as np
import pandas as pd

from preprocessing.artifact_removal.outlier_detection.zscore_table import (
    build_zscore_table
)


def test_compute_feature_robust_zscores():

    feature_df = pd.DataFrame({
        "epoch": [
            0, 1, 2, 3, 4,
            0, 1, 2, 3, 4,
        ],

        "channel": [
            "F3", "F3", "F3", "F3", "F3",
            "FC5", "FC5", "FC5", "FC5", "FC5",
        ],

        "peak_to_peak": [
            10, 11, 9, 10, 50,       # F3: 50 outlier
            20, 21, 19, 20, 22,      # FC5: normal
        ],

        "variance": [
            5, 6, 4, 5, 30,
            10, 11, 9, 10, 12,
        ],

        "kurtosis": [
            1, 1.1, 0.9, 1, 8,
            2, 2.1, 1.9, 2, 2.2,
        ],

        "max_gradient": [
            2, 2.1, 1.9, 2, 15,
            4, 4.1, 3.9, 4, 4.2,
        ],

        "hf_power_ratio": [
            0.10, 0.11, 0.09, 0.10, 0.60,
            0.20, 0.21, 0.19, 0.20, 0.22,
        ],

        "mean_amplitude_deviation": [
            0, 1, -1, 0, 10,
            0, 1, -1, 0, 2,
        ],
    })

    result = build_zscore_table(
        feature_df
    )

    expected_zscore_columns = [
        "peak_to_peak_rz",
        "variance_rz",
        "kurtosis_rz",
        "max_gradient_rz",
        "hf_power_ratio_rz",
        "mean_amplitude_deviation_rz",
    ]

    # Tüm robust z-score kolonları oluşmuş mu?
    for column in expected_zscore_columns:
        assert column in result.columns

    # F3'ün son epoch'u P2P açısından
    # F3 içindeki en büyük outlier olmalı
    f3 = result[result["channel"] == "F3"]

    outlier_z = np.abs(
        f3.loc[
            f3["epoch"] == 4,
            "peak_to_peak_rz"
        ].iloc[0]
    )

    assert outlier_z == np.max(
        np.abs(f3["peak_to_peak_rz"])
    )