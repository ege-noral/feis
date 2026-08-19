import numpy as np

from preprocessing.artifact_removal.dataset_feature_extraction import (
    extract_dataset_features
)


def test_extract_dataset_features():
    # 2 epoch, 3 kanal, 5 sample
    epochs = np.array([
        [
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7],
        ],
        [
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7],
            [4, 5, 6, 7, 8],
        ],
    ], dtype=float)

    channel_names = [
        "F3",
        "FC5",
        "AF3",
    ]

    result = extract_dataset_features(
        epochs,
        channel_names=channel_names,
        fs=256,
    )

    # 2 epoch × 3 kanal = 6 satır
    assert len(result) == 6

    assert "peak_to_peak" in result.columns
    assert "variance" in result.columns
    assert "kurtosis" in result.columns
    assert "max_gradient" in result.columns
    assert "hf_power_ratio" in result.columns
    assert "mean_amplitude_deviation" in result.columns

    assert set(result["channel"]) == {
        "F3",
        "FC5",
        "AF3",
    }