import numpy as np
import pandas as pd

from preprocessing.artifact_removal.feature_vector import (
    artifact_feature_vector
)


def extract_dataset_features(
    epochs,
    channel_names=None,
    fs=256,
):
    """
    Extract artifact features for every channel in every epoch.

    Parameters
    ----------
    epochs : np.ndarray
        Shape: (n_epochs, n_channels, n_samples)

    channel_names : list[str] | None
        EEG channel names.

    fs : int
        Sampling frequency.

    Returns
    -------
    pd.DataFrame
        One row for each epoch-channel pair.
    """

    n_epochs, n_channels, _ = epochs.shape

    # Her kanalın bütün epochlar ve bütün örnekler boyunca ortalaması
    reference_means = np.mean(
        epochs,
        axis=(0, 2)
    )

    rows = []

    for epoch_idx in range(n_epochs):

        for channel_idx in range(n_channels):

            signal = epochs[
                epoch_idx,
                channel_idx,
                :
            ]

            features = artifact_feature_vector(
                signal=signal,
                reference_mean=reference_means[channel_idx],
                fs=fs,
            )

            row = {
                "epoch": epoch_idx,
                "channel_idx": channel_idx,
                **features,
            }

            if channel_names is not None:
                row["channel"] = channel_names[channel_idx]

            rows.append(row)

    return pd.DataFrame(rows)