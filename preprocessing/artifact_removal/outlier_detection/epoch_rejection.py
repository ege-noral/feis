import numpy as np


def reject_bad_epochs(
    epochs,
    artifact_scores,
    min_suspicious_channel_ratio=0.5,
):
    """
    Reject globally contaminated EEG epochs.

    An epoch is considered bad when the proportion of
    suspicious channels is greater than or equal to the
    specified threshold.

    Parameters
    ----------
    epochs : np.ndarray
        EEG epochs with shape:
        (n_epochs, n_channels, n_samples)

    artifact_scores : pandas.DataFrame
        Epoch-level artifact summary containing:
        - epoch
        - suspicious_channel_ratio

    min_suspicious_channel_ratio : float
        Minimum fraction of suspicious channels required
        to reject an epoch.

        Default = 0.5
        For 14 channels, this corresponds to 7 channels.

    Returns
    -------
    clean_epochs : np.ndarray
        Epochs that are not globally contaminated.

    bad_epoch_indices : np.ndarray
        Indices of rejected epochs.
    """

    bad_epoch_indices = (
        artifact_scores.loc[
            artifact_scores["suspicious_channel_ratio"]
            >= min_suspicious_channel_ratio,
            "epoch",
        ]
        .to_numpy(dtype=int)
    )

    keep_mask = np.ones(
        len(epochs),
        dtype=bool
    )

    keep_mask[bad_epoch_indices] = False

    clean_epochs = epochs[keep_mask]

    return clean_epochs, bad_epoch_indices