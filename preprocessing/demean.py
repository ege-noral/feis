import numpy as np


def demean(eeg, axis=-1):
    """
    Remove the mean (DC offset) from EEG data.

    Parameters
    ----------
    eeg : np.ndarray
        EEG signal or EEG array.

    axis : int
        Axis corresponding to time.

    Returns
    -------
    np.ndarray
        Demeaned EEG data.
    """

    eeg = np.asarray(eeg, dtype=float)

    mean = np.mean(
        eeg,
        axis=axis,
        keepdims=True,
    )

    return eeg - mean