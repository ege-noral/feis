import numpy as np


def compute_mean_amplitude_deviation(signal, reference_mean):
    """
    Compute the deviation of an epoch's mean amplitude
    from the reference mean of the same channel.

    Parameters
    ----------
    signal : np.ndarray
        1D EEG signal for one channel in one epoch.

    reference_mean : float
        Reference mean amplitude for the same channel.

    Returns
    -------
    float
        Signed mean amplitude deviation.
    """

    epoch_mean = np.mean(signal)

    return epoch_mean - reference_mean