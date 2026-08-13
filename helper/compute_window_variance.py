import numpy as np


def compute_window_variance(
    trial_windows
):
    """
    Parameters
    ----------
    trial_windows : np.ndarray

    Shape:
    (n_trials,
     n_windows,
     n_channels,
     window_size)

    Returns
    -------
    np.ndarray

    Shape:
    (n_trials,
     n_windows,
     n_channels)
    """

    return np.var(
        trial_windows,
        axis=2
    )