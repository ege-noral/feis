import numpy as np

from .windowing import create_sliding_windows


def create_trial_windows(
    trials,
    window_size,
    overlap=0.5
):
    """
    Parameters
    ----------
    trials : np.ndarray
        Shape (n_trials, n_channels, n_samples)

    Returns
    -------
    np.ndarray
        Shape
        (n_trials,
         n_windows,
         n_channels,
         window_size)
    """

    all_windows = []

    for trial in trials:

        windows = create_sliding_windows(
            trial,
            window_size=window_size,
            overlap=overlap
        )

        all_windows.append(windows)

    return np.array(all_windows)