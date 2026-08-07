import numpy as np


def select_task_related_windows(
    thinking_windows,
    thinking_variance,
    resting_variance,
    z_threshold=2.0
):
    """
    Select thinking windows whose variance is significantly
    higher than the resting baseline.

    Parameters
    ----------
    thinking_windows : np.ndarray
        Shape (n_windows, n_channels, window_size)

    thinking_variance : np.ndarray
        Shape (n_windows, n_channels)

    resting_variance : np.ndarray
        Shape (n_windows, n_channels)

    z_threshold : float
        Z-score threshold relative to resting variance.

    Returns
    -------
    selected_windows : np.ndarray
    selected_indices : np.ndarray
    window_scores : np.ndarray
    """

    resting_mean = np.mean(resting_variance, axis=0)
    resting_std = np.std(resting_variance, axis=0)

    # Avoid division by zero
    resting_std = np.where(resting_std == 0, 1e-10, resting_std)

    z_scores = (
        thinking_variance - resting_mean
    ) / resting_std

    window_scores = np.mean(z_scores, axis=1)

    selected_indices = np.where(
        window_scores > z_threshold
    )[0]

    selected_windows = thinking_windows[selected_indices]

    return (
        selected_windows,
        selected_indices,
        window_scores
    )