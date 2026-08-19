import numpy as np


def compute_peak_to_peak(signal: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute peak-to-peak amplitude of a signal.

    Peak-to-peak amplitude is defined as:

        max(signal) - min(signal)

    Parameters
    ----------
    signal : np.ndarray
        Input EEG signal.

        Examples:
        - Single channel: (n_samples,)
        - Single epoch: (n_channels, n_samples)
        - Multiple epochs: (n_epochs, n_channels, n_samples)

    axis : int, default=-1
        Axis corresponding to time/samples.

    Returns
    -------
    np.ndarray
        Peak-to-peak amplitude computed along the specified axis.

        For FEIS:
        input  shape: (160, 14, 1280)
        output shape: (160, 14)
    """

    signal = np.asarray(signal)

    if signal.size == 0:
        raise ValueError("Signal cannot be empty.")

    return np.ptp(signal, axis=axis)