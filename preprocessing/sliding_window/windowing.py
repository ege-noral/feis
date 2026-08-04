import numpy as np

def create_sliding_windows(eeg_data, window_size, overlap=0.5):
    """
    Create overlapping sliding windows from EEG data.

    Parameters
    ----------
    eeg_data : np.ndarray
        EEG data with shape (n_channels, n_samples)

    window_size : int
        Window length in samples

    overlap : float
        Overlap ratio between consecutive windows
        (e.g., 0.5 for 50% overlap)

    Returns
    -------
    np.ndarray
        Shape: (n_windows, n_channels, window_size)
    """

    step = int(window_size * (1 - overlap))

    if step <= 0:
        raise ValueError("Overlap is too large.")

    windows = []

    n_channels, n_samples = eeg_data.shape

    for start in range(0, n_samples - window_size + 1, step):
        end = start + window_size
        windows.append(eeg_data[:, start:end])

    return np.array(windows)