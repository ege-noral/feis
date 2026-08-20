import numpy as np
from scipy.signal import welch


def compute_hf_power_ratio(
    signal,
    fs=256,
    hf_band=(20, 45),
    total_band=(0.5, 45),
):
    """
    Compute the ratio of high-frequency power to total EEG power.

    Parameters
    ----------
    signal : np.ndarray
        1D EEG signal.

    fs : int
        Sampling frequency.

    hf_band : tuple
        High-frequency band used for artifact detection.

    total_band : tuple
        Frequency range used as total power.

    Returns
    -------
    float
        High-frequency power ratio.
    """

    freqs, psd = welch(
        signal,
        fs=fs,
        nperseg=min(fs, len(signal))
    )

    hf_mask = (
        (freqs >= hf_band[0]) &
        (freqs <= hf_band[1])
    )

    total_mask = (
        (freqs >= total_band[0]) &
        (freqs <= total_band[1])
    )

    hf_power = np.trapezoid(
        psd[hf_mask],
        freqs[hf_mask]
    )

    total_power = np.trapezoid(
        psd[total_mask],
        freqs[total_mask]
    )

    if total_power == 0:
        return 0.0

    return hf_power / total_power