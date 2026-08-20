import numpy as np
from scipy.signal import fftconvolve


def compute_morlet_tfr(
    signal,
    fs=256,
    freqs=None,
    n_cycles=6.0,
):
    """
    Compute Morlet wavelet time-frequency power
    representation of a 1D EEG signal.

    Parameters
    ----------
    signal : np.ndarray
        1D EEG signal.

    fs : int or float
        Sampling frequency in Hz.

    freqs : np.ndarray | None
        Frequencies to analyze.
        Default: 4-40 Hz.

    n_cycles : float
        Number of Morlet cycles.

    Returns
    -------
    freqs : np.ndarray
        Frequencies used.

    power : np.ndarray
        TFR power matrix with shape:
        (n_frequencies, n_samples)
    """

    signal = np.asarray(
        signal,
        dtype=float
    )

    if signal.ndim != 1:
        raise ValueError(
            "signal must be a 1D array."
        )

    if freqs is None:
        freqs = np.arange(
            4.0,
            41.0,
            1.0
        )

    freqs = np.asarray(
        freqs,
        dtype=float
    )

    power = np.zeros(
        (
            len(freqs),
            len(signal),
        ),
        dtype=float
    )

    for freq_idx, freq in enumerate(freqs):

        # Gaussian temporal width
        sigma_t = (
            n_cycles
            / (2 * np.pi * freq)
        )

        # Wavelet duration: ±4 sigma
        half_duration = 4 * sigma_t

        wavelet_time = np.arange(
            -half_duration,
            half_duration,
            1 / fs
        )

        # Complex Morlet wavelet
        wavelet = (
            np.exp(
                2j
                * np.pi
                * freq
                * wavelet_time
            )
            *
            np.exp(
                -(wavelet_time ** 2)
                / (2 * sigma_t ** 2)
            )
        )

        # Normalize wavelet energy
        wavelet = (
            wavelet
            / np.sqrt(
                np.sum(
                    np.abs(wavelet) ** 2
                )
            )
        )

        # ------------------------------------------
        # Reflection padding
        # ------------------------------------------

        pad_size = len(wavelet) // 2

        padded_signal = np.pad(
            signal,
            pad_width=pad_size,
            mode="reflect"
        )

        # ------------------------------------------
        # Convolution
        # ------------------------------------------

        padded_coefficients = fftconvolve(
            padded_signal,
            wavelet,
            mode="same"
        )

        # Remove temporary padding
        coefficients = padded_coefficients[
            pad_size:
            pad_size + len(signal)
        ]

        # Power
        power[freq_idx, :] = (
            np.abs(coefficients) ** 2
        )

    return freqs, power