import numpy as np


def compute_tfr_self_similarity(
    power,
    fs=256,
    window_seconds=0.75,
    step_seconds=0.10,
):
    """
    Compute a self-similarity matrix from a TFR power matrix
    using overlapping sliding windows.

    Parameters
    ----------
    power : np.ndarray
        TFR power matrix with shape:
        (n_frequencies, n_samples)

    fs : int or float
        Sampling frequency in Hz.

    window_seconds : float
        Length of each TFR window in seconds.

    step_seconds : float
        Sliding-window step size in seconds.

    Returns
    -------
    similarity_matrix : np.ndarray
        Pairwise similarity matrix with shape:
        (n_windows, n_windows)

    window_centers_seconds : np.ndarray
        Center time of each window in seconds.
    """

    power = np.asarray(power, dtype=float)

    if power.ndim != 2:
        raise ValueError(
            "power must be a 2D array "
            "(n_frequencies, n_samples)."
        )

    _, n_samples = power.shape

    window_samples = int(
        round(window_seconds * fs)
    )

    step_samples = int(
        round(step_seconds * fs)
    )

    if window_samples > n_samples:
        raise ValueError(
            "window_seconds is longer than the TFR duration."
        )

    if window_samples <= 0 or step_samples <= 0:
        raise ValueError(
            "window_seconds and step_seconds must be positive."
        )

    # --------------------------------------------------
    # Sliding-window starting positions
    # --------------------------------------------------

    starts = np.arange(
        0,
        n_samples - window_samples + 1,
        step_samples,
    )

    n_windows = len(starts)

    patches = []

    window_centers_seconds = []

    # --------------------------------------------------
    # Extract TFR patches
    # --------------------------------------------------

    for start in starts:

        end = start + window_samples

        patch = power[
            :,
            start:end
        ]

        # Power values can span a very large range.
        # Log transform makes the representation less
        # dominated by a few very large values.
        patch = np.log1p(patch)

        # Convert 2D frequency × time patch to one vector.
        vector = patch.ravel()

        # Remove its own mean so that we compare the
        # pattern rather than absolute power level.
        vector = vector - np.mean(vector)

        # Normalize length.
        norm = np.linalg.norm(vector)

        if norm > 0:
            vector = vector / norm

        patches.append(vector)

        center_sample = (
            start + window_samples / 2
        )

        window_centers_seconds.append(
            center_sample / fs
        )

    patches = np.asarray(patches)

    window_centers_seconds = np.asarray(
        window_centers_seconds
    )

    # --------------------------------------------------
    # Pairwise similarity
    #
    # Each row has unit norm, so matrix multiplication
    # gives cosine similarity.
    # --------------------------------------------------

    similarity_matrix = (
        patches @ patches.T
    )

    return (
        similarity_matrix,
        window_centers_seconds,
    )