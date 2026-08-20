import numpy as np
from scipy.signal import find_peaks


def detect_repetition_candidates(
    similarity_matrix,
    window_centers,
    window_seconds=0.75,
    similarity_percentile=95.0,
    min_peak_distance_seconds=0.50,
    smoothing_windows=3,
):
    """
    Detect candidate repetition times from a TFR
    self-similarity matrix.

    Each sliding window receives a score according to
    how strongly it matches non-overlapping windows at
    other time points.

    Peaks in this score are interpreted as candidate
    repeated motif times.
    """

    similarity_matrix = np.asarray(
        similarity_matrix,
        dtype=float,
    )

    window_centers = np.asarray(
        window_centers,
        dtype=float,
    )

    if similarity_matrix.ndim != 2:
        raise ValueError(
            "similarity_matrix must be 2D."
        )

    if (
        similarity_matrix.shape[0]
        != similarity_matrix.shape[1]
    ):
        raise ValueError(
            "similarity_matrix must be square."
        )

    if (
        len(window_centers)
        != similarity_matrix.shape[0]
    ):
        raise ValueError(
            "window_centers length must match "
            "similarity_matrix size."
        )

    # --------------------------------------------------
    # Time difference between every pair of windows
    # --------------------------------------------------

    time_difference = np.abs(
        window_centers[:, None]
        - window_centers[None, :]
    )

    # --------------------------------------------------
    # Ignore overlapping / nearby windows
    # --------------------------------------------------

    valid_mask = (
        time_difference
        >= window_seconds
    )

    valid_values = similarity_matrix[
        valid_mask
    ]

    if len(valid_values) == 0:
        return {
            "threshold": np.nan,
            "raw_scores": np.array([]),
            "smoothed_scores": np.array([]),
            "peak_indices": np.array([]),
            "peak_times": np.array([]),
            "peak_scores": np.array([]),
            "connection_counts": np.array([]),
        }

    # --------------------------------------------------
    # Keep strongest non-local similarities
    # --------------------------------------------------

    threshold = np.percentile(
        valid_values,
        similarity_percentile,
    )

    strong_mask = (
        valid_mask
        & (
            similarity_matrix
            >= threshold
        )
    )

    # --------------------------------------------------
    # Score each window
    #
    # Only similarity ABOVE threshold contributes.
    # --------------------------------------------------

    excess_similarity = np.where(
        strong_mask,
        similarity_matrix - threshold,
        0.0,
    )

    raw_scores = np.sum(
        excess_similarity,
        axis=1,
    )

    connection_counts = np.sum(
        strong_mask,
        axis=1,
    )

    # --------------------------------------------------
    # Smooth neighbouring sliding-window scores
    # --------------------------------------------------

    if smoothing_windows > 1:

        kernel = np.ones(
            smoothing_windows,
            dtype=float,
        )

        kernel /= np.sum(kernel)

        smoothed_scores = np.convolve(
            raw_scores,
            kernel,
            mode="same",
        )

    else:

        smoothed_scores = (
            raw_scores.copy()
        )

    # --------------------------------------------------
    # Estimate step size
    # --------------------------------------------------

    if len(window_centers) > 1:

        step_seconds = np.median(
            np.diff(window_centers)
        )

    else:

        step_seconds = (
            window_seconds
        )

    min_peak_distance_windows = max(
        1,
        int(
            round(
                min_peak_distance_seconds
                / step_seconds
            )
        ),
    )

    # --------------------------------------------------
    # Find candidate repetition peaks
    # --------------------------------------------------

    peak_indices, _ = find_peaks(
        smoothed_scores,
        distance=min_peak_distance_windows,
    )

    # Ignore zero-score peaks
    peak_indices = peak_indices[
        smoothed_scores[
            peak_indices
        ] > 0
    ]

    peak_times = window_centers[
        peak_indices
    ]

    peak_scores = smoothed_scores[
        peak_indices
    ]

    return {
        "threshold": threshold,
        "raw_scores": raw_scores,
        "smoothed_scores": smoothed_scores,
        "peak_indices": peak_indices,
        "peak_times": peak_times,
        "peak_scores": peak_scores,
        "connection_counts": connection_counts,
    }