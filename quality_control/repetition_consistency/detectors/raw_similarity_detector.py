import numpy as np

from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


def compute_raw_repetition_scores(
    similarity_matrix,
    window_centers,
    min_lag_seconds,
    top_k=3,
):
    """
    Compute a repetition score for every sliding window.

    IMPORTANT:
    This function does NOT use percentile thresholds.

    For each window:

        1. Ignore nearby windows.
        2. Look at similarities with distant windows.
        3. Take the strongest top-k similarities.
        4. Average them.

    Therefore the score stays on the original
    similarity scale and can be compared between
    different epochs.

    Parameters
    ----------
    similarity_matrix : np.ndarray
        Self-similarity matrix:
        shape = (n_windows, n_windows)

    window_centers : np.ndarray
        Center time of every sliding window in seconds.

    min_lag_seconds : float
        Minimum temporal distance between two windows
        before they are allowed to count as a repetition.

    top_k : int
        Number of strongest distant similarities used
        to compute the score.

    Returns
    -------
    scores : np.ndarray
        Repetition score for every window.
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

    if top_k < 1:
        raise ValueError(
            "top_k must be at least 1."
        )

    n_windows = len(
        window_centers
    )

    scores = np.zeros(
        n_windows,
        dtype=float,
    )

    # --------------------------------------------------
    # Calculate score for every window
    # --------------------------------------------------

    for window_idx in range(
        n_windows
    ):

        current_time = (
            window_centers[
                window_idx
            ]
        )

        time_difference = np.abs(
            window_centers
            - current_time
        )

        # Only compare sufficiently distant windows.
        valid_mask = (
            time_difference
            >= min_lag_seconds
        )

        valid_similarities = (
            similarity_matrix[
                window_idx,
                valid_mask,
            ]
        )

        if len(
            valid_similarities
        ) == 0:
            scores[
                window_idx
            ] = 0.0

            continue

        k = min(
            top_k,
            len(valid_similarities),
        )

        # Get the strongest k similarities.
        top_values = np.partition(
            valid_similarities,
            -k,
        )[-k:]

        scores[
            window_idx
        ] = np.mean(
            top_values
        )

    return scores


def detect_repetition_peaks(
    similarity_matrix,
    window_centers,
    min_lag_seconds,
    top_k=3,
    smoothing_windows=3,
    peak_min_height=0.40,
    peak_prominence=0.05,
    min_peak_distance_seconds=0.50,
):
    """
    Detect repetition candidates directly from raw
    self-similarity values.

    Unlike the old detector, this function does NOT
    select the strongest percentile inside each epoch.

    Therefore its scores are comparable between
    different EEG epochs.

    Returns
    -------
    result : dict
        Contains raw scores, smoothed scores and
        detected peak times.
    """

    window_centers = np.asarray(
        window_centers,
        dtype=float,
    )

    # --------------------------------------------------
    # 1. Raw similarity-based score
    # --------------------------------------------------

    raw_scores = (
        compute_raw_repetition_scores(
            similarity_matrix=(
                similarity_matrix
            ),
            window_centers=(
                window_centers
            ),
            min_lag_seconds=(
                min_lag_seconds
            ),
            top_k=top_k,
        )
    )

    # --------------------------------------------------
    # 2. Smooth the score curve
    # --------------------------------------------------

    smoothing_windows = int(
        smoothing_windows
    )

    if smoothing_windows < 1:
        smoothing_windows = 1

    if smoothing_windows > 1:

        smoothed_scores = (
            uniform_filter1d(
                raw_scores,
                size=smoothing_windows,
                mode="nearest",
            )
        )

    else:

        smoothed_scores = (
            raw_scores.copy()
        )

    # --------------------------------------------------
    # 3. Convert minimum peak distance
    #    seconds -> sliding-window samples
    # --------------------------------------------------

    if len(window_centers) > 1:

        step_seconds = np.median(
            np.diff(
                window_centers
            )
        )

    else:

        step_seconds = 1.0

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
    # 4. Allow peaks close to the epoch boundaries
    #
    # scipy.find_peaks normally cannot identify the
    # first or last point as a peak.
    # We temporarily add low values around the curve.
    # --------------------------------------------------

    edge_floor = (
        np.min(smoothed_scores)
        - 1.0
    )

    padded_scores = np.concatenate([
        [edge_floor],
        smoothed_scores,
        [edge_floor],
    ])

    padded_peak_indices, properties = (
        find_peaks(
            padded_scores,
            height=peak_min_height,
            prominence=peak_prominence,
            distance=(
                min_peak_distance_windows
            ),
        )
    )

    # Remove padding offset.
    peak_indices = (
        padded_peak_indices
        - 1
    )

    # Only keep indices belonging to the real curve.
    valid_peak_mask = (
        (peak_indices >= 0)
        & (
            peak_indices
            < len(smoothed_scores)
        )
    )

    peak_indices = (
        peak_indices[
            valid_peak_mask
        ]
    )

    peak_times = (
        window_centers[
            peak_indices
        ]
    )

    peak_scores = (
        smoothed_scores[
            peak_indices
        ]
    )

    return {
        "raw_scores": raw_scores,
        "smoothed_scores": (
            smoothed_scores
        ),
        "peak_indices": peak_indices,
        "peak_times": peak_times,
        "peak_scores": peak_scores,
    }