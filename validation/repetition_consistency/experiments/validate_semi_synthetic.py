import numpy as np
import pandas as pd

from constants import EEG_CHANNELS
from data_loader import get_one_epoch

from preprocessing.demean import demean

from quality_control.repetition_consistency.morlet_tfr import (
    compute_morlet_tfr
)

from quality_control.repetition_consistency.self_similarity import (
    compute_tfr_self_similarity
)

from quality_control.repetition_consistency.repetition_detection import (
    detect_repetition_candidates
)

from quality_control.repetition_consistency.semi_synthetic import (
    create_semi_synthetic_epoch
)


# ======================================================
# CONFIGURATION
# ======================================================

SUBJECT_ID = 8
EPOCH_IDX = 44

CHANNEL_NAME = "P7"

FS = 256


# ------------------------------------------------------
# Current algorithm parameters.
#
# These are NOT optimized yet.
# We are first measuring how the current algorithm
# behaves.
# ------------------------------------------------------

WINDOW_SECONDS = 0.50

STEP_SECONDS = 0.10

SIMILARITY_PERCENTILE = 95.0

MIN_PEAK_DISTANCE_SECONDS = 0.50


# ------------------------------------------------------
# Semi-synthetic experiment
# ------------------------------------------------------

REPEAT_COUNTS = [
    0,
    2,
    3,
    4,
    5,
]

N_EXAMPLES_PER_COUNT = 20

MATCH_TOLERANCE_SECONDS = 0.35

BASE_RANDOM_SEED = 42


# ======================================================
# HELPER:
# MATCH DETECTED PEAKS TO TRUE REPETITIONS
# ======================================================

def match_detected_to_true(
    detected_times,
    true_times,
    tolerance_seconds=0.35,
):
    """
    Greedily match detected repetition times to known
    ground-truth event times.

    A detection is counted as correct if it lies within
    tolerance_seconds of an unmatched true event.
    """

    detected_times = np.asarray(
        detected_times,
        dtype=float,
    )

    true_times = np.asarray(
        true_times,
        dtype=float,
    )

    if len(true_times) == 0:

        return {
            "matched_count": 0,
            "false_positive_count": (
                len(detected_times)
            ),
            "missed_count": 0,
            "timing_errors": [],
        }

    if len(detected_times) == 0:

        return {
            "matched_count": 0,
            "false_positive_count": 0,
            "missed_count": (
                len(true_times)
            ),
            "timing_errors": [],
        }

    candidate_pairs = []

    for detected_idx, detected_time in enumerate(
        detected_times
    ):

        for true_idx, true_time in enumerate(
            true_times
        ):

            error = abs(
                detected_time
                - true_time
            )

            if error <= tolerance_seconds:

                candidate_pairs.append(
                    (
                        error,
                        detected_idx,
                        true_idx,
                    )
                )

    # Closest matches first
    candidate_pairs.sort(
        key=lambda x: x[0]
    )

    used_detected = set()
    used_true = set()

    timing_errors = []

    for (
        error,
        detected_idx,
        true_idx
    ) in candidate_pairs:

        if detected_idx in used_detected:
            continue

        if true_idx in used_true:
            continue

        used_detected.add(
            detected_idx
        )

        used_true.add(
            true_idx
        )

        timing_errors.append(
            error
        )

    matched_count = len(
        timing_errors
    )

    false_positive_count = (
        len(detected_times)
        - matched_count
    )

    missed_count = (
        len(true_times)
        - matched_count
    )

    return {
        "matched_count": matched_count,
        "false_positive_count": (
            false_positive_count
        ),
        "missed_count": missed_count,
        "timing_errors": timing_errors,
    }


# ======================================================
# LOAD REAL FEIS BACKGROUND
# ======================================================

epoch = get_one_epoch(
    subject_id=SUBJECT_ID,
    condition="resting",
    epoch_idx=EPOCH_IDX,
)

channel_idx = EEG_CHANNELS.index(
    CHANNEL_NAME
)

background = epoch[
    channel_idx
]

print(
    "Background shape:",
    background.shape
)


# ======================================================
# RUN EXPERIMENT
# ======================================================

rows = []

example_id = 0


for n_repeats in REPEAT_COUNTS:

    print(
        "\n========================================"
    )

    print(
        f"Testing {n_repeats} repetitions"
    )

    print(
        "========================================"
    )

    for repetition_example in range(
        N_EXAMPLES_PER_COUNT
    ):

        seed = (
            BASE_RANDOM_SEED
            + example_id
        )

        example_id += 1

        # ----------------------------------------------
        # Semi-synthetic EEG
        # ----------------------------------------------

        synthetic_signal, event_info = (
            create_semi_synthetic_epoch(
                background_signal=background,
                fs=FS,
                n_repeats=n_repeats,
                duration_range=(
                    0.40,
                    0.80,
                ),
                motif_std_ratio=0.50,
                min_gap_seconds=0.10,
                edge_margin_seconds=0.10,
                seed=seed,
            )
        )

        true_times = np.array([
            event[
                "center_seconds"
            ]
            for event in event_info
        ])

        # ----------------------------------------------
        # Same preprocessing used on real FEIS
        # ----------------------------------------------

        signal = demean(
            synthetic_signal
        )

        # ----------------------------------------------
        # Morlet TFR
        # ----------------------------------------------

        freqs, power = (
            compute_morlet_tfr(
                signal,
                fs=FS,
            )
        )

        # ----------------------------------------------
        # TFR self-similarity
        # ----------------------------------------------

        (
            similarity_matrix,
            window_centers,
        ) = (
            compute_tfr_self_similarity(
                power,
                fs=FS,
                window_seconds=(
                    WINDOW_SECONDS
                ),
                step_seconds=(
                    STEP_SECONDS
                ),
            )
        )

        # ----------------------------------------------
        # Current repetition detector
        # ----------------------------------------------

        result = (
            detect_repetition_candidates(
                similarity_matrix,
                window_centers,
                window_seconds=(
                    WINDOW_SECONDS
                ),
                similarity_percentile=(
                    SIMILARITY_PERCENTILE
                ),
                min_peak_distance_seconds=(
                    MIN_PEAK_DISTANCE_SECONDS
                ),
            )
        )

        detected_times = (
            result[
                "peak_times"
            ]
        )

        # ----------------------------------------------
        # Compare against known ground truth
        # ----------------------------------------------

        match_result = (
            match_detected_to_true(
                detected_times,
                true_times,
                tolerance_seconds=(
                    MATCH_TOLERANCE_SECONDS
                ),
            )
        )

        matched = (
            match_result[
                "matched_count"
            ]
        )

        false_positives = (
            match_result[
                "false_positive_count"
            ]
        )

        missed = (
            match_result[
                "missed_count"
            ]
        )

        timing_errors = (
            match_result[
                "timing_errors"
            ]
        )

        # ----------------------------------------------
        # Precision / recall
        # ----------------------------------------------

        if len(detected_times) > 0:

            precision = (
                matched
                / len(detected_times)
            )

        else:

            precision = (
                1.0
                if n_repeats == 0
                else 0.0
            )


        if n_repeats > 0:

            recall = (
                matched
                / n_repeats
            )

        else:

            recall = (
                1.0
                if len(detected_times) == 0
                else 0.0
            )


        if (
            precision + recall
        ) > 0:

            f1 = (
                2
                * precision
                * recall
                / (
                    precision
                    + recall
                )
            )

        else:

            f1 = 0.0


        if len(timing_errors) > 0:

            timing_mae = float(
                np.mean(
                    timing_errors
                )
            )

        else:

            timing_mae = np.nan


        rows.append({
            "true_repeats": n_repeats,
            "example": repetition_example,
            "detected_repeats": (
                len(detected_times)
            ),
            "matched": matched,
            "false_positives": (
                false_positives
            ),
            "missed": missed,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "timing_mae_seconds": (
                timing_mae
            ),
        })


# ======================================================
# RESULTS
# ======================================================

results = pd.DataFrame(
    rows
)


print(
    "\n\n========================================"
)

print(
    "INDIVIDUAL RESULTS"
)

print(
    "========================================"
)

print(
    results.to_string(
        index=False
    )
)


# ======================================================
# SUMMARY BY TRUE REPETITION COUNT
# ======================================================

summary = (
    results
    .groupby(
        "true_repeats"
    )
    .agg(
        mean_detected=(
            "detected_repeats",
            "mean",
        ),
        mean_matched=(
            "matched",
            "mean",
        ),
        mean_false_positives=(
            "false_positives",
            "mean",
        ),
        mean_missed=(
            "missed",
            "mean",
        ),
        mean_precision=(
            "precision",
            "mean",
        ),
        mean_recall=(
            "recall",
            "mean",
        ),
        mean_f1=(
            "f1",
            "mean",
        ),
        mean_timing_error=(
            "timing_mae_seconds",
            "mean",
        ),
    )
    .reset_index()
)


print(
    "\n\n========================================"
)

print(
    "SUMMARY"
)

print(
    "========================================"
)

print(
    summary.to_string(
        index=False
    )
)


# ======================================================
# GLOBAL SCORE
# ======================================================

mean_f1 = results[
    "f1"
].mean()

mean_count_error = np.mean(
    np.abs(
        results[
            "detected_repeats"
        ]
        - results[
            "true_repeats"
        ]
    )
)


print(
    "\n========================================"
)

print(
    "GLOBAL PERFORMANCE"
)

print(
    "========================================"
)

print(
    f"Mean F1: "
    f"{mean_f1:.3f}"
)

print(
    f"Mean absolute count error: "
    f"{mean_count_error:.3f}"
)