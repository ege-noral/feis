import json

import numpy as np
import optuna
import pandas as pd

from constants import EEG_CHANNELS
from data_loader import get_one_epoch

from preprocessing.demean import demean

from quality_control.repetition_consistency.morlet_tfr import (
    compute_morlet_tfr,
)

from quality_control.repetition_consistency.self_similarity import (
    compute_tfr_self_similarity,
)

from quality_control.repetition_consistency.detectors.raw_similarity_detector import (
    detect_repetition_peaks,
)

from quality_control.repetition_consistency.semi_synthetic import (
    create_semi_synthetic_epoch,
)


# ======================================================
# GENERAL SETTINGS
# ======================================================

CHANNEL_NAME = "P7"

FS = 256


# ======================================================
# TRAIN / VALIDATION BACKGROUND EPOCHS
#
# IMPORTANT:
#
# We use different resting epochs for optimization
# and final validation.
#
# Therefore Optuna cannot simply memorize one
# particular resting epoch.
# ======================================================

# ======================================================
# SUBJECT-WISE TRAIN / VALIDATION SPLIT
# ======================================================

TRAIN_SUBJECTS = [
    1,
    2,
    3,
    4,
]

VALIDATION_SUBJECTS = [
    5,
    8,
]


# Her subject'ten kaç resting epoch kullanacağız?
BACKGROUND_EPOCHS = list(
    range(10)
)

# Yani:
#
# Subject 1 -> resting epochs 0-9
# Subject 2 -> resting epochs 0-9
# ...
#
# Training:
# 4 subjects × 10 epochs = 40 backgrounds
#
# Validation:
# 2 subjects × 10 epochs = 20 backgrounds


# ======================================================
# REPETITION COUNTS
# ======================================================

REPEAT_COUNTS = [
    0,
    2,
    3,
    4,
    5,
]


# ======================================================
# SEMI-SYNTHETIC SETTINGS
# ======================================================

DURATION_RANGE = (
    0.40,
    0.80,
)

MOTIF_STD_RATIO = 0.50

MIN_GAP_SECONDS = 0.10

EDGE_MARGIN_SECONDS = 0.10


# ======================================================
# MATCHING SETTINGS
#
# A detected peak is considered correct if it is
# within ±0.35 seconds of a true repetition center.
# ======================================================

MATCH_TOLERANCE_SECONDS = 0.35


# ======================================================
# OPTUNA
# ======================================================

N_TRIALS = 60

RANDOM_SEED = 42


# ======================================================
# MATCH DETECTED PEAKS TO TRUE EVENTS
# ======================================================

def match_detected_to_true(
    detected_times,
    true_times,
    tolerance_seconds,
):
    """
    Match detected peak times with known true repetition
    times.

    Each detection can match at most one true event.
    Each true event can match at most one detection.
    """

    detected_times = np.asarray(
        detected_times,
        dtype=float,
    )

    true_times = np.asarray(
        true_times,
        dtype=float,
    )

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

            if (
                error
                <= tolerance_seconds
            ):

                candidate_pairs.append(
                    (
                        error,
                        detected_idx,
                        true_idx,
                    )
                )

    # Closest matches first.
    candidate_pairs.sort(
        key=lambda x: x[0]
    )

    used_detected = set()
    used_true = set()

    timing_errors = []

    for (
        error,
        detected_idx,
        true_idx,
    ) in candidate_pairs:

        if (
            detected_idx
            in used_detected
        ):
            continue

        if (
            true_idx
            in used_true
        ):
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
        "matched": matched_count,
        "false_positives": (
            false_positive_count
        ),
        "missed": missed_count,
        "timing_errors": (
            timing_errors
        ),
    }


# ======================================================
# SCORE ONE EXAMPLE
# ======================================================

def calculate_example_score(
    detected_times,
    true_times,
):
    """
    Calculate performance for one semi-synthetic epoch.

    For positive examples:
        standard event-level F1 score.

    For zero-repetition examples:
        false positives are strongly penalized.
    """

    detected_times = np.asarray(
        detected_times,
        dtype=float,
    )

    true_times = np.asarray(
        true_times,
        dtype=float,
    )

    match_result = (
        match_detected_to_true(
            detected_times=(
                detected_times
            ),
            true_times=true_times,
            tolerance_seconds=(
                MATCH_TOLERANCE_SECONDS
            ),
        )
    )

    matched = (
        match_result[
            "matched"
        ]
    )

    false_positives = (
        match_result[
            "false_positives"
        ]
    )

    missed = (
        match_result[
            "missed"
        ]
    )

    # --------------------------------------------------
    # Special case: no real repetition exists
    # --------------------------------------------------

    if len(true_times) == 0:

        # Perfect:
        # 0 real events, 0 detections
        if len(detected_times) == 0:

            score = 1.0

        else:

            # Every false positive lowers the score.
            #
            # 1 false positive -> 0.75
            # 2 false positives -> 0.50
            # 3 false positives -> 0.25
            # 4+ false positives -> 0
            score = max(
                0.0,
                1.0
                - (
                    0.25
                    * len(
                        detected_times
                    )
                ),
            )

        precision = (
            1.0
            if len(detected_times) == 0
            else 0.0
        )

        recall = 1.0

        f1 = score

    # --------------------------------------------------
    # Normal positive examples
    # --------------------------------------------------

    else:

        if len(detected_times) > 0:

            precision = (
                matched
                / len(detected_times)
            )

        else:

            precision = 0.0

        recall = (
            matched
            / len(true_times)
        )

        if (
            precision
            + recall
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

        score = f1

    if len(
        match_result[
            "timing_errors"
        ]
    ) > 0:

        timing_error = float(
            np.mean(
                match_result[
                    "timing_errors"
                ]
            )
        )

    else:

        timing_error = np.nan

    return {
        "score": score,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "matched": matched,
        "false_positives": (
            false_positives
        ),
        "missed": missed,
        "timing_error": (
            timing_error
        ),
    }


# ======================================================
# CREATE DATASET
# ======================================================

def create_dataset(
    subject_ids,
    background_epoch_indices,
    seed_offset,
):
    """
    Create a semi-synthetic dataset using resting EEG
    backgrounds from multiple subjects.

    Each real resting background is used to create:

        0 repetitions
        2 repetitions
        3 repetitions
        4 repetitions
        5 repetitions

    Train and validation subjects are kept separate.
    """

    channel_idx = (
        EEG_CHANNELS.index(
            CHANNEL_NAME
        )
    )

    dataset = []

    example_number = 0

    # ==================================================
    # SUBJECT LOOP
    # ==================================================

    for subject_id in subject_ids:

        # ==============================================
        # EPOCH LOOP
        # ==============================================

        for epoch_idx in (
            background_epoch_indices
        ):

            # ------------------------------------------
            # Load real FEIS resting EEG
            # ------------------------------------------

            resting_epoch = (
                get_one_epoch(
                    subject_id=subject_id,
                    condition="resting",
                    epoch_idx=epoch_idx,
                )
            )

            background_signal = (
                resting_epoch[
                    channel_idx
                ]
            )

            # ==========================================
            # Create:
            #
            # 0 / 2 / 3 / 4 / 5 repetition versions
            # ==========================================

            for n_repeats in (
                REPEAT_COUNTS
            ):

                seed = (
                    RANDOM_SEED
                    + seed_offset
                    + example_number
                )

                example_number += 1

                (
                    synthetic_signal,
                    event_info,
                ) = (
                    create_semi_synthetic_epoch(
                        background_signal=(
                            background_signal
                        ),
                        fs=FS,
                        n_repeats=n_repeats,
                        duration_range=(
                            DURATION_RANGE
                        ),
                        motif_std_ratio=(
                            MOTIF_STD_RATIO
                        ),
                        min_gap_seconds=(
                            MIN_GAP_SECONDS
                        ),
                        edge_margin_seconds=(
                            EDGE_MARGIN_SECONDS
                        ),
                        seed=seed,
                    )
                )

                # --------------------------------------
                # Ground truth repetition centers
                # --------------------------------------

                true_times = np.array([
                    event[
                        "center_seconds"
                    ]
                    for event
                    in event_info
                ])

                # --------------------------------------
                # Same preprocessing as real EEG
                # --------------------------------------

                signal = demean(
                    synthetic_signal
                )

                # --------------------------------------
                # Morlet TFR
                #
                # Calculated only once.
                # --------------------------------------

                (
                    freqs,
                    power,
                ) = (
                    compute_morlet_tfr(
                        signal,
                        fs=FS,
                    )
                )

                # --------------------------------------
                # Save example
                # --------------------------------------

                dataset.append({
                    "subject_id": (
                        subject_id
                    ),
                    "background_epoch": (
                        epoch_idx
                    ),
                    "true_repeats": (
                        n_repeats
                    ),
                    "true_times": (
                        true_times
                    ),
                    "power": power,
                })

    return dataset

# ======================================================
# EVALUATE ONE PARAMETER SET
# ======================================================

def evaluate_parameters(
    dataset,
    params,
):
    """
    Run one hyperparameter combination on a complete
    dataset.
    """

    rows = []

    for example in dataset:

        power = example[
            "power"
        ]

        # ----------------------------------------------
        # Self-similarity
        # ----------------------------------------------

        (
            similarity_matrix,
            window_centers,
        ) = (
            compute_tfr_self_similarity(
                power=power,
                fs=FS,
                window_seconds=(
                    params[
                        "window_seconds"
                    ]
                ),
                step_seconds=(
                    params[
                        "step_seconds"
                    ]
                ),
            )
        )

        # ----------------------------------------------
        # Minimum lag
        #
        # At least the window size itself.
        #
        # extra_lag_seconds allows Optuna to demand
        # even more temporal separation.
        # ----------------------------------------------

        min_lag_seconds = (
            params[
                "window_seconds"
            ]
            + params[
                "extra_lag_seconds"
            ]
        )

        # ----------------------------------------------
        # Detect repetitions
        # ----------------------------------------------

        result = (
            detect_repetition_peaks(
                similarity_matrix=(
                    similarity_matrix
                ),
                window_centers=(
                    window_centers
                ),
                min_lag_seconds=(
                    min_lag_seconds
                ),
                top_k=(
                    params[
                        "top_k"
                    ]
                ),
                smoothing_windows=(
                    params[
                        "smoothing_windows"
                    ]
                ),
                peak_min_height=(
                    params[
                        "peak_min_height"
                    ]
                ),
                peak_prominence=(
                    params[
                        "peak_prominence"
                    ]
                ),
                min_peak_distance_seconds=(
                    params[
                        "min_peak_distance_seconds"
                    ]
                ),
            )
        )

        detected_times = (
            result[
                "peak_times"
            ]
        )

        true_times = (
            example[
                "true_times"
            ]
        )

        metrics = (
            calculate_example_score(
                detected_times=(
                    detected_times
                ),
                true_times=(
                    true_times
                ),
            )
        )

        rows.append({
            "subject_id": (
                example[
                    "subject_id"
                ]
            ),
    
            "background_epoch": (
                example[
                    "background_epoch"
                ]
            ),
    
            "true_repeats": (
                example[
                    "true_repeats"
                ]
            ),
    
            "detected_repeats": (
                len(
                    detected_times
                )
            ),
    
            "matched": (
                metrics[
                    "matched"
                ]
            ),
    
            "false_positives": (
                metrics[
                    "false_positives"
                ]
            ),
    
            "missed": (
                metrics[
                    "missed"
                ]
            ),
    
            "precision": (
                metrics[
                    "precision"
                ]
            ),
    
            "recall": (
                metrics[
                    "recall"
                ]
            ),
    
            "f1": (
                metrics[
                    "f1"
                ]
            ),
    
            "score": (
                metrics[
                    "score"
                ]
            ),
    
            "timing_error": (
                metrics[
                    "timing_error"
                ]
        ),
    })

    results = pd.DataFrame(
        rows
    )

    mean_score = (
        results[
            "score"
        ].mean()
    )

    return (
        mean_score,
        results,
    )


# ======================================================
# OPTUNA OBJECTIVE
# ======================================================

def objective(
    trial,
):
    """
    Optuna changes these parameters automatically and
    asks:

        Which combination gives the highest
        semi-synthetic detection score?
    """

    params = {

        # ----------------------------------------------
        # TFR sliding-window duration
        # ----------------------------------------------

        "window_seconds":
            trial.suggest_categorical(
                "window_seconds",
                [
                    0.40,
                    0.50,
                    0.60,
                    0.75,
                    0.90,
                    1.00,
                ],
            ),

        # ----------------------------------------------
        # Sliding step
        # ----------------------------------------------

        "step_seconds":
            trial.suggest_categorical(
                "step_seconds",
                [
                    0.05,
                    0.10,
                    0.15,
                ],
            ),

        # ----------------------------------------------
        # Extra distance beyond window length before
        # two windows can count as repetitions.
        # ----------------------------------------------

        "extra_lag_seconds":
            trial.suggest_categorical(
                "extra_lag_seconds",
                [
                    0.00,
                    0.10,
                    0.20,
                    0.30,
                    0.40,
                ],
            ),

        # ----------------------------------------------
        # How many strongest distant similarities are
        # averaged for every time window?
        # ----------------------------------------------

        "top_k":
            trial.suggest_int(
                "top_k",
                1,
                5,
            ),

        # ----------------------------------------------
        # Smooth candidate score curve
        # ----------------------------------------------

        "smoothing_windows":
            trial.suggest_categorical(
                "smoothing_windows",
                [
                    1,
                    3,
                    5,
                ],
            ),

        # ----------------------------------------------
        # Minimum RAW similarity score needed for a
        # candidate peak.
        #
        # This is GLOBAL.
        #
        # It does NOT change separately for every epoch.
        # ----------------------------------------------

        "peak_min_height":
            trial.suggest_float(
                "peak_min_height",
                0.10,
                0.80,
            ),

        # ----------------------------------------------
        # Peak must stand out from its neighborhood.
        # ----------------------------------------------

        "peak_prominence":
            trial.suggest_float(
                "peak_prominence",
                0.00,
                0.25,
            ),

        # ----------------------------------------------
        # Two detected events cannot be closer than
        # this.
        # ----------------------------------------------

        "min_peak_distance_seconds":
            trial.suggest_categorical(
                "min_peak_distance_seconds",
                [
                    0.30,
                    0.40,
                    0.50,
                    0.60,
                    0.75,
                    0.90,
                ],
            ),
    }

    mean_score, _ = (
        evaluate_parameters(
            dataset=TRAIN_DATASET,
            params=params,
        )
    )

    return mean_score


# ======================================================
# BUILD FIXED DATASETS
# ======================================================

print(
    "\nCreating TRAIN dataset..."
)

TRAIN_DATASET = (
    create_dataset(
        subject_ids=(
            TRAIN_SUBJECTS
        ),
        background_epoch_indices=(
            BACKGROUND_EPOCHS
        ),
        seed_offset=0,
    )
)

print(
    "Train examples:",
    len(TRAIN_DATASET)
)


print(
    "\nCreating VALIDATION dataset..."
)

VALIDATION_DATASET = (
    create_dataset(
        subject_ids=(
            VALIDATION_SUBJECTS
        ),
        background_epoch_indices=(
            BACKGROUND_EPOCHS
        ),
        seed_offset=10000,
    )
)

print(
    "Validation examples:",
    len(
        VALIDATION_DATASET
    )
)


# ======================================================
# RUN OPTUNA
# ======================================================

print(
    "\nStarting hyperparameter optimization..."
)

sampler = (
    optuna.samplers.TPESampler(
        seed=RANDOM_SEED
    )
)

study = (
    optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )
)

study.optimize(
    objective,
    n_trials=N_TRIALS,
    show_progress_bar=True,
)


# ======================================================
# BEST PARAMETERS
# ======================================================

best_params = (
    study.best_params
)

best_train_score = (
    study.best_value
)


print(
    "\n\n========================================"
)

print(
    "BEST PARAMETERS"
)

print(
    "========================================"
)

for key, value in (
    best_params.items()
):

    print(
        f"{key}: {value}"
    )


print(
    "\nBest training score:",
    round(
        best_train_score,
        4,
    )
)


# ======================================================
# EVALUATE BEST PARAMETERS ON TRAIN DATA
# ======================================================

(
    train_score,
    train_results,
) = (
    evaluate_parameters(
        dataset=TRAIN_DATASET,
        params=best_params,
    )
)


# ======================================================
# IMPORTANT:
# Evaluate ON UNSEEN validation backgrounds.
# ======================================================

(
    validation_score,
    validation_results,
) = (
    evaluate_parameters(
        dataset=(
            VALIDATION_DATASET
        ),
        params=best_params,
    )
)


print(
    "\n\n========================================"
)

print(
    "TRAIN SCORE"
)

print(
    "========================================"
)

print(
    round(
        train_score,
        4,
    )
)


print(
    "\n========================================"
)

print(
    "VALIDATION SCORE"
)

print(
    "========================================"
)

print(
    round(
        validation_score,
        4,
    )
)


# ======================================================
# SUMMARY FUNCTION
# ======================================================

def print_summary(
    results,
    name,
):

    print(
        "\n\n========================================"
    )

    print(
        name
    )

    print(
        "========================================"
    )

    summary = (
        results
        .groupby(
            "true_repeats"
        )
        .agg(
            examples=(
                "true_repeats",
                "size",
            ),
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
            mean_score=(
                "score",
                "mean",
            ),
            mean_timing_error=(
                "timing_error",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        summary.to_string(
            index=False
        )
    )


print_summary(
    train_results,
    "TRAIN SUMMARY",
)

print_summary(
    validation_results,
    "VALIDATION SUMMARY",
)

print(
    "\n\n========================================"
)

print(
    "VALIDATION SUMMARY BY SUBJECT"
)

print(
    "========================================"
)

subject_summary = (
    validation_results
    .groupby(
        [
            "subject_id",
            "true_repeats",
        ]
    )
    .agg(
        examples=(
            "true_repeats",
            "size",
        ),

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

        mean_f1=(
            "f1",
            "mean",
        ),

        mean_timing_error=(
            "timing_error",
            "mean",
        ),
    )
    .reset_index()
)

print(
    subject_summary.to_string(
        index=False
    )
)


# ======================================================
# SAVE PARAMETERS
# ======================================================

output_path = (
    "validation/"
    "repetition_consistency/"
    "best_repetition_params.json"
)

with open(
    output_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        best_params,
        file,
        indent=4,
    )


print(
    "\nBest parameters saved to:"
)

print(
    output_path
)
