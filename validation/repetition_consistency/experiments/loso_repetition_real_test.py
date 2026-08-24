import gc
import json
from pathlib import Path

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
# LOSO REAL THINKING / RESTING REPETITION TEST
# ======================================================
#
# WHAT THIS SCRIPT DOES
#
# For each held-out subject:
#
#   1) Completely removes that subject from parameter search.
#
#   2) Uses ONLY the other subjects' real RESTING EEG as
#      backgrounds for semi-synthetic repetition examples.
#
#   3) Runs Optuna on those semi-synthetic examples.
#
#   4) Freezes the best parameters for that fold.
#
#   5) Applies those frozen parameters to the held-out
#      subject's REAL:
#
#          thinking epochs
#          resting epochs
#
#   6) Compares detected repetition counts.
#
# Therefore each subject is tested with parameters that were
# selected without using ANY EEG from that subject.
#
# IMPORTANT:
# Real FEIS thinking epochs contain 5 imagined repetitions,
# but we do not have exact within-epoch event-center timestamps.
# So real-data evaluation here is NOT event-level F1.
#
# Instead we compare:
#
#   - mean detected repetitions
#   - median detected repetitions
#   - distribution of detected counts
#   - percentage of epochs with >= 4 detected peaks
#
# across REAL thinking and REAL resting.
#
# ======================================================


# ======================================================
# GENERAL SETTINGS
# ======================================================

FS = 256

CHANNEL_NAME = "P7"

ALL_SUBJECTS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
]


# ======================================================
# SEMI-SYNTHETIC TRAINING SETTINGS
# ======================================================

# From each training subject, use resting epochs 0-9 as
# semi-synthetic backgrounds.
BACKGROUND_EPOCHS = list(
    range(10)
)

REPEAT_COUNTS = [
    0,
    2,
    3,
    4,
    5,
]

DURATION_RANGE = (
    0.40,
    0.80,
)

MOTIF_STD_RATIO = 0.50

MIN_GAP_SECONDS = 0.10

EDGE_MARGIN_SECONDS = 0.10

MATCH_TOLERANCE_SECONDS = 0.35


# ======================================================
# REAL TEST SETTINGS
# ======================================================

REAL_CONDITIONS = [
    "thinking",
    "resting",
]

# FEIS epochs used in this project:
# 160 epochs per condition.
REAL_EPOCH_INDICES = list(
    range(160)
)


# ======================================================
# OPTUNA SETTINGS
# ======================================================

N_TRIALS = 60

RANDOM_SEED = 42


# ======================================================
# OUTPUT SETTINGS
# ======================================================

OUTPUT_DIR = Path(
    "validation/"
    "repetition_consistency/"
    "loso"
)


# ======================================================
# MATCH DETECTED PEAKS TO TRUE SEMI-SYNTHETIC EVENTS
# ======================================================

def match_detected_to_true(
    detected_times,
    true_times,
    tolerance_seconds,
):
    """
    Match detected peak times to known semi-synthetic event
    centers.

    Each detected event can match at most one true event.
    Each true event can match at most one detection.
    Closest valid pairs are matched first.
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
        len(
            detected_times
        )
        - matched_count
    )

    missed_count = (
        len(
            true_times
        )
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
# SCORE ONE SEMI-SYNTHETIC EXAMPLE
# ======================================================

def calculate_example_score(
    detected_times,
    true_times,
):
    """
    Score one semi-synthetic example.

    Positive examples:
        event-level F1.

    Zero-repetition examples:
        false detections are penalized:
            0 detections -> 1.00
            1 detection  -> 0.75
            2 detections -> 0.50
            3 detections -> 0.25
            4+           -> 0.00
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
            true_times=(
                true_times
            ),
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
    # No true repetition exists
    # --------------------------------------------------

    if len(
        true_times
    ) == 0:

        if len(
            detected_times
        ) == 0:

            score = 1.0

        else:

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
            if len(
                detected_times
            ) == 0
            else 0.0
        )

        recall = 1.0

        f1 = score

    # --------------------------------------------------
    # Positive example
    # --------------------------------------------------

    else:

        if len(
            detected_times
        ) > 0:

            precision = (
                matched
                / len(
                    detected_times
                )
            )

        else:

            precision = 0.0

        recall = (
            matched
            / len(
                true_times
            )
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
# CREATE SEMI-SYNTHETIC TRAINING DATASET
# ======================================================

def create_semi_synthetic_dataset(
    subject_ids,
    background_epoch_indices,
    seed_offset,
):
    """
    Build one LOSO fold's semi-synthetic training dataset.

    Only RESTING EEG from training subjects is used as
    background.

    Every background produces:
        0 / 2 / 3 / 4 / 5 repetition versions.

    TFR is calculated once here and stored so Optuna does not
    need to recalculate Morlet TFR for every parameter trial.
    """

    channel_idx = (
        EEG_CHANNELS.index(
            CHANNEL_NAME
        )
    )

    dataset = []

    example_number = 0

    for subject_id in subject_ids:

        print(
            f"  Creating backgrounds from "
            f"subject {subject_id}..."
        )

        for epoch_idx in (
            background_epoch_indices
        ):

            resting_epoch = (
                get_one_epoch(
                    subject_id=(
                        subject_id
                    ),
                    condition="resting",
                    epoch_idx=(
                        epoch_idx
                    ),
                )
            )

            background_signal = (
                resting_epoch[
                    channel_idx
                ]
            )

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
                        n_repeats=(
                            n_repeats
                        ),
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

                true_times = np.array(
                    [
                        event[
                            "center_seconds"
                        ]
                        for event
                        in event_info
                    ],
                    dtype=float,
                )

                signal = demean(
                    synthetic_signal
                )

                (
                    freqs,
                    power,
                ) = (
                    compute_morlet_tfr(
                        signal,
                        fs=FS,
                    )
                )

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
# EVALUATE ONE PARAMETER SET ON SEMI-SYNTHETIC DATA
# ======================================================

def evaluate_parameters(
    dataset,
    params,
):
    """
    Evaluate one complete parameter combination on the
    semi-synthetic training dataset.
    """

    rows = []

    for example in dataset:

        power = (
            example[
                "power"
            ]
        )

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

        min_lag_seconds = (
            params[
                "window_seconds"
            ]
            + params[
                "extra_lag_seconds"
            ]
        )

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

        detected_times = np.asarray(
            result[
                "peak_times"
            ],
            dtype=float,
        )

        true_times = np.asarray(
            example[
                "true_times"
            ],
            dtype=float,
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

        # IMPORTANT:
        # This append MUST remain INSIDE the example loop.
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

    mean_score = float(
        results[
            "score"
        ].mean()
    )

    return (
        mean_score,
        results,
    )


# ======================================================
# OPTUNA PARAMETER SPACE
# ======================================================

def suggest_parameters(
    trial,
):
    """
    Same parameter search space used in the previous
    corrected parameter-search experiment.
    """

    params = {

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

        "step_seconds":
            trial.suggest_categorical(
                "step_seconds",
                [
                    0.05,
                    0.10,
                    0.15,
                ],
            ),

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

        "top_k":
            trial.suggest_int(
                "top_k",
                1,
                5,
            ),

        "smoothing_windows":
            trial.suggest_categorical(
                "smoothing_windows",
                [
                    1,
                    3,
                    5,
                ],
            ),

        "peak_min_height":
            trial.suggest_float(
                "peak_min_height",
                0.10,
                0.80,
            ),

        "peak_prominence":
            trial.suggest_float(
                "peak_prominence",
                0.00,
                0.25,
            ),

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

    return params


# ======================================================
# OPTIMIZE ONE LOSO FOLD
# ======================================================

def optimize_fold(
    train_dataset,
    held_out_subject,
):
    """
    Optimize detector parameters using only the subjects
    other than held_out_subject.
    """

    def objective(
        trial,
    ):

        params = (
            suggest_parameters(
                trial
            )
        )

        mean_score, _ = (
            evaluate_parameters(
                dataset=(
                    train_dataset
                ),
                params=params,
            )
        )

        return mean_score

    sampler = (
        optuna.samplers.TPESampler(
            seed=(
                RANDOM_SEED
                + held_out_subject
            )
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

    best_params = dict(
        study.best_params
    )

    best_score = float(
        study.best_value
    )

    return (
        best_params,
        best_score,
        study,
    )


# ======================================================
# DETECT PEAKS IN ONE REAL EEG EPOCH
# ======================================================

def detect_real_epoch(
    subject_id,
    condition,
    epoch_idx,
    params,
):
    """
    Apply one fold's already-frozen detector to one REAL
    EEG epoch from the held-out subject.
    """

    channel_idx = (
        EEG_CHANNELS.index(
            CHANNEL_NAME
        )
    )

    epoch = (
        get_one_epoch(
            subject_id=(
                subject_id
            ),
            condition=(
                condition
            ),
            epoch_idx=(
                epoch_idx
            ),
        )
    )

    signal = (
        epoch[
            channel_idx
        ]
    )

    signal = demean(
        signal
    )

    (
        freqs,
        power,
    ) = (
        compute_morlet_tfr(
            signal,
            fs=FS,
        )
    )

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

    min_lag_seconds = (
        params[
            "window_seconds"
        ]
        + params[
            "extra_lag_seconds"
        ]
    )

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

    peak_times = np.asarray(
        result[
            "peak_times"
        ],
        dtype=float,
    )

    return peak_times


# ======================================================
# TEST ONE HELD-OUT SUBJECT ON REAL DATA
# ======================================================

def test_held_out_subject(
    held_out_subject,
    params,
):
    """
    Run the frozen fold parameters on all real thinking and
    resting epochs of the held-out subject.
    """

    rows = []

    for condition in (
        REAL_CONDITIONS
    ):

        print(
            f"  REAL condition: {condition}"
        )

        processed = 0

        for epoch_idx in (
            REAL_EPOCH_INDICES
        ):

            try:

                peak_times = (
                    detect_real_epoch(
                        subject_id=(
                            held_out_subject
                        ),
                        condition=(
                            condition
                        ),
                        epoch_idx=(
                            epoch_idx
                        ),
                        params=params,
                    )
                )

            except (
                IndexError,
                KeyError,
            ):

                break

            except Exception as error:

                print(
                    f"    Skipped "
                    f"subject={held_out_subject}, "
                    f"condition={condition}, "
                    f"epoch={epoch_idx}: "
                    f"{error}"
                )

                continue

            processed += 1

            rows.append({
                "subject_id": (
                    held_out_subject
                ),
                "condition": (
                    condition
                ),
                "epoch_idx": (
                    epoch_idx
                ),
                "detected_repeats": (
                    len(
                        peak_times
                    )
                ),
                "peak_times": (
                    ";".join(
                        f"{time:.4f}"
                        for time
                        in peak_times
                    )
                ),
            })

        print(
            f"    Processed epochs: "
            f"{processed}"
        )

    return pd.DataFrame(
        rows
    )


# ======================================================
# SUMMARIZE SEMI-SYNTHETIC TRAINING RESULT
# ======================================================

def summarize_semi_synthetic(
    results,
):
    """
    Return one row per true repetition count.
    """

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

    return summary


# ======================================================
# SUMMARIZE ONE SUBJECT'S REAL THINKING / RESTING RESULT
# ======================================================

def summarize_real_subject(
    real_results,
    held_out_subject,
    semi_synthetic_train_score,
):
    """
    Produce one LOSO result row for one held-out subject.
    """

    thinking = (
        real_results[
            real_results[
                "condition"
            ]
            == "thinking"
        ]
    )

    resting = (
        real_results[
            real_results[
                "condition"
            ]
            == "resting"
        ]
    )

    thinking_mean = float(
        thinking[
            "detected_repeats"
        ].mean()
    )

    resting_mean = float(
        resting[
            "detected_repeats"
        ].mean()
    )

    thinking_median = float(
        thinking[
            "detected_repeats"
        ].median()
    )

    resting_median = float(
        resting[
            "detected_repeats"
        ].median()
    )

    thinking_four_plus = float(
        (
            thinking[
                "detected_repeats"
            ]
            >= 4
        ).mean()
        * 100.0
    )

    resting_four_plus = float(
        (
            resting[
                "detected_repeats"
            ]
            >= 4
        ).mean()
        * 100.0
    )

    return {
        "subject_id": (
            held_out_subject
        ),
        "semi_synthetic_train_score": (
            semi_synthetic_train_score
        ),
        "thinking_examples": (
            len(
                thinking
            )
        ),
        "resting_examples": (
            len(
                resting
            )
        ),
        "thinking_mean_detected": (
            thinking_mean
        ),
        "resting_mean_detected": (
            resting_mean
        ),
        "mean_difference_thinking_minus_resting": (
            thinking_mean
            - resting_mean
        ),
        "thinking_median_detected": (
            thinking_median
        ),
        "resting_median_detected": (
            resting_median
        ),
        "thinking_ge4_percent": (
            thinking_four_plus
        ),
        "resting_ge4_percent": (
            resting_four_plus
        ),
        "ge4_difference_percentage_points": (
            thinking_four_plus
            - resting_four_plus
        ),
        "thinking_greater_than_resting": (
            thinking_mean
            > resting_mean
        ),
    }


# ======================================================
# PRINT ONE FOLD'S REAL RESULT
# ======================================================

def print_fold_real_summary(
    fold_row,
):
    """
    Print the immediately useful result after each fold.
    """

    print(
        "\n  REAL HELD-OUT RESULT"
    )

    print(
        "  ----------------------------------------"
    )

    print(
        "  Thinking mean detected:",
        round(
            fold_row[
                "thinking_mean_detected"
            ],
            4,
        ),
    )

    print(
        "  Resting mean detected:",
        round(
            fold_row[
                "resting_mean_detected"
            ],
            4,
        ),
    )

    print(
        "  Difference:",
        round(
            fold_row[
                "mean_difference_thinking_minus_resting"
            ],
            4,
        ),
    )

    print(
        "  Thinking >=4 peaks:",
        f"{fold_row['thinking_ge4_percent']:.2f}%",
    )

    print(
        "  Resting >=4 peaks:",
        f"{fold_row['resting_ge4_percent']:.2f}%",
    )


# ======================================================
# PRINT FINAL LOSO SUMMARY
# ======================================================

def print_final_summary(
    subject_summary,
    all_real_results,
):
    """
    Print final subject-wise and overall descriptive results.
    """

    print(
        "\n\n"
        "========================================"
    )

    print(
        "FINAL LOSO SUBJECT SUMMARY"
    )

    print(
        "========================================"
    )

    columns = [
        "subject_id",
        "semi_synthetic_train_score",
        "thinking_mean_detected",
        "resting_mean_detected",
        "mean_difference_thinking_minus_resting",
        "thinking_ge4_percent",
        "resting_ge4_percent",
        "ge4_difference_percentage_points",
        "thinking_greater_than_resting",
    ]

    print(
        subject_summary[
            columns
        ].to_string(
            index=False
        )
    )

    positive_subjects = int(
        subject_summary[
            "thinking_greater_than_resting"
        ].sum()
    )

    total_subjects = int(
        len(
            subject_summary
        )
    )

    mean_subject_difference = float(
        subject_summary[
            "mean_difference_thinking_minus_resting"
        ].mean()
    )

    median_subject_difference = float(
        subject_summary[
            "mean_difference_thinking_minus_resting"
        ].median()
    )

    mean_ge4_difference = float(
        subject_summary[
            "ge4_difference_percentage_points"
        ].mean()
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "ACROSS-SUBJECT LOSO SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        "Subjects with thinking mean > resting mean:",
        f"{positive_subjects}/{total_subjects}",
    )

    print(
        "Mean subject-level difference "
        "(thinking - resting):",
        round(
            mean_subject_difference,
            4,
        ),
    )

    print(
        "Median subject-level difference "
        "(thinking - resting):",
        round(
            median_subject_difference,
            4,
        ),
    )

    print(
        "Mean >=4-peak percentage-point difference:",
        round(
            mean_ge4_difference,
            4,
        ),
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "POOLED REAL THINKING VS RESTING SUMMARY"
    )

    print(
        "========================================"
    )

    pooled_summary = (
        all_real_results
        .groupby(
            "condition"
        )
        .agg(
            examples=(
                "detected_repeats",
                "size",
            ),
            mean_detected=(
                "detected_repeats",
                "mean",
            ),
            median_detected=(
                "detected_repeats",
                "median",
            ),
            std_detected=(
                "detected_repeats",
                "std",
            ),
            min_detected=(
                "detected_repeats",
                "min",
            ),
            max_detected=(
                "detected_repeats",
                "max",
            ),
        )
        .reset_index()
    )

    print(
        pooled_summary.to_string(
            index=False
        )
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "POOLED DETECTED-REPEAT DISTRIBUTION"
    )

    print(
        "========================================"
    )

    distribution = (
        all_real_results
        .groupby(
            [
                "condition",
                "detected_repeats",
            ]
        )
        .size()
        .reset_index(
            name="epochs"
        )
    )

    totals = (
        all_real_results
        .groupby(
            "condition"
        )
        .size()
        .to_dict()
    )

    distribution[
        "percentage"
    ] = (
        distribution.apply(
            lambda row:
                100.0
                * row[
                    "epochs"
                ]
                / totals[
                    row[
                        "condition"
                    ]
                ],
            axis=1,
        )
    )

    print(
        distribution.to_string(
            index=False
        )
    )


# ======================================================
# RUN ONE LOSO FOLD
# ======================================================

def run_fold(
    held_out_subject,
):
    """
    Complete one LOSO fold:
        train on 7 subjects' semi-synthetic data
        test on held-out subject's real EEG
    """

    fold_dir = (
        OUTPUT_DIR
        / (
            f"subject_"
            f"{held_out_subject:02d}"
        )
    )

    fold_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_subjects = [
        subject_id
        for subject_id
        in ALL_SUBJECTS
        if subject_id
        != held_out_subject
    ]

    print(
        "\n\n"
        "========================================"
    )

    print(
        f"LOSO FOLD: HELD-OUT SUBJECT "
        f"{held_out_subject}"
    )

    print(
        "========================================"
    )

    print(
        "Training subjects:",
        train_subjects,
    )

    print(
        "Held-out subject:",
        held_out_subject,
    )

    # --------------------------------------------------
    # 1. Semi-synthetic training data
    # --------------------------------------------------

    print(
        "\nCreating semi-synthetic training dataset..."
    )

    train_dataset = (
        create_semi_synthetic_dataset(
            subject_ids=(
                train_subjects
            ),
            background_epoch_indices=(
                BACKGROUND_EPOCHS
            ),
            seed_offset=(
                held_out_subject
                * 100000
            ),
        )
    )

    print(
        "Semi-synthetic training examples:",
        len(
            train_dataset
        ),
    )

    expected_examples = (
        len(
            train_subjects
        )
        * len(
            BACKGROUND_EPOCHS
        )
        * len(
            REPEAT_COUNTS
        )
    )

    print(
        "Expected examples:",
        expected_examples,
    )

    if len(
        train_dataset
    ) != expected_examples:

        raise RuntimeError(
            "Semi-synthetic dataset size is not as expected."
        )

    # --------------------------------------------------
    # 2. Optuna
    # --------------------------------------------------

    print(
        "\nStarting Optuna for this LOSO fold..."
    )

    (
        best_params,
        best_optuna_score,
        study,
    ) = (
        optimize_fold(
            train_dataset=(
                train_dataset
            ),
            held_out_subject=(
                held_out_subject
            ),
        )
    )

    print(
        "\nBEST PARAMETERS"
    )

    print(
        "----------------------------------------"
    )

    for key, value in (
        best_params.items()
    ):

        print(
            f"{key}: {value}"
        )

    print(
        "Best Optuna training score:",
        round(
            best_optuna_score,
            4,
        ),
    )

    # --------------------------------------------------
    # 3. Re-evaluate best parameters on complete
    #    semi-synthetic training set
    # --------------------------------------------------

    (
        semi_train_score,
        semi_train_results,
    ) = (
        evaluate_parameters(
            dataset=(
                train_dataset
            ),
            params=(
                best_params
            ),
        )
    )

    semi_summary = (
        summarize_semi_synthetic(
            semi_train_results
        )
    )

    print(
        "\nSEMI-SYNTHETIC TRAIN SUMMARY"
    )

    print(
        "----------------------------------------"
    )

    print(
        semi_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # 4. SAVE fold optimization outputs BEFORE real test
    # --------------------------------------------------

    with open(
        fold_dir
        / "best_params.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            best_params,
            file,
            indent=4,
        )

    study.trials_dataframe().to_csv(
        fold_dir
        / "optuna_trials.csv",
        index=False,
    )

    semi_train_results.to_csv(
        fold_dir
        / "semi_synthetic_train_results.csv",
        index=False,
    )

    semi_summary.to_csv(
        fold_dir
        / "semi_synthetic_train_summary.csv",
        index=False,
    )

    # --------------------------------------------------
    # 5. REAL held-out test
    # --------------------------------------------------

    print(
        "\nTesting held-out subject on REAL EEG..."
    )

    real_results = (
        test_held_out_subject(
            held_out_subject=(
                held_out_subject
            ),
            params=(
                best_params
            ),
        )
    )

    if len(
        real_results
    ) == 0:

        raise RuntimeError(
            f"No real EEG results for "
            f"subject {held_out_subject}."
        )

    real_results.to_csv(
        fold_dir
        / "real_test_results.csv",
        index=False,
    )

    fold_row = (
        summarize_real_subject(
            real_results=(
                real_results
            ),
            held_out_subject=(
                held_out_subject
            ),
            semi_synthetic_train_score=(
                semi_train_score
            ),
        )
    )

    print_fold_real_summary(
        fold_row
    )

    # --------------------------------------------------
    # Clean fold training data before next subject
    # --------------------------------------------------

    del train_dataset

    del semi_train_results

    gc.collect()

    return (
        fold_row,
        real_results,
    )


# ======================================================
# MAIN
# ======================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "LOSO REPETITION DETECTION EXPERIMENT"
    )

    print(
        "========================================"
    )

    print(
        "Channel:",
        CHANNEL_NAME,
    )

    print(
        "Subjects:",
        ALL_SUBJECTS,
    )

    print(
        "Semi-synthetic background epochs "
        "per training subject:",
        len(
            BACKGROUND_EPOCHS
        ),
    )

    print(
        "Optuna trials per fold:",
        N_TRIALS,
    )

    print(
        "Real epochs per condition:",
        len(
            REAL_EPOCH_INDICES
        ),
    )

    subject_rows = []

    all_real_frames = []

    # ==================================================
    # LOSO LOOP
    # ==================================================

    for held_out_subject in (
        ALL_SUBJECTS
    ):

        (
            fold_row,
            real_results,
        ) = (
            run_fold(
                held_out_subject=(
                    held_out_subject
                )
            )
        )

        subject_rows.append(
            fold_row
        )

        all_real_frames.append(
            real_results
        )

        # Save progress after every completed fold.
        partial_subject_summary = (
            pd.DataFrame(
                subject_rows
            )
        )

        partial_subject_summary.to_csv(
            OUTPUT_DIR
            / "loso_subject_summary_partial.csv",
            index=False,
        )

        partial_real_results = (
            pd.concat(
                all_real_frames,
                ignore_index=True,
            )
        )

        partial_real_results.to_csv(
            OUTPUT_DIR
            / "loso_real_epoch_results_partial.csv",
            index=False,
        )

    # ==================================================
    # FINAL RESULTS
    # ==================================================

    subject_summary = (
        pd.DataFrame(
            subject_rows
        )
        .sort_values(
            "subject_id"
        )
        .reset_index(
            drop=True
        )
    )

    all_real_results = (
        pd.concat(
            all_real_frames,
            ignore_index=True,
        )
    )

    subject_summary.to_csv(
        OUTPUT_DIR
        / "loso_subject_summary.csv",
        index=False,
    )

    all_real_results.to_csv(
        OUTPUT_DIR
        / "loso_real_epoch_results.csv",
        index=False,
    )

    print_final_summary(
        subject_summary=(
            subject_summary
        ),
        all_real_results=(
            all_real_results
        ),
    )

    # --------------------------------------------------
    # One compact final JSON too
    # --------------------------------------------------

    positive_subjects = int(
        subject_summary[
            "thinking_greater_than_resting"
        ].sum()
    )

    final_json = {
        "channel": CHANNEL_NAME,
        "subjects": (
            ALL_SUBJECTS
        ),
        "n_subjects": (
            len(
                ALL_SUBJECTS
            )
        ),
        "n_trials_per_fold": (
            N_TRIALS
        ),
        "background_epochs_per_training_subject": (
            len(
                BACKGROUND_EPOCHS
            )
        ),
        "subjects_with_thinking_mean_greater_than_resting": (
            positive_subjects
        ),
        "mean_subject_level_difference": float(
            subject_summary[
                "mean_difference_thinking_minus_resting"
            ].mean()
        ),
        "median_subject_level_difference": float(
            subject_summary[
                "mean_difference_thinking_minus_resting"
            ].median()
        ),
        "mean_ge4_percentage_point_difference": float(
            subject_summary[
                "ge4_difference_percentage_points"
            ].mean()
        ),
    }

    with open(
        OUTPUT_DIR
        / "loso_final_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            final_json,
            file,
            indent=4,
        )

    print(
        "\n\n"
        "========================================"
    )

    print(
        "FILES SAVED TO"
    )

    print(
        "========================================"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
