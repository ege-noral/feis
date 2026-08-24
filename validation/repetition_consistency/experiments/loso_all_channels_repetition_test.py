import gc
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from scipy.stats import wilcoxon

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
# MULTI-CHANNEL LOSO REPETITION DETECTION EXPERIMENT
# ======================================================
#
# PURPOSE
#
# Run the same repetition-detection logic independently on
# EVERY EEG CHANNEL.
#
# For every channel and every held-out subject:
#
#   1) Completely remove the held-out subject.
#
#   2) Use ONLY the other 7 subjects' REAL RESTING EEG
#      as backgrounds.
#
#   3) Create semi-synthetic examples containing:
#          0 / 2 / 3 / 4 / 5 repetitions
#
#   4) Optimize detector parameters with Optuna using only
#      those 7 subjects.
#
#   5) Freeze the best parameters for that channel/fold.
#
#   6) Apply those frozen parameters to the held-out
#      subject's REAL:
#          thinking epochs
#          resting epochs
#
#   7) Record thinking-resting differences.
#
# After all 14 channels x 8 subjects:
#
#   - subject-wise channel table
#   - channel-wise mean/median differences
#   - number of subjects with thinking > resting
#   - Wilcoxon signed-rank p-value across subjects
#   - Benjamini-Hochberg FDR correction across channels
#
# IMPORTANT:
#
# This script is deliberately strict:
# the held-out subject is NEVER used during parameter
# optimization for that fold.
#
# ======================================================


# ======================================================
# GENERAL SETTINGS
# ======================================================

FS = 256

# Use all channels defined by the project.
CHANNELS = list(
    EEG_CHANNELS
)

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

REAL_EPOCH_INDICES = list(
    range(160)
)


# ======================================================
# OPTUNA SETTINGS
# ======================================================

N_TRIALS = 60

RANDOM_SEED = 42


# ======================================================
# OUTPUT / RESUME SETTINGS
# ======================================================

OUTPUT_DIR = Path(
    "validation/"
    "repetition_consistency/"
    "loso_all_channels"
)

# If True, a channel/subject fold that already contains
# completed_summary.json will be loaded and skipped.
#
# This is useful because the full experiment contains:
#
#   14 channels x 8 subjects = 112 LOSO folds
#
# If execution stops halfway, simply run this script again.
RESUME_COMPLETED_FOLDS = True


# ======================================================
# BENJAMINI-HOCHBERG FDR
# ======================================================

def benjamini_hochberg(
    p_values,
):
    """
    Benjamini-Hochberg FDR correction.

    Returns corrected p-values in the original order.
    """

    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    n = len(
        p_values
    )

    if n == 0:
        return np.array(
            [],
            dtype=float,
        )

    order = np.argsort(
        p_values
    )

    ranked = (
        p_values[
            order
        ]
    )

    adjusted_ranked = np.empty(
        n,
        dtype=float,
    )

    running_min = 1.0

    for idx in range(
        n - 1,
        -1,
        -1,
    ):

        rank = (
            idx
            + 1
        )

        adjusted = (
            ranked[
                idx
            ]
            * n
            / rank
        )

        running_min = min(
            running_min,
            adjusted,
        )

        adjusted_ranked[
            idx
        ] = min(
            running_min,
            1.0,
        )

    adjusted_original = np.empty(
        n,
        dtype=float,
    )

    adjusted_original[
        order
    ] = adjusted_ranked

    return adjusted_original


# ======================================================
# MATCH DETECTED PEAKS TO TRUE SEMI-SYNTHETIC EVENTS
# ======================================================

def match_detected_to_true(
    detected_times,
    true_times,
    tolerance_seconds,
):
    """
    Match detected peaks to known semi-synthetic repetition
    centers.

    Each detection can match only one true event.
    Each true event can match only one detection.
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
    Positive examples:
        event-level F1 score.

    Zero-repetition examples:
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
    # Zero-repetition example
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
# CREATE SEMI-SYNTHETIC DATASET FOR ONE CHANNEL/FOLD
# ======================================================

def create_semi_synthetic_dataset(
    channel_name,
    subject_ids,
    background_epoch_indices,
    seed_offset,
):
    """
    Build one channel + LOSO fold training dataset.

    Only RESTING EEG from the 7 training subjects is used.
    """

    channel_idx = (
        EEG_CHANNELS.index(
            channel_name
        )
    )

    dataset = []

    example_number = 0

    for subject_id in (
        subject_ids
    ):

        print(
            f"    Creating semi-synthetic backgrounds "
            f"from subject {subject_id}..."
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
# EVALUATE ONE PARAMETER SET
# ======================================================

def evaluate_parameters(
    dataset,
    params,
):
    """
    Evaluate one parameter combination over the complete
    semi-synthetic training dataset.
    """

    rows = []

    for example in (
        dataset
    ):

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
        # Keep this INSIDE the loop.
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
# OPTUNA SEARCH SPACE
# ======================================================

def suggest_parameters(
    trial,
):
    """
    Same search space as the previous corrected experiment.
    """

    return {

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


# ======================================================
# OPTIMIZE ONE CHANNEL/FOLD
# ======================================================

def optimize_fold(
    train_dataset,
    channel_name,
    held_out_subject,
):
    """
    Optimize only on the 7 training subjects.
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

    channel_seed = sum(
        ord(
            char
        )
        for char in channel_name
    )

    sampler = (
        optuna.samplers.TPESampler(
            seed=(
                RANDOM_SEED
                + channel_seed
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

    return (
        dict(
            study.best_params
        ),
        float(
            study.best_value
        ),
        study,
    )


# ======================================================
# DETECT PEAKS IN ONE REAL EEG EPOCH
# ======================================================

def detect_real_epoch(
    channel_name,
    subject_id,
    condition,
    epoch_idx,
    params,
):
    """
    Apply frozen fold parameters to one real EEG epoch.
    """

    channel_idx = (
        EEG_CHANNELS.index(
            channel_name
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

    return np.asarray(
        result[
            "peak_times"
        ],
        dtype=float,
    )


# ======================================================
# REAL TEST FOR ONE HELD-OUT SUBJECT / CHANNEL
# ======================================================

def test_held_out_subject(
    channel_name,
    held_out_subject,
    params,
):
    """
    Test all real thinking and resting epochs for one
    held-out subject.
    """

    rows = []

    for condition in (
        REAL_CONDITIONS
    ):

        print(
            f"    REAL condition: {condition}"
        )

        processed = 0

        for epoch_idx in (
            REAL_EPOCH_INDICES
        ):

            try:

                peak_times = (
                    detect_real_epoch(
                        channel_name=(
                            channel_name
                        ),
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
                    f"      Skipped "
                    f"channel={channel_name}, "
                    f"subject={held_out_subject}, "
                    f"condition={condition}, "
                    f"epoch={epoch_idx}: "
                    f"{error}"
                )

                continue

            processed += 1

            rows.append({
                "channel": (
                    channel_name
                ),
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
            "      Processed epochs:",
            processed,
        )

    return pd.DataFrame(
        rows
    )


# ======================================================
# SEMI-SYNTHETIC SUMMARY
# ======================================================

def summarize_semi_synthetic(
    results,
):
    return (
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


# ======================================================
# SUMMARIZE ONE CHANNEL / SUBJECT REAL TEST
# ======================================================

def summarize_real_subject(
    channel_name,
    real_results,
    held_out_subject,
    semi_synthetic_train_score,
):
    """
    Produce one row for the final channel x subject matrix.
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

    thinking_ge4 = float(
        (
            thinking[
                "detected_repeats"
            ]
            >= 4
        ).mean()
        * 100.0
    )

    resting_ge4 = float(
        (
            resting[
                "detected_repeats"
            ]
            >= 4
        ).mean()
        * 100.0
    )

    return {
        "channel": (
            channel_name
        ),
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
            thinking_ge4
        ),
        "resting_ge4_percent": (
            resting_ge4
        ),
        "ge4_difference_percentage_points": (
            thinking_ge4
            - resting_ge4
        ),
        "thinking_greater_than_resting": (
            thinking_mean
            > resting_mean
        ),
    }


# ======================================================
# LOAD COMPLETED FOLD
# ======================================================

def load_completed_fold(
    fold_dir,
):
    """
    Load a previously completed channel/subject fold.
    """

    summary_path = (
        fold_dir
        / "completed_summary.json"
    )

    real_path = (
        fold_dir
        / "real_test_results.csv"
    )

    if not (
        summary_path.exists()
        and real_path.exists()
    ):
        return None

    with open(
        summary_path,
        "r",
        encoding="utf-8",
    ) as file:

        fold_row = json.load(
            file
        )

    real_results = pd.read_csv(
        real_path
    )

    return (
        fold_row,
        real_results,
    )


# ======================================================
# RUN ONE CHANNEL / LOSO FOLD
# ======================================================

def run_fold(
    channel_name,
    held_out_subject,
):
    """
    Complete one channel x held-out-subject fold.
    """

    fold_dir = (
        OUTPUT_DIR
        / channel_name
        / (
            f"subject_"
            f"{held_out_subject:02d}"
        )
    )

    fold_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Resume if already complete
    # --------------------------------------------------

    if (
        RESUME_COMPLETED_FOLDS
    ):

        completed = (
            load_completed_fold(
                fold_dir
            )
        )

        if completed is not None:

            print(
                "\n"
                f"  [RESUME] "
                f"{channel_name} / "
                f"subject {held_out_subject} "
                f"already completed."
            )

            return completed

    train_subjects = [
        subject_id
        for subject_id
        in ALL_SUBJECTS
        if subject_id
        != held_out_subject
    ]

    print(
        "\n"
        "  ========================================"
    )

    print(
        f"  CHANNEL {channel_name} | "
        f"HELD-OUT SUBJECT {held_out_subject}"
    )

    print(
        "  ========================================"
    )

    print(
        "  Training subjects:",
        train_subjects,
    )

    # --------------------------------------------------
    # 1. Semi-synthetic dataset
    # --------------------------------------------------

    channel_seed = sum(
        ord(
            char
        )
        for char in channel_name
    )

    seed_offset = (
        channel_seed
        * 1000000
        + held_out_subject
        * 100000
    )

    print(
        "\n  Creating semi-synthetic dataset..."
    )

    train_dataset = (
        create_semi_synthetic_dataset(
            channel_name=(
                channel_name
            ),
            subject_ids=(
                train_subjects
            ),
            background_epoch_indices=(
                BACKGROUND_EPOCHS
            ),
            seed_offset=(
                seed_offset
            ),
        )
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
        "  Training examples:",
        len(
            train_dataset
        ),
    )

    print(
        "  Expected examples:",
        expected_examples,
    )

    if len(
        train_dataset
    ) != expected_examples:

        raise RuntimeError(
            "Semi-synthetic dataset size mismatch."
        )

    # --------------------------------------------------
    # 2. Optuna
    # --------------------------------------------------

    print(
        "\n  Starting Optuna..."
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
            channel_name=(
                channel_name
            ),
            held_out_subject=(
                held_out_subject
            ),
        )
    )

    print(
        "\n  BEST PARAMETERS"
    )

    print(
        "  ----------------------------------------"
    )

    for key, value in (
        best_params.items()
    ):

        print(
            f"  {key}: {value}"
        )

    print(
        "  Best Optuna training score:",
        round(
            best_optuna_score,
            4,
        ),
    )

    # --------------------------------------------------
    # 3. Evaluate best params on training set
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

    # --------------------------------------------------
    # 4. Save optimization results
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
    # 5. Held-out REAL test
    # --------------------------------------------------

    print(
        "\n  Testing held-out REAL EEG..."
    )

    real_results = (
        test_held_out_subject(
            channel_name=(
                channel_name
            ),
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
            f"No real results for "
            f"{channel_name}, "
            f"subject {held_out_subject}."
        )

    real_results.to_csv(
        fold_dir
        / "real_test_results.csv",
        index=False,
    )

    fold_row = (
        summarize_real_subject(
            channel_name=(
                channel_name
            ),
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

    # Save completion marker only AFTER everything succeeded.
    with open(
        fold_dir
        / "completed_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            fold_row,
            file,
            indent=4,
        )

    print(
        "\n  REAL RESULT"
    )

    print(
        "  ----------------------------------------"
    )

    print(
        "  Thinking mean:",
        round(
            fold_row[
                "thinking_mean_detected"
            ],
            4,
        ),
    )

    print(
        "  Resting mean:",
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

    del train_dataset

    del semi_train_results

    gc.collect()

    return (
        fold_row,
        real_results,
    )


# ======================================================
# CHANNEL-LEVEL STATISTICS
# ======================================================

def build_channel_summary(
    subject_summary,
):
    """
    Summarize the 8 held-out subjects for each channel.

    Statistical unit = SUBJECT, not epoch.

    This is important because treating all epochs as
    independent would artificially inflate sample size.
    """

    rows = []

    for channel_name in (
        CHANNELS
    ):

        channel_df = (
            subject_summary[
                subject_summary[
                    "channel"
                ]
                == channel_name
            ]
            .sort_values(
                "subject_id"
            )
        )

        differences = np.asarray(
            channel_df[
                "mean_difference_thinking_minus_resting"
            ],
            dtype=float,
        )

        ge4_differences = np.asarray(
            channel_df[
                "ge4_difference_percentage_points"
            ],
            dtype=float,
        )

        positive_subjects = int(
            np.sum(
                differences
                > 0
            )
        )

        negative_subjects = int(
            np.sum(
                differences
                < 0
            )
        )

        tied_subjects = int(
            np.sum(
                differences
                == 0
            )
        )

        # ----------------------------------------------
        # Wilcoxon across SUBJECT-LEVEL differences
        # ----------------------------------------------

        if np.allclose(
            differences,
            0.0,
        ):

            wilcoxon_statistic = 0.0

            p_two_sided = 1.0

            p_greater = 1.0

        else:

            try:

                test_two = wilcoxon(
                    differences,
                    alternative="two-sided",
                    zero_method="wilcox",
                    correction=False,
                    method="auto",
                )

                wilcoxon_statistic = float(
                    test_two.statistic
                )

                p_two_sided = float(
                    test_two.pvalue
                )

            except ValueError:

                wilcoxon_statistic = np.nan

                p_two_sided = 1.0

            try:

                test_greater = wilcoxon(
                    differences,
                    alternative="greater",
                    zero_method="wilcox",
                    correction=False,
                    method="auto",
                )

                p_greater = float(
                    test_greater.pvalue
                )

            except ValueError:

                p_greater = 1.0

        rows.append({
            "channel": (
                channel_name
            ),
            "subjects": (
                len(
                    channel_df
                )
            ),
            "positive_subjects": (
                positive_subjects
            ),
            "negative_subjects": (
                negative_subjects
            ),
            "tied_subjects": (
                tied_subjects
            ),
            "mean_difference": float(
                np.mean(
                    differences
                )
            ),
            "median_difference": float(
                np.median(
                    differences
                )
            ),
            "std_difference": float(
                np.std(
                    differences,
                    ddof=1,
                )
            ),
            "mean_ge4_difference_pp": float(
                np.mean(
                    ge4_differences
                )
            ),
            "median_ge4_difference_pp": float(
                np.median(
                    ge4_differences
                )
            ),
            "wilcoxon_statistic": (
                wilcoxon_statistic
            ),
            "p_two_sided": (
                p_two_sided
            ),
            "p_greater_thinking_gt_resting": (
                p_greater
            ),
            "mean_semi_synthetic_train_score": float(
                channel_df[
                    "semi_synthetic_train_score"
                ].mean()
            ),
        })

    summary = pd.DataFrame(
        rows
    )

    summary[
        "p_two_sided_fdr_bh"
    ] = (
        benjamini_hochberg(
            summary[
                "p_two_sided"
            ].values
        )
    )

    summary[
        "p_greater_fdr_bh"
    ] = (
        benjamini_hochberg(
            summary[
                "p_greater_thinking_gt_resting"
            ].values
        )
    )

    summary[
        "significant_two_sided_fdr_0_05"
    ] = (
        summary[
            "p_two_sided_fdr_bh"
        ]
        < 0.05
    )

    summary[
        "significant_greater_fdr_0_05"
    ] = (
        summary[
            "p_greater_fdr_bh"
        ]
        < 0.05
    )

    summary = (
        summary
        .sort_values(
            [
                "p_greater_fdr_bh",
                "mean_difference",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    return summary


# ======================================================
# BUILD SUBJECT x CHANNEL DIFFERENCE MATRIX
# ======================================================

def build_difference_matrix(
    subject_summary,
):
    """
    Rows    = subjects
    Columns = EEG channels
    Values  = thinking mean - resting mean
    """

    matrix = (
        subject_summary
        .pivot(
            index="subject_id",
            columns="channel",
            values=(
                "mean_difference_thinking_minus_resting"
            ),
        )
    )

    # Preserve project channel order.
    matrix = (
        matrix[
            CHANNELS
        ]
    )

    return matrix


# ======================================================
# BUILD >=4 PEAK MATRIX
# ======================================================

def build_ge4_matrix(
    subject_summary,
):
    """
    Rows    = subjects
    Columns = EEG channels
    Values  = percentage-point difference:
              thinking >=4 peak rate
              minus
              resting >=4 peak rate
    """

    matrix = (
        subject_summary
        .pivot(
            index="subject_id",
            columns="channel",
            values=(
                "ge4_difference_percentage_points"
            ),
        )
    )

    matrix = (
        matrix[
            CHANNELS
        ]
    )

    return matrix


# ======================================================
# PRINT FINAL RESULTS
# ======================================================

def print_final_results(
    subject_summary,
    channel_summary,
):
    print(
        "\n\n"
        "========================================"
    )

    print(
        "FINAL CHANNEL RANKING"
    )

    print(
        "========================================"
    )

    display_columns = [
        "channel",
        "positive_subjects",
        "negative_subjects",
        "mean_difference",
        "median_difference",
        "mean_ge4_difference_pp",
        "p_greater_thinking_gt_resting",
        "p_greater_fdr_bh",
        "p_two_sided",
        "p_two_sided_fdr_bh",
        "mean_semi_synthetic_train_score",
    ]

    print(
        channel_summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\n\n"
        "========================================"
    )

    print(
        "SUBJECT x CHANNEL DIFFERENCE MATRIX"
    )

    print(
        "thinking mean detected - resting mean detected"
    )

    print(
        "========================================"
    )

    difference_matrix = (
        build_difference_matrix(
            subject_summary
        )
    )

    print(
        difference_matrix.to_string()
    )

    significant_channels = (
        channel_summary[
            channel_summary[
                "significant_greater_fdr_0_05"
            ]
        ][
            "channel"
        ].tolist()
    )

    print(
        "\n\n"
        "========================================"
    )

    print(
        "FDR RESULT"
    )

    print(
        "========================================"
    )

    if len(
        significant_channels
    ) > 0:

        print(
            "Channels with thinking > resting "
            "after BH-FDR correction:"
        )

        print(
            significant_channels
        )

    else:

        print(
            "No channel survives BH-FDR correction "
            "at alpha = 0.05."
        )


# ======================================================
# SAVE GLOBAL PROGRESS
# ======================================================

def save_global_progress(
    subject_rows,
    real_frames,
):
    """
    Save partial global outputs after every completed fold.
    """

    if len(
        subject_rows
    ) > 0:

        partial_subject = (
            pd.DataFrame(
                subject_rows
            )
        )

        partial_subject.to_csv(
            OUTPUT_DIR
            / "all_channels_subject_summary_partial.csv",
            index=False,
        )

    if len(
        real_frames
    ) > 0:

        partial_real = (
            pd.concat(
                real_frames,
                ignore_index=True,
            )
        )

        partial_real.to_csv(
            OUTPUT_DIR
            / "all_channels_real_epoch_results_partial.csv",
            index=False,
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
        "ALL-CHANNEL LOSO REPETITION EXPERIMENT"
    )

    print(
        "========================================"
    )

    print(
        "Channels:",
        CHANNELS,
    )

    print(
        "Number of channels:",
        len(
            CHANNELS
        ),
    )

    print(
        "Subjects:",
        ALL_SUBJECTS,
    )

    print(
        "Number of LOSO folds:",
        (
            len(
                CHANNELS
            )
            * len(
                ALL_SUBJECTS
            )
        ),
    )

    print(
        "Optuna trials per fold:",
        N_TRIALS,
    )

    print(
        "Semi-synthetic examples per fold:",
        (
            7
            * len(
                BACKGROUND_EPOCHS
            )
            * len(
                REPEAT_COUNTS
            )
        ),
    )

    subject_rows = []

    all_real_frames = []

    # ==================================================
    # CHANNEL LOOP
    # ==================================================

    for channel_number, channel_name in enumerate(
        CHANNELS,
        start=1,
    ):

        print(
            "\n\n"
            "########################################"
        )

        print(
            f"CHANNEL "
            f"{channel_number}/{len(CHANNELS)}: "
            f"{channel_name}"
        )

        print(
            "########################################"
        )

        # ==============================================
        # LOSO SUBJECT LOOP
        # ==============================================

        for held_out_subject in (
            ALL_SUBJECTS
        ):

            (
                fold_row,
                real_results,
            ) = (
                run_fold(
                    channel_name=(
                        channel_name
                    ),
                    held_out_subject=(
                        held_out_subject
                    ),
                )
            )

            subject_rows.append(
                fold_row
            )

            all_real_frames.append(
                real_results
            )

            save_global_progress(
                subject_rows=(
                    subject_rows
                ),
                real_frames=(
                    all_real_frames
                ),
            )

        # ----------------------------------------------
        # Save current channel summary immediately
        # ----------------------------------------------

        current_subject_summary = (
            pd.DataFrame(
                subject_rows
            )
        )

        current_channel_rows = (
            current_subject_summary[
                current_subject_summary[
                    "channel"
                ]
                == channel_name
            ]
        )

        current_channel_rows.to_csv(
            OUTPUT_DIR
            / (
                f"{channel_name}"
                f"_subject_summary.csv"
            ),
            index=False,
        )

        print(
            "\n"
            f"Completed channel: "
            f"{channel_name}"
        )

    # ==================================================
    # FINAL GLOBAL OUTPUTS
    # ==================================================

    subject_summary = (
        pd.DataFrame(
            subject_rows
        )
        .sort_values(
            [
                "channel",
                "subject_id",
            ]
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

    channel_summary = (
        build_channel_summary(
            subject_summary
        )
    )

    difference_matrix = (
        build_difference_matrix(
            subject_summary
        )
    )

    ge4_matrix = (
        build_ge4_matrix(
            subject_summary
        )
    )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    subject_summary.to_csv(
        OUTPUT_DIR
        / "all_channels_subject_summary.csv",
        index=False,
    )

    all_real_results.to_csv(
        OUTPUT_DIR
        / "all_channels_real_epoch_results.csv",
        index=False,
    )

    channel_summary.to_csv(
        OUTPUT_DIR
        / "channel_ranking_and_statistics.csv",
        index=False,
    )

    difference_matrix.to_csv(
        OUTPUT_DIR
        / "subject_by_channel_difference_matrix.csv"
    )

    ge4_matrix.to_csv(
        OUTPUT_DIR
        / "subject_by_channel_ge4_difference_matrix.csv"
    )

    # --------------------------------------------------
    # JSON compact summary
    # --------------------------------------------------

    top_channel_row = (
        channel_summary.iloc[
            0
        ]
    )

    final_json = {
        "channels": (
            CHANNELS
        ),
        "subjects": (
            ALL_SUBJECTS
        ),
        "n_channels": (
            len(
                CHANNELS
            )
        ),
        "n_subjects": (
            len(
                ALL_SUBJECTS
            )
        ),
        "n_total_loso_folds": (
            len(
                CHANNELS
            )
            * len(
                ALL_SUBJECTS
            )
        ),
        "n_trials_per_fold": (
            N_TRIALS
        ),
        "top_ranked_channel": str(
            top_channel_row[
                "channel"
            ]
        ),
        "top_channel_mean_difference": float(
            top_channel_row[
                "mean_difference"
            ]
        ),
        "top_channel_positive_subjects": int(
            top_channel_row[
                "positive_subjects"
            ]
        ),
        "top_channel_one_sided_p": float(
            top_channel_row[
                "p_greater_thinking_gt_resting"
            ]
        ),
        "top_channel_one_sided_fdr_bh": float(
            top_channel_row[
                "p_greater_fdr_bh"
            ]
        ),
        "significant_channels_fdr_greater": (
            channel_summary[
                channel_summary[
                    "significant_greater_fdr_0_05"
                ]
            ][
                "channel"
            ].tolist()
        ),
    }

    with open(
        OUTPUT_DIR
        / "final_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            final_json,
            file,
            indent=4,
        )

    # --------------------------------------------------
    # PRINT
    # --------------------------------------------------

    print_final_results(
        subject_summary=(
            subject_summary
        ),
        channel_summary=(
            channel_summary
        ),
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
