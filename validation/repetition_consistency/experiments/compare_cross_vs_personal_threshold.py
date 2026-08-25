import numpy as np
import pandas as pd
from pathlib import Path

from validation.repetition_consistency.experiments import (
    validate_personal_threshold as personal
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

ALL_SUBJECTS = list(range(1, 9))

CHANNEL_NAME = "FC6"

RESTING_PERCENTILE = 95.0

# Threshold öğrenmek için:
CALIBRATION_EPOCHS = list(range(0, 40))

# Her iki threshold'u da AYNI epoch'larda test edeceğiz:
TEST_EPOCHS = list(range(40, 60))


# ============================================================
# SYNC SETTINGS WITH EXISTING MODULE
# ============================================================

personal.CHANNEL_NAME = CHANNEL_NAME
personal.RESTING_PERCENTILE = RESTING_PERCENTILE
personal.CALIBRATION_EPOCHS = CALIBRATION_EPOCHS
personal.TEST_BACKGROUND_EPOCHS = TEST_EPOCHS


# ============================================================
# CROSS-SUBJECT THRESHOLD
# ============================================================

def learn_cross_subject_threshold(
    test_subject,
):
    """
    Threshold is learned ONLY from the other 7 subjects.

    Important:
    We use the same calibration epoch range (0-39)
    that is used for the personal threshold.
    """

    train_subjects = [
        subject
        for subject in ALL_SUBJECTS
        if subject != test_subject
    ]

    all_values = []

    print(
        f"Learning CROSS-SUBJECT threshold "
        f"for held-out Subject {test_subject}..."
    )

    print(
        f"Training subjects: {train_subjects}"
    )

    for train_subject in train_subjects:

        for epoch_idx in CALIBRATION_EPOCHS:

            signal = personal.get_channel_signal(
                subject_id=train_subject,
                condition="resting",
                epoch_idx=epoch_idx,
            )

            similarity_matrix, centers = (
                personal.compute_similarity(
                    signal
                )
            )

            values = (
                personal.get_distant_similarity_values(
                    similarity_matrix,
                    centers,
                )
            )

            all_values.append(
                values
            )

    all_values = np.concatenate(
        all_values
    )

    threshold = np.percentile(
        all_values,
        RESTING_PERCENTILE,
    )

    return (
        float(threshold),
        all_values,
    )


# ============================================================
# PERSONAL THRESHOLD
# ============================================================

def learn_personal_threshold(
    subject_id,
):
    """
    Threshold is learned ONLY from this subject's
    calibration resting epochs 0-39.
    """

    threshold, values = (
        personal.learn_personal_threshold(
            subject_id=subject_id
        )
    )

    return (
        float(threshold),
        values,
    )


# ============================================================
# METRIC HELPERS
# ============================================================

def initialize_metric_storage():

    return {
        "zero_detected": [],

        "five_detected": [],
        "five_matched": [],
        "five_false_positives": [],
        "five_missed": [],
        "five_precision": [],
        "five_recall": [],
        "five_f1": [],

        "timing_errors": [],
    }


def add_zero_result(
    storage,
    peak_times,
):

    storage[
        "zero_detected"
    ].append(
        len(peak_times)
    )


def add_five_result(
    storage,
    peak_times,
    true_times,
):

    match_result = (
        personal.match_detected_to_true(
            detected_times=peak_times,
            true_times=true_times,
        )
    )

    storage[
        "five_detected"
    ].append(
        len(peak_times)
    )

    storage[
        "five_matched"
    ].append(
        match_result[
            "matched"
        ]
    )

    storage[
        "five_false_positives"
    ].append(
        match_result[
            "false_positives"
        ]
    )

    storage[
        "five_missed"
    ].append(
        match_result[
            "missed"
        ]
    )

    storage[
        "five_precision"
    ].append(
        match_result[
            "precision"
        ]
    )

    storage[
        "five_recall"
    ].append(
        match_result[
            "recall"
        ]
    )

    storage[
        "five_f1"
    ].append(
        match_result[
            "f1"
        ]
    )

    storage[
        "timing_errors"
    ].extend(
        match_result[
            "timing_errors"
        ]
    )


def summarize_metrics(
    storage,
):

    zero_detected = np.asarray(
        storage[
            "zero_detected"
        ],
        dtype=float,
    )

    five_detected = np.asarray(
        storage[
            "five_detected"
        ],
        dtype=float,
    )

    five_matched = np.asarray(
        storage[
            "five_matched"
        ],
        dtype=float,
    )

    five_false_positives = np.asarray(
        storage[
            "five_false_positives"
        ],
        dtype=float,
    )

    five_missed = np.asarray(
        storage[
            "five_missed"
        ],
        dtype=float,
    )

    five_precision = np.asarray(
        storage[
            "five_precision"
        ],
        dtype=float,
    )

    five_recall = np.asarray(
        storage[
            "five_recall"
        ],
        dtype=float,
    )

    five_f1 = np.asarray(
        storage[
            "five_f1"
        ],
        dtype=float,
    )

    # --------------------------------------------------------
    # 0-repeat
    # --------------------------------------------------------

    zero_group_rate = np.mean(
        zero_detected > 0
    )

    zero_mean_detected = np.mean(
        zero_detected
    )

    # --------------------------------------------------------
    # 5-repeat
    # --------------------------------------------------------

    five_mean_detected = np.mean(
        five_detected
    )

    five_mean_matched = np.mean(
        five_matched
    )

    five_mean_false_positives = np.mean(
        five_false_positives
    )

    five_mean_missed = np.mean(
        five_missed
    )

    five_mean_precision = np.mean(
        five_precision
    )

    five_mean_recall = np.mean(
        five_recall
    )

    five_mean_f1 = np.mean(
        five_f1
    )

    # Aggregate precision:
    # total matched / total detected
    if np.sum(
        five_detected
    ) > 0:

        aggregate_precision = (
            np.sum(
                five_matched
            )
            /
            np.sum(
                five_detected
            )
        )

    else:

        aggregate_precision = np.nan

    if storage[
        "timing_errors"
    ]:

        timing_error = np.mean(
            storage[
                "timing_errors"
            ]
        )

    else:

        timing_error = np.nan

    return {
        "zero_group_rate": (
            zero_group_rate
        ),

        "zero_mean_detected": (
            zero_mean_detected
        ),

        "five_mean_detected": (
            five_mean_detected
        ),

        "five_mean_matched": (
            five_mean_matched
        ),

        "five_mean_false_positives": (
            five_mean_false_positives
        ),

        "five_mean_missed": (
            five_mean_missed
        ),

        "five_mean_precision": (
            five_mean_precision
        ),

        "five_aggregate_precision": (
            aggregate_precision
        ),

        "five_mean_recall": (
            five_mean_recall
        ),

        "five_mean_f1": (
            five_mean_f1
        ),

        "five_mean_timing_error": (
            timing_error
        ),
    }


# ============================================================
# SUBJECT EVALUATION
# ============================================================

def evaluate_subject(
    subject_id,
):

    # ========================================================
    # LEARN BOTH THRESHOLDS
    # ========================================================

    cross_threshold, cross_values = (
        learn_cross_subject_threshold(
            test_subject=subject_id
        )
    )

    personal_threshold, personal_values = (
        learn_personal_threshold(
            subject_id=subject_id
        )
    )

    print()
    print(
        f"Cross-subject threshold: "
        f"{cross_threshold:.4f}"
    )

    print(
        f"Personal threshold:      "
        f"{personal_threshold:.4f}"
    )

    print(
        "Threshold difference "
        "(personal - cross): "
        f"{personal_threshold - cross_threshold:+.4f}"
    )

    print(
        "Cross calibration similarity count:",
        len(cross_values),
    )

    print(
        "Personal calibration similarity count:",
        len(personal_values),
    )

    # ========================================================
    # STORAGE
    # ========================================================

    cross_storage = (
        initialize_metric_storage()
    )

    personal_storage = (
        initialize_metric_storage()
    )

    # ========================================================
    # IMPORTANT:
    #
    # Both methods are evaluated on EXACTLY THE SAME:
    #
    # - background EEG
    # - synthetic signal
    # - injected event positions
    #
    # ========================================================

    for test_number, epoch_idx in enumerate(
        TEST_EPOCHS
    ):

        # ----------------------------------------------------
        # SAME REAL RESTING BACKGROUND
        # ----------------------------------------------------

        background = (
            personal.get_channel_signal(
                subject_id=subject_id,
                condition="resting",
                epoch_idx=epoch_idx,
            )
        )

        # ====================================================
        # 0 REPEAT
        # ====================================================

        cross_zero_peaks = (
            personal.detect_repetitions(
                signal=background,
                similarity_threshold=(
                    cross_threshold
                ),
            )
        )

        personal_zero_peaks = (
            personal.detect_repetitions(
                signal=background,
                similarity_threshold=(
                    personal_threshold
                ),
            )
        )

        add_zero_result(
            storage=cross_storage,
            peak_times=cross_zero_peaks,
        )

        add_zero_result(
            storage=personal_storage,
            peak_times=personal_zero_peaks,
        )

        # ====================================================
        # CREATE ONE SYNTHETIC SIGNAL
        #
        # VERY IMPORTANT:
        # We create it ONCE and give the SAME signal to
        # both threshold methods.
        # ====================================================

        synthetic_signal, event_info = (
            personal.create_semi_synthetic_epoch(
                background_signal=background,
                fs=personal.FS,
                n_repeats=5,
                duration_range=(
                    personal.MOTIF_DURATION_RANGE
                ),
                motif_std_ratio=(
                    personal.MOTIF_STD_RATIO
                ),
                amplitude_jitter=(
                    personal.AMPLITUDE_JITTER
                ),
                min_gap_seconds=(
                    personal.MIN_GAP_SECONDS
                ),
                edge_margin_seconds=(
                    personal.EDGE_MARGIN_SECONDS
                ),
                seed=(
                    subject_id
                    * 1000
                    + test_number
                ),
            )
        )

        true_times = (
            personal.extract_event_centers(
                event_info
            )
        )

        # ====================================================
        # 5 REPEATS - CROSS
        # ====================================================

        cross_five_peaks = (
            personal.detect_repetitions(
                signal=synthetic_signal,
                similarity_threshold=(
                    cross_threshold
                ),
            )
        )

        add_five_result(
            storage=cross_storage,
            peak_times=cross_five_peaks,
            true_times=true_times,
        )

        # ====================================================
        # 5 REPEATS - PERSONAL
        # ====================================================

        personal_five_peaks = (
            personal.detect_repetitions(
                signal=synthetic_signal,
                similarity_threshold=(
                    personal_threshold
                ),
            )
        )

        add_five_result(
            storage=personal_storage,
            peak_times=personal_five_peaks,
            true_times=true_times,
        )

    # ========================================================
    # SUMMARIZE
    # ========================================================

    cross_summary = (
        summarize_metrics(
            cross_storage
        )
    )

    personal_summary = (
        summarize_metrics(
            personal_storage
        )
    )

    return {
        "subject_id": subject_id,

        "cross_threshold": (
            cross_threshold
        ),

        "personal_threshold": (
            personal_threshold
        ),

        "threshold_difference": (
            personal_threshold
            - cross_threshold
        ),

        # ----------------------------------------------------
        # CROSS
        # ----------------------------------------------------

        "cross_zero_group_rate": (
            cross_summary[
                "zero_group_rate"
            ]
        ),

        "cross_zero_mean_detected": (
            cross_summary[
                "zero_mean_detected"
            ]
        ),

        "cross_five_mean_detected": (
            cross_summary[
                "five_mean_detected"
            ]
        ),

        "cross_five_mean_matched": (
            cross_summary[
                "five_mean_matched"
            ]
        ),

        "cross_five_mean_false_positives": (
            cross_summary[
                "five_mean_false_positives"
            ]
        ),

        "cross_five_aggregate_precision": (
            cross_summary[
                "five_aggregate_precision"
            ]
        ),

        "cross_five_mean_recall": (
            cross_summary[
                "five_mean_recall"
            ]
        ),

        "cross_five_mean_f1": (
            cross_summary[
                "five_mean_f1"
            ]
        ),

        # ----------------------------------------------------
        # PERSONAL
        # ----------------------------------------------------

        "personal_zero_group_rate": (
            personal_summary[
                "zero_group_rate"
            ]
        ),

        "personal_zero_mean_detected": (
            personal_summary[
                "zero_mean_detected"
            ]
        ),

        "personal_five_mean_detected": (
            personal_summary[
                "five_mean_detected"
            ]
        ),

        "personal_five_mean_matched": (
            personal_summary[
                "five_mean_matched"
            ]
        ),

        "personal_five_mean_false_positives": (
            personal_summary[
                "five_mean_false_positives"
            ]
        ),

        "personal_five_aggregate_precision": (
            personal_summary[
                "five_aggregate_precision"
            ]
        ),

        "personal_five_mean_recall": (
            personal_summary[
                "five_mean_recall"
            ]
        ),

        "personal_five_mean_f1": (
            personal_summary[
                "five_mean_f1"
            ]
        ),

        # ----------------------------------------------------
        # DIFFERENCES
        #
        # Positive zero_reduction = PERSONAL is better.
        #
        # Positive matched_change = PERSONAL detects more
        # true motifs.
        # ----------------------------------------------------

        "zero_mean_detected_reduction": (
            cross_summary[
                "zero_mean_detected"
            ]
            -
            personal_summary[
                "zero_mean_detected"
            ]
        ),

        "five_matched_change": (
            personal_summary[
                "five_mean_matched"
            ]
            -
            cross_summary[
                "five_mean_matched"
            ]
        ),

        "five_recall_change": (
            personal_summary[
                "five_mean_recall"
            ]
            -
            cross_summary[
                "five_mean_recall"
            ]
        ),

        "five_f1_change": (
            personal_summary[
                "five_mean_f1"
            ]
            -
            cross_summary[
                "five_mean_f1"
            ]
        ),
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 80)
    print(
        "CROSS-SUBJECT VS PERSONAL RESTING THRESHOLD"
    )
    print("=" * 80)

    print(
        f"Channel: {CHANNEL_NAME}"
    )

    print(
        f"Resting percentile: "
        f"{RESTING_PERCENTILE}"
    )

    print(
        f"Calibration epochs: "
        f"{CALIBRATION_EPOCHS[0]}-"
        f"{CALIBRATION_EPOCHS[-1]}"
    )

    print(
        f"Test epochs: "
        f"{TEST_EPOCHS[0]}-"
        f"{TEST_EPOCHS[-1]}"
    )

    print(
        f"Minimum neighbors: "
        f"{personal.MIN_NEIGHBORS_PER_NODE}"
    )

    all_results = []

    # ========================================================
    # SUBJECT LOOP
    # ========================================================

    for subject_id in ALL_SUBJECTS:

        print()
        print()
        print("=" * 80)

        print(
            f"SUBJECT {subject_id}"
        )

        print("=" * 80)

        result = evaluate_subject(
            subject_id=subject_id
        )

        all_results.append(
            result
        )

        # ====================================================
        # PRINT SIDE BY SIDE
        # ====================================================

        print()
        print("-" * 80)

        print(
            "0 REPEAT COMPARISON"
        )

        print("-" * 80)

        print(
            f"Cross-subject mean detected: "
            f"{result['cross_zero_mean_detected']:.3f}"
        )

        print(
            f"Personal mean detected:      "
            f"{result['personal_zero_mean_detected']:.3f}"
        )

        print(
            f"Reduction with personal:     "
            f"{result['zero_mean_detected_reduction']:+.3f}"
        )

        print()
        print("-" * 80)

        print(
            "5 REPEAT COMPARISON"
        )

        print("-" * 80)

        print(
            f"Cross matched:    "
            f"{result['cross_five_mean_matched']:.3f}"
        )

        print(
            f"Personal matched: "
            f"{result['personal_five_mean_matched']:.3f}"
        )

        print(
            f"Matched change:   "
            f"{result['five_matched_change']:+.3f}"
        )

        print()

        print(
            f"Cross recall:     "
            f"{result['cross_five_mean_recall']:.3f}"
        )

        print(
            f"Personal recall:  "
            f"{result['personal_five_mean_recall']:.3f}"
        )

        print()

        print(
            f"Cross F1:         "
            f"{result['cross_five_mean_f1']:.3f}"
        )

        print(
            f"Personal F1:      "
            f"{result['personal_five_mean_f1']:.3f}"
        )

    # ========================================================
    # DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    # ========================================================
    # FINAL SUBJECT TABLE
    # ========================================================

    print()
    print()
    print("=" * 100)

    print(
        "FINAL SUBJECT COMPARISON"
    )

    print("=" * 100)

    comparison_columns = [
        "subject_id",

        "cross_threshold",
        "personal_threshold",

        "cross_zero_mean_detected",
        "personal_zero_mean_detected",
        "zero_mean_detected_reduction",

        "cross_five_mean_matched",
        "personal_five_mean_matched",
        "five_matched_change",

        "cross_five_mean_recall",
        "personal_five_mean_recall",

        "cross_five_mean_f1",
        "personal_five_mean_f1",
    ]

    print(
        results_df[
            comparison_columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # OVERALL COMPARISON
    # ========================================================

    print()
    print("=" * 80)

    print(
        "OVERALL COMPARISON"
    )

    print("=" * 80)

    print()
    print("0 REPEAT")

    print(
        "Cross-subject mean detected:",
        f"{results_df['cross_zero_mean_detected'].mean():.3f}",
    )

    print(
        "Personal mean detected:",
        f"{results_df['personal_zero_mean_detected'].mean():.3f}",
    )

    print(
        "Mean reduction:",
        f"{results_df['zero_mean_detected_reduction'].mean():+.3f}",
    )

    print()
    print("5 REPEATS")

    print(
        "Cross mean matched:",
        f"{results_df['cross_five_mean_matched'].mean():.3f}",
    )

    print(
        "Personal mean matched:",
        f"{results_df['personal_five_mean_matched'].mean():.3f}",
    )

    print(
        "Cross mean recall:",
        f"{results_df['cross_five_mean_recall'].mean():.3f}",
    )

    print(
        "Personal mean recall:",
        f"{results_df['personal_five_mean_recall'].mean():.3f}",
    )

    print(
        "Cross mean F1:",
        f"{results_df['cross_five_mean_f1'].mean():.3f}",
    )

    print(
        "Personal mean F1:",
        f"{results_df['personal_five_mean_f1'].mean():.3f}",
    )

    # ========================================================
    # SAVE
    # ========================================================

    output_directory = Path(
        "validation/repetition_consistency/archive"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / "cross_vs_personal_threshold_fc6.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Saved to:\n{output_path}"
    )

    print()
    print("=" * 80)

    print("DONE")

    print("=" * 80)