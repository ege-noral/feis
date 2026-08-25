import numpy as np
import pandas as pd
from pathlib import Path

from constants import EEG_CHANNELS
from data_loader import get_one_epoch

from preprocessing.demean import demean

from quality_control.repetition_consistency.morlet_tfr import (
    compute_morlet_tfr,
)

from quality_control.repetition_consistency.self_similarity import (
    compute_tfr_self_similarity,
)

from quality_control.repetition_consistency.semi_synthetic import (
    create_semi_synthetic_epoch,
)

from quality_control.repetition_consistency.detectors.motif_group_detector import (
    detect_motif_groups,
)


# ============================================================
# FIXED PARAMETERS
# ============================================================

ALL_SUBJECTS = list(range(1, 9))

CHANNEL_NAME = "FC6"

FS = 256

RESTING_PERCENTILE = 95.0

WINDOW_SECONDS = 0.40
STEP_SECONDS = 0.15

MIN_TIME_SEPARATION_SECONDS = 0.50

MIN_GROUP_SIZE = 3
MAX_GROUP_SIZE = 5

MIN_GROUP_DENSITY = 0.0
MIN_NEIGHBORS_PER_NODE = 2

MERGE_TOLERANCE_SECONDS = 0.20

MATCH_TOLERANCE_SECONDS = 0.35


# ============================================================
# CALIBRATION / TEST SPLIT
# ============================================================

CALIBRATION_EPOCHS = list(range(0, 40))

TEST_BACKGROUND_EPOCHS = list(range(40, 60))


# ============================================================
# SEMI-SYNTHETIC PARAMETERS
# ============================================================

MOTIF_DURATION_RANGE = (0.40, 0.80)

MOTIF_STD_RATIO = 0.50

AMPLITUDE_JITTER = (0.80, 1.20)

MIN_GAP_SECONDS = 0.10

EDGE_MARGIN_SECONDS = 0.10


# ============================================================
# HELPERS
# ============================================================

def get_channel_signal(
    subject_id,
    condition,
    epoch_idx,
):
    eeg = get_one_epoch(
        subject_id=subject_id,
        condition=condition,
        epoch_idx=epoch_idx,
    )

    channel_idx = EEG_CHANNELS.index(
        CHANNEL_NAME
    )

    signal = eeg[channel_idx]

    return np.asarray(
        signal,
        dtype=float,
    )


def compute_similarity(signal):

    signal = demean(
        signal
    )

    _, power = compute_morlet_tfr(
        signal,
        fs=FS,
    )

    similarity_matrix, centers = (
        compute_tfr_self_similarity(
            power,
            fs=FS,
            window_seconds=WINDOW_SECONDS,
            step_seconds=STEP_SECONDS,
        )
    )

    return (
        similarity_matrix,
        centers,
    )


def get_distant_similarity_values(
    similarity_matrix,
    centers,
):

    values = []

    n_windows = len(
        centers
    )

    for i in range(n_windows):

        for j in range(
            i + 1,
            n_windows,
        ):

            time_difference = abs(
                centers[j]
                - centers[i]
            )

            if (
                time_difference
                < MIN_TIME_SEPARATION_SECONDS
            ):
                continue

            values.append(
                similarity_matrix[i, j]
            )

    return np.asarray(
        values,
        dtype=float,
    )


def learn_personal_threshold(
    subject_id,
):

    all_values = []

    print(
        f"Learning personal resting threshold "
        f"for Subject {subject_id}..."
    )

    for epoch_idx in CALIBRATION_EPOCHS:

        signal = get_channel_signal(
            subject_id=subject_id,
            condition="resting",
            epoch_idx=epoch_idx,
        )

        similarity_matrix, centers = (
            compute_similarity(
                signal
            )
        )

        values = (
            get_distant_similarity_values(
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


def detect_repetitions(
    signal,
    similarity_threshold,
):

    similarity_matrix, centers = (
        compute_similarity(
            signal
        )
    )

    result = detect_motif_groups(
        similarity_matrix=similarity_matrix,
        window_centers_seconds=centers,
        similarity_threshold=similarity_threshold,
        min_time_separation_seconds=(
            MIN_TIME_SEPARATION_SECONDS
        ),
        min_group_size=MIN_GROUP_SIZE,
        max_group_size=MAX_GROUP_SIZE,
        min_group_density=MIN_GROUP_DENSITY,
        min_neighbors_per_node=(
            MIN_NEIGHBORS_PER_NODE
        ),
        merge_tolerance_seconds=(
            MERGE_TOLERANCE_SECONDS
        ),
    )

    peak_times = result.get(
        "peak_times",
        [],
    )

    peak_times = np.asarray(
        peak_times,
        dtype=float,
    )

    return peak_times


def extract_event_centers(
    event_info,
):

    centers = []

    for event in event_info:

        if "center_seconds" in event:
            center = event[
                "center_seconds"
            ]

        elif "center" in event:
            center = event[
                "center"
            ]

        elif "center_time" in event:
            center = event[
                "center_time"
            ]

        else:
            raise KeyError(
                "Could not find event center "
                "in semi-synthetic event info."
            )

        center = float(
            center
        )

        # Safety:
        # if center appears to be a sample index,
        # convert it to seconds.
        if center > 5.0:
            center = (
                center / FS
            )

        centers.append(
            center
        )

    return np.asarray(
        centers,
        dtype=float,
    )


def match_detected_to_true(
    detected_times,
    true_times,
):

    detected_times = np.asarray(
        detected_times,
        dtype=float,
    )

    true_times = np.asarray(
        true_times,
        dtype=float,
    )

    unmatched_true = set(
        range(
            len(true_times)
        )
    )

    matched = 0

    timing_errors = []

    for detected_time in detected_times:

        candidates = []

        for true_idx in unmatched_true:

            error = abs(
                detected_time
                - true_times[true_idx]
            )

            if (
                error
                <= MATCH_TOLERANCE_SECONDS
            ):
                candidates.append(
                    (
                        error,
                        true_idx,
                    )
                )

        if not candidates:
            continue

        error, best_true_idx = min(
            candidates,
            key=lambda item: item[0],
        )

        unmatched_true.remove(
            best_true_idx
        )

        matched += 1

        timing_errors.append(
            error
        )

    false_positives = (
        len(detected_times)
        - matched
    )

    missed = (
        len(true_times)
        - matched
    )

    if len(detected_times) > 0:

        precision = (
            matched
            / len(detected_times)
        )

    else:

        precision = 0.0

    if len(true_times) > 0:

        recall = (
            matched
            / len(true_times)
        )

    else:

        recall = 0.0

    if precision + recall > 0:

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

    return {
        "matched": matched,
        "false_positives": false_positives,
        "missed": missed,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "timing_errors": timing_errors,
    }


# ============================================================
# SUBJECT EVALUATION
# ============================================================

def evaluate_subject(
    subject_id,
):

    threshold, similarity_values = (
        learn_personal_threshold(
            subject_id
        )
    )

    print(
        f"Personal threshold: "
        f"{threshold:.4f}"
    )

    print(
        "Calibration similarity count:",
        len(similarity_values),
    )

    zero_detected = []

    five_detected = []

    five_matched = []

    five_false_positives = []

    five_missed = []

    five_precision = []

    five_recall = []

    five_f1 = []

    all_timing_errors = []

    # ========================================================
    # HELD-OUT TEST BACKGROUNDS
    # ========================================================

    for test_number, epoch_idx in enumerate(
        TEST_BACKGROUND_EPOCHS
    ):

        background = get_channel_signal(
            subject_id=subject_id,
            condition="resting",
            epoch_idx=epoch_idx,
        )

        # ====================================================
        # 0 REPEAT
        # ====================================================

        zero_peaks = detect_repetitions(
            signal=background,
            similarity_threshold=threshold,
        )

        zero_detected.append(
            len(zero_peaks)
        )

        # ====================================================
        # 5 REPEATS
        # ====================================================

        synthetic_signal, event_info = (
            create_semi_synthetic_epoch(
                background_signal=background,
                fs=FS,
                n_repeats=5,
                duration_range=(
                    MOTIF_DURATION_RANGE
                ),
                motif_std_ratio=(
                    MOTIF_STD_RATIO
                ),
                amplitude_jitter=(
                    AMPLITUDE_JITTER
                ),
                min_gap_seconds=(
                    MIN_GAP_SECONDS
                ),
                edge_margin_seconds=(
                    EDGE_MARGIN_SECONDS
                ),
                seed=(
                    subject_id
                    * 1000
                    + test_number
                ),
            )
        )

        true_times = (
            extract_event_centers(
                event_info
            )
        )

        five_peaks = (
            detect_repetitions(
                signal=synthetic_signal,
                similarity_threshold=(
                    threshold
                ),
            )
        )

        match_result = (
            match_detected_to_true(
                detected_times=five_peaks,
                true_times=true_times,
            )
        )

        five_detected.append(
            len(five_peaks)
        )

        five_matched.append(
            match_result[
                "matched"
            ]
        )

        five_false_positives.append(
            match_result[
                "false_positives"
            ]
        )

        five_missed.append(
            match_result[
                "missed"
            ]
        )

        five_precision.append(
            match_result[
                "precision"
            ]
        )

        five_recall.append(
            match_result[
                "recall"
            ]
        )

        five_f1.append(
            match_result[
                "f1"
            ]
        )

        all_timing_errors.extend(
            match_result[
                "timing_errors"
            ]
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    zero_detected = np.asarray(
        zero_detected
    )

    five_detected = np.asarray(
        five_detected
    )

    five_matched = np.asarray(
        five_matched
    )

    five_false_positives = np.asarray(
        five_false_positives
    )

    five_missed = np.asarray(
        five_missed
    )

    five_precision = np.asarray(
        five_precision
    )

    five_recall = np.asarray(
        five_recall
    )

    five_f1 = np.asarray(
        five_f1
    )

    zero_group_rate = np.mean(
        zero_detected > 0
    )

    zero_mean_detected = np.mean(
        zero_detected
    )

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

    if np.sum(
        five_detected
    ) > 0:

        aggregate_precision = (
            np.sum(
                five_matched
            )
            / np.sum(
                five_detected
            )
        )

    else:

        aggregate_precision = np.nan

    if all_timing_errors:

        mean_timing_error = np.mean(
            all_timing_errors
        )

    else:

        mean_timing_error = np.nan

    return {
        "subject_id": subject_id,
        "personal_threshold": threshold,

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
            mean_timing_error
        ),
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "SUBJECT-SPECIFIC RESTING THRESHOLD VALIDATION"
    )

    print("=" * 70)

    print(
        f"Channel: {CHANNEL_NAME}"
    )

    print(
        f"Resting percentile: "
        f"{RESTING_PERCENTILE}"
    )

    print(
        "Minimum neighbors:",
        MIN_NEIGHBORS_PER_NODE,
    )

    print(
        "Calibration epochs:",
        (
            f"{CALIBRATION_EPOCHS[0]}"
            f"-{CALIBRATION_EPOCHS[-1]}"
        ),
    )

    print(
        "Held-out test epochs:",
        (
            f"{TEST_BACKGROUND_EPOCHS[0]}"
            f"-{TEST_BACKGROUND_EPOCHS[-1]}"
        ),
    )

    all_results = []

    for subject_id in ALL_SUBJECTS:

        print()
        print()
        print("=" * 70)

        print(
            f"SUBJECT {subject_id}"
        )

        print("=" * 70)

        result = evaluate_subject(
            subject_id
        )

        all_results.append(
            result
        )

        print()
        print("0 REPEAT")

        print(
            "Group rate:",
            f"{result['zero_group_rate']:.3f}",
        )

        print(
            "Mean detected:",
            f"{result['zero_mean_detected']:.3f}",
        )

        print()
        print("5 REPEATS")

        print(
            "Mean detected:",
            f"{result['five_mean_detected']:.3f}",
        )

        print(
            "Mean matched:",
            f"{result['five_mean_matched']:.3f}",
        )

        print(
            "False positives:",
            f"{result['five_mean_false_positives']:.3f}",
        )

        print(
            "Missed:",
            f"{result['five_mean_missed']:.3f}",
        )

        print(
            "Aggregate precision:",
            f"{result['five_aggregate_precision']:.3f}",
        )

        print(
            "Recall:",
            f"{result['five_mean_recall']:.3f}",
        )

        print(
            "F1:",
            f"{result['five_mean_f1']:.3f}",
        )

        print(
            "Timing error:",
            f"{result['five_mean_timing_error']:.3f} s",
        )

    # ========================================================
    # DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    print()
    print()
    print("=" * 70)

    print(
        "SUBJECT-SPECIFIC THRESHOLD SUMMARY"
    )

    print("=" * 70)

    columns = [
        "subject_id",
        "personal_threshold",
        "zero_group_rate",
        "zero_mean_detected",
        "five_mean_detected",
        "five_mean_matched",
        "five_mean_false_positives",
        "five_aggregate_precision",
        "five_mean_recall",
        "five_mean_f1",
        "five_mean_timing_error",
    ]

    print(
        results_df[
            columns
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 70)

    print(
        "OVERALL SUMMARY"
    )

    print("=" * 70)

    print(
        "Mean zero group rate:",
        f"{results_df['zero_group_rate'].mean():.3f}",
    )

    print(
        "Mean zero detected:",
        f"{results_df['zero_mean_detected'].mean():.3f}",
    )

    print(
        "Mean five detected:",
        f"{results_df['five_mean_detected'].mean():.3f}",
    )

    print(
        "Mean five matched:",
        f"{results_df['five_mean_matched'].mean():.3f}",
    )

    print(
        "Mean five false positives:",
        f"{results_df['five_mean_false_positives'].mean():.3f}",
    )

    print(
        "Mean five recall:",
        f"{results_df['five_mean_recall'].mean():.3f}",
    )

    print(
        "Mean five F1:",
        f"{results_df['five_mean_f1'].mean():.3f}",
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
        / "motif_group_fc6_personal_threshold_sanity.csv"
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
    print("=" * 70)

    print("DONE")

    print("=" * 70)