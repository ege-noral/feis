import numpy as np
import pandas as pd
from pathlib import Path

from validation.repetition_consistency.experiments import (
    validate_personal_threshold as base
)

from quality_control.repetition_consistency.detectors.motif_group_detector import (
    detect_motif_groups,
)


# ============================================================
# FIXED SETTINGS
# ============================================================

ALL_SUBJECTS = list(range(1, 9))

CHANNEL_NAME = "FC6"

CALIBRATION_EPOCHS = list(range(0, 40))
TEST_EPOCHS = list(range(40, 60))

# Detector structure
MIN_GROUP_SIZE = 4
MAX_GROUP_SIZE = 5
MIN_NEIGHBORS_PER_NODE = 2
MIN_GROUP_DENSITY = 0.0

# Desired epoch-level false-positive rate
TARGET_CALIBRATION_GROUP_RATE = 0.10

# Binary-search precision
N_BINARY_SEARCH_STEPS = 20


# ============================================================
# SYNC BASE MODULE SETTINGS
# ============================================================

base.CHANNEL_NAME = CHANNEL_NAME

base.CALIBRATION_EPOCHS = CALIBRATION_EPOCHS
base.TEST_BACKGROUND_EPOCHS = TEST_EPOCHS

base.MIN_GROUP_SIZE = MIN_GROUP_SIZE
base.MAX_GROUP_SIZE = MAX_GROUP_SIZE

base.MIN_NEIGHBORS_PER_NODE = MIN_NEIGHBORS_PER_NODE
base.MIN_GROUP_DENSITY = MIN_GROUP_DENSITY


# ============================================================
# DETECTION FROM PRECOMPUTED SIMILARITY
# ============================================================

def detect_from_similarity(
    similarity_matrix,
    centers,
    threshold,
):

    result = detect_motif_groups(
        similarity_matrix=similarity_matrix,
        window_centers_seconds=centers,
        similarity_threshold=threshold,
        min_time_separation_seconds=(
            base.MIN_TIME_SEPARATION_SECONDS
        ),
        min_group_size=MIN_GROUP_SIZE,
        max_group_size=MAX_GROUP_SIZE,
        min_group_density=MIN_GROUP_DENSITY,
        min_neighbors_per_node=(
            MIN_NEIGHBORS_PER_NODE
        ),
        merge_tolerance_seconds=(
            base.MERGE_TOLERANCE_SECONDS
        ),
    )

    peak_times = np.asarray(
        result.get("peak_times", []),
        dtype=float,
    )

    return peak_times


# ============================================================
# PRECOMPUTE CALIBRATION RESTING
# ============================================================

def prepare_calibration_data(
    subject_id,
):

    calibration_data = []
    all_similarity_values = []

    print(
        f"Preparing calibration resting "
        f"for Subject {subject_id}..."
    )

    for epoch_idx in CALIBRATION_EPOCHS:

        signal = base.get_channel_signal(
            subject_id=subject_id,
            condition="resting",
            epoch_idx=epoch_idx,
        )

        similarity_matrix, centers = (
            base.compute_similarity(
                signal
            )
        )

        calibration_data.append(
            (
                similarity_matrix,
                centers,
            )
        )

        distant_values = (
            base.get_distant_similarity_values(
                similarity_matrix,
                centers,
            )
        )

        all_similarity_values.append(
            distant_values
        )

    all_similarity_values = np.concatenate(
        all_similarity_values
    )

    return (
        calibration_data,
        all_similarity_values,
    )


# ============================================================
# CALIBRATION GROUP RATE
# ============================================================

def compute_calibration_group_rate(
    calibration_data,
    threshold,
):

    epochs_with_group = 0

    for similarity_matrix, centers in calibration_data:

        peak_times = detect_from_similarity(
            similarity_matrix=similarity_matrix,
            centers=centers,
            threshold=threshold,
        )

        if len(peak_times) > 0:

            epochs_with_group += 1

    group_rate = (
        epochs_with_group
        / len(calibration_data)
    )

    return group_rate


# ============================================================
# LEARN DETECTOR-LEVEL THRESHOLD
# ============================================================

def learn_detector_calibrated_threshold(
    calibration_data,
    all_similarity_values,
):

    # Search range comes from the subject's own
    # calibration similarity distribution.

    low = float(
        np.min(
            all_similarity_values
        )
    )

    high = float(
        np.max(
            all_similarity_values
        )
        + 1e-6
    )

    # At very high threshold, group rate should be zero.
    # We search for the LOWEST threshold that satisfies
    # the desired resting group rate.

    for _ in range(
        N_BINARY_SEARCH_STEPS
    ):

        middle = (
            low + high
        ) / 2.0

        group_rate = (
            compute_calibration_group_rate(
                calibration_data=(
                    calibration_data
                ),
                threshold=middle,
            )
        )

        if (
            group_rate
            <= TARGET_CALIBRATION_GROUP_RATE
        ):
            # Threshold works.
            # Try lowering it while still
            # controlling false positives.
            high = middle

        else:
            # Too many resting groups.
            # Need a stricter threshold.
            low = middle

    threshold = high

    final_group_rate = (
        compute_calibration_group_rate(
            calibration_data=calibration_data,
            threshold=threshold,
        )
    )

    return (
        float(threshold),
        float(final_group_rate),
    )


# ============================================================
# METRIC STORAGE
# ============================================================

def create_storage():

    return {
        "zero_detected": [],

        "five_detected": [],
        "five_matched": [],
        "five_false_positives": [],
        "five_missed": [],
        "five_recall": [],
        "five_f1": [],

        "timing_errors": [],
    }


def add_zero(
    storage,
    peak_times,
):

    storage[
        "zero_detected"
    ].append(
        len(peak_times)
    )


def add_five(
    storage,
    peak_times,
    true_times,
):

    match = (
        base.match_detected_to_true(
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
        match["matched"]
    )

    storage[
        "five_false_positives"
    ].append(
        match["false_positives"]
    )

    storage[
        "five_missed"
    ].append(
        match["missed"]
    )

    storage[
        "five_recall"
    ].append(
        match["recall"]
    )

    storage[
        "five_f1"
    ].append(
        match["f1"]
    )

    storage[
        "timing_errors"
    ].extend(
        match["timing_errors"]
    )


# ============================================================
# SUMMARY
# ============================================================

def summarize(
    storage,
):

    zero_detected = np.asarray(
        storage["zero_detected"],
        dtype=float,
    )

    five_detected = np.asarray(
        storage["five_detected"],
        dtype=float,
    )

    five_matched = np.asarray(
        storage["five_matched"],
        dtype=float,
    )

    five_fp = np.asarray(
        storage["five_false_positives"],
        dtype=float,
    )

    five_missed = np.asarray(
        storage["five_missed"],
        dtype=float,
    )

    five_recall = np.asarray(
        storage["five_recall"],
        dtype=float,
    )

    five_f1 = np.asarray(
        storage["five_f1"],
        dtype=float,
    )

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

        timing_error = float(
            np.mean(
                storage[
                    "timing_errors"
                ]
            )
        )

    else:

        timing_error = np.nan

    return {
        "zero_group_rate": float(
            np.mean(
                zero_detected > 0
            )
        ),

        "zero_mean_detected": float(
            np.mean(
                zero_detected
            )
        ),

        "five_mean_detected": float(
            np.mean(
                five_detected
            )
        ),

        "five_mean_matched": float(
            np.mean(
                five_matched
            )
        ),

        "five_mean_false_positives": float(
            np.mean(
                five_fp
            )
        ),

        "five_mean_missed": float(
            np.mean(
                five_missed
            )
        ),

        "five_aggregate_precision": float(
            aggregate_precision
        ),

        "five_mean_recall": float(
            np.mean(
                five_recall
            )
        ),

        "five_mean_f1": float(
            np.mean(
                five_f1
            )
        ),

        "five_mean_timing_error": (
            timing_error
        ),
    }


# ============================================================
# TEST SUBJECT
# ============================================================

def evaluate_subject(
    subject_id,
):

    # ========================================================
    # CALIBRATION
    # ========================================================

    (
        calibration_data,
        all_similarity_values,
    ) = prepare_calibration_data(
        subject_id
    )

    # Old personal 95th percentile threshold
    percentile_threshold = float(
        np.percentile(
            all_similarity_values,
            95.0,
        )
    )

    # New detector-level threshold
    (
        detector_threshold,
        calibration_group_rate,
    ) = learn_detector_calibrated_threshold(
        calibration_data=calibration_data,
        all_similarity_values=(
            all_similarity_values
        ),
    )

    print()
    print(
        f"Old personal 95% threshold: "
        f"{percentile_threshold:.4f}"
    )

    print(
        f"Detector-calibrated threshold: "
        f"{detector_threshold:.4f}"
    )

    print(
        "Calibration resting group rate:",
        f"{calibration_group_rate:.3f}",
    )

    # ========================================================
    # TEST
    # ========================================================

    storage = create_storage()

    for test_number, epoch_idx in enumerate(
        TEST_EPOCHS
    ):

        # ----------------------------------------------------
        # REAL HELD-OUT RESTING
        # ----------------------------------------------------

        background = (
            base.get_channel_signal(
                subject_id=subject_id,
                condition="resting",
                epoch_idx=epoch_idx,
            )
        )

        zero_peaks = (
            base.detect_repetitions(
                signal=background,
                similarity_threshold=(
                    detector_threshold
                ),
            )
        )

        add_zero(
            storage=storage,
            peak_times=zero_peaks,
        )

        # ----------------------------------------------------
        # SAME BACKGROUND + 5 SYNTHETIC MOTIFS
        # ----------------------------------------------------

        synthetic_signal, event_info = (
            base.create_semi_synthetic_epoch(
                background_signal=background,
                fs=base.FS,
                n_repeats=5,
                duration_range=(
                    base.MOTIF_DURATION_RANGE
                ),
                motif_std_ratio=(
                    base.MOTIF_STD_RATIO
                ),
                amplitude_jitter=(
                    base.AMPLITUDE_JITTER
                ),
                min_gap_seconds=(
                    base.MIN_GAP_SECONDS
                ),
                edge_margin_seconds=(
                    base.EDGE_MARGIN_SECONDS
                ),
                seed=(
                    subject_id
                    * 1000
                    + test_number
                ),
            )
        )

        true_times = (
            base.extract_event_centers(
                event_info
            )
        )

        five_peaks = (
            base.detect_repetitions(
                signal=synthetic_signal,
                similarity_threshold=(
                    detector_threshold
                ),
            )
        )

        add_five(
            storage=storage,
            peak_times=five_peaks,
            true_times=true_times,
        )

    summary = summarize(
        storage
    )

    return {
        "subject_id": subject_id,

        "percentile95_threshold": (
            percentile_threshold
        ),

        "detector_threshold": (
            detector_threshold
        ),

        "calibration_group_rate": (
            calibration_group_rate
        ),

        **summary,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 75)

    print(
        "DETECTOR-LEVEL SUBJECT-SPECIFIC "
        "RESTING CALIBRATION"
    )

    print("=" * 75)

    print(
        f"Channel: {CHANNEL_NAME}"
    )

    print(
        f"Minimum group size: "
        f"{MIN_GROUP_SIZE}"
    )

    print(
        f"Minimum neighbors: "
        f"{MIN_NEIGHBORS_PER_NODE}"
    )

    print(
        "Target calibration group rate:",
        TARGET_CALIBRATION_GROUP_RATE,
    )

    print(
        f"Calibration epochs: "
        f"{CALIBRATION_EPOCHS[0]}-"
        f"{CALIBRATION_EPOCHS[-1]}"
    )

    print(
        f"Held-out test epochs: "
        f"{TEST_EPOCHS[0]}-"
        f"{TEST_EPOCHS[-1]}"
    )

    all_results = []

    for subject_id in ALL_SUBJECTS:

        print()
        print()
        print("=" * 75)

        print(
            f"SUBJECT {subject_id}"
        )

        print("=" * 75)

        result = evaluate_subject(
            subject_id
        )

        all_results.append(
            result
        )

        print()
        print("0 REPEAT")

        print(
            "Held-out group rate:",
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
    # FINAL TABLE
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    print()
    print()
    print("=" * 100)

    print(
        "DETECTOR-CALIBRATED SUMMARY"
    )

    print("=" * 100)

    columns = [
        "subject_id",
        "percentile95_threshold",
        "detector_threshold",
        "calibration_group_rate",

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

    # ========================================================
    # OVERALL
    # ========================================================

    print()
    print("=" * 75)

    print(
        "OVERALL SUMMARY"
    )

    print("=" * 75)

    print(
        "Mean held-out zero group rate:",
        f"{results_df['zero_group_rate'].mean():.3f}",
    )

    print(
        "Mean held-out zero detected:",
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
        / "motif_group_fc6_detector_calibrated_threshold.csv"
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
    print("=" * 75)
    print("DONE")
    print("=" * 75)