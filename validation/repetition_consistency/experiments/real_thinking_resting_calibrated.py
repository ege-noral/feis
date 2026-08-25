import numpy as np
import pandas as pd
from pathlib import Path

from validation.repetition_consistency.experiments import (
    validate_detector_calibrated_threshold as calibration
)


# ============================================================
# FIXED SETTINGS
# ============================================================

ALL_SUBJECTS = list(range(1, 9))

CHANNEL_NAME = "FC6"

# Resting'in ilk 40 epoch'u SADECE calibration.
CALIBRATION_EPOCHS = list(range(0, 40))

# Gerçek karşılaştırmada iki condition için
# AYNI epoch index aralığını kullanalım.
TEST_EPOCHS = list(range(40, 160))


# ============================================================
# LOCK DETECTOR SETTINGS
# ============================================================

MIN_GROUP_SIZE = 4
MAX_GROUP_SIZE = 5

MIN_NEIGHBORS_PER_NODE = 2
MIN_GROUP_DENSITY = 0.0

TARGET_CALIBRATION_GROUP_RATE = 0.10


# ============================================================
# SYNC MODULE SETTINGS
# ============================================================

calibration.CHANNEL_NAME = CHANNEL_NAME

calibration.CALIBRATION_EPOCHS = CALIBRATION_EPOCHS

calibration.MIN_GROUP_SIZE = MIN_GROUP_SIZE
calibration.MAX_GROUP_SIZE = MAX_GROUP_SIZE

calibration.MIN_NEIGHBORS_PER_NODE = (
    MIN_NEIGHBORS_PER_NODE
)

calibration.MIN_GROUP_DENSITY = (
    MIN_GROUP_DENSITY
)

calibration.TARGET_CALIBRATION_GROUP_RATE = (
    TARGET_CALIBRATION_GROUP_RATE
)


# Base module also uses these globals.
calibration.base.CHANNEL_NAME = CHANNEL_NAME

calibration.base.CALIBRATION_EPOCHS = (
    CALIBRATION_EPOCHS
)

calibration.base.MIN_GROUP_SIZE = (
    MIN_GROUP_SIZE
)

calibration.base.MAX_GROUP_SIZE = (
    MAX_GROUP_SIZE
)

calibration.base.MIN_NEIGHBORS_PER_NODE = (
    MIN_NEIGHBORS_PER_NODE
)

calibration.base.MIN_GROUP_DENSITY = (
    MIN_GROUP_DENSITY
)


# ============================================================
# REAL CONDITION EVALUATION
# ============================================================

def evaluate_condition(
    subject_id,
    condition,
    threshold,
):

    detected_counts = []

    epoch_rows = []

    for epoch_idx in TEST_EPOCHS:

        signal = (
            calibration.base.get_channel_signal(
                subject_id=subject_id,
                condition=condition,
                epoch_idx=epoch_idx,
            )
        )

        peak_times = (
            calibration.base.detect_repetitions(
                signal=signal,
                similarity_threshold=threshold,
            )
        )

        detected_count = len(
            peak_times
        )

        detected_counts.append(
            detected_count
        )

        epoch_rows.append(
            {
                "subject_id": subject_id,
                "condition": condition,
                "epoch_idx": epoch_idx,
                "detected_repeats": detected_count,
                "peak_times": ";".join(
                    f"{time:.4f}"
                    for time in peak_times
                ),
            }
        )

    detected_counts = np.asarray(
        detected_counts,
        dtype=float,
    )

    summary = {
        "mean_detected": float(
            np.mean(
                detected_counts
            )
        ),

        "median_detected": float(
            np.median(
                detected_counts
            )
        ),

        "std_detected": float(
            np.std(
                detected_counts
            )
        ),

        "group_rate": float(
            np.mean(
                detected_counts > 0
            )
        ),

        "rate_ge_3": float(
            np.mean(
                detected_counts >= 3
            )
        ),

        "rate_ge_4": float(
            np.mean(
                detected_counts >= 4
            )
        ),

        "max_detected": int(
            np.max(
                detected_counts
            )
        ),
    }

    return (
        summary,
        epoch_rows,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 80)

    print(
        "REAL THINKING VS RESTING "
        "- CALIBRATED MOTIF DETECTOR"
    )

    print("=" * 80)

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
        f"Calibration resting epochs: "
        f"{CALIBRATION_EPOCHS[0]}-"
        f"{CALIBRATION_EPOCHS[-1]}"
    )

    print(
        f"Real test epochs: "
        f"{TEST_EPOCHS[0]}-"
        f"{TEST_EPOCHS[-1]}"
    )

    subject_rows = []
    all_epoch_rows = []

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

        # ====================================================
        # 1. PERSONAL RESTING CALIBRATION
        # ====================================================

        (
            calibration_data,
            similarity_values,
        ) = calibration.prepare_calibration_data(
            subject_id=subject_id
        )

        (
            threshold,
            calibration_group_rate,
        ) = (
            calibration.learn_detector_calibrated_threshold(
                calibration_data=calibration_data,
                all_similarity_values=(
                    similarity_values
                ),
            )
        )

        print()
        print(
            f"Personal detector threshold: "
            f"{threshold:.4f}"
        )

        print(
            f"Calibration group rate: "
            f"{calibration_group_rate:.3f}"
        )

        # ====================================================
        # 2. REAL RESTING
        # ====================================================

        resting_summary, resting_rows = (
            evaluate_condition(
                subject_id=subject_id,
                condition="resting",
                threshold=threshold,
            )
        )

        # ====================================================
        # 3. REAL THINKING
        # ====================================================

        thinking_summary, thinking_rows = (
            evaluate_condition(
                subject_id=subject_id,
                condition="thinking",
                threshold=threshold,
            )
        )

        all_epoch_rows.extend(
            resting_rows
        )

        all_epoch_rows.extend(
            thinking_rows
        )

        # ====================================================
        # DIFFERENCE
        # ====================================================

        delta_mean = (
            thinking_summary[
                "mean_detected"
            ]
            -
            resting_summary[
                "mean_detected"
            ]
        )

        # Our exploratory criterion
        criterion_success = (
            thinking_summary[
                "mean_detected"
            ] >= 3.0
            and
            resting_summary[
                "mean_detected"
            ] <= 1.5
            and
            delta_mean >= 1.5
        )

        # Strong criterion
        strong_success = (
            thinking_summary[
                "mean_detected"
            ] >= 4.0
            and
            resting_summary[
                "mean_detected"
            ] <= 1.0
            and
            delta_mean >= 3.0
        )

        # ====================================================
        # PRINT
        # ====================================================

        print()
        print("-" * 80)
        print("REAL RESTING")
        print("-" * 80)

        print(
            "Mean detected:",
            f"{resting_summary['mean_detected']:.3f}",
        )

        print(
            "Median detected:",
            f"{resting_summary['median_detected']:.3f}",
        )

        print(
            "Epochs with motif group:",
            f"{resting_summary['group_rate']:.3f}",
        )

        print(
            "Epochs with >= 3 repeats:",
            f"{resting_summary['rate_ge_3']:.3f}",
        )

        print(
            "Epochs with >= 4 repeats:",
            f"{resting_summary['rate_ge_4']:.3f}",
        )

        print()
        print("-" * 80)
        print("REAL THINKING")
        print("-" * 80)

        print(
            "Mean detected:",
            f"{thinking_summary['mean_detected']:.3f}",
        )

        print(
            "Median detected:",
            f"{thinking_summary['median_detected']:.3f}",
        )

        print(
            "Epochs with motif group:",
            f"{thinking_summary['group_rate']:.3f}",
        )

        print(
            "Epochs with >= 3 repeats:",
            f"{thinking_summary['rate_ge_3']:.3f}",
        )

        print(
            "Epochs with >= 4 repeats:",
            f"{thinking_summary['rate_ge_4']:.3f}",
        )

        print()
        print(
            "THINKING - RESTING mean difference:",
            f"{delta_mean:+.3f}",
        )

        print(
            "Exploratory criterion passed:",
            criterion_success,
        )

        print(
            "Strong criterion passed:",
            strong_success,
        )

        # ====================================================
        # STORE SUBJECT SUMMARY
        # ====================================================

        subject_rows.append(
            {
                "subject_id": subject_id,

                "threshold": threshold,

                "calibration_group_rate": (
                    calibration_group_rate
                ),

                "resting_mean": (
                    resting_summary[
                        "mean_detected"
                    ]
                ),

                "resting_median": (
                    resting_summary[
                        "median_detected"
                    ]
                ),

                "resting_group_rate": (
                    resting_summary[
                        "group_rate"
                    ]
                ),

                "resting_rate_ge_3": (
                    resting_summary[
                        "rate_ge_3"
                    ]
                ),

                "resting_rate_ge_4": (
                    resting_summary[
                        "rate_ge_4"
                    ]
                ),

                "thinking_mean": (
                    thinking_summary[
                        "mean_detected"
                    ]
                ),

                "thinking_median": (
                    thinking_summary[
                        "median_detected"
                    ]
                ),

                "thinking_group_rate": (
                    thinking_summary[
                        "group_rate"
                    ]
                ),

                "thinking_rate_ge_3": (
                    thinking_summary[
                        "rate_ge_3"
                    ]
                ),

                "thinking_rate_ge_4": (
                    thinking_summary[
                        "rate_ge_4"
                    ]
                ),

                "thinking_minus_resting": (
                    delta_mean
                ),

                "criterion_success": (
                    criterion_success
                ),

                "strong_success": (
                    strong_success
                ),
            }
        )

    # ========================================================
    # FINAL TABLES
    # ========================================================

    subject_df = pd.DataFrame(
        subject_rows
    )

    epoch_df = pd.DataFrame(
        all_epoch_rows
    )

    print()
    print()
    print("=" * 100)

    print(
        "REAL SUBJECT SUMMARY"
    )

    print("=" * 100)

    columns = [
        "subject_id",
        "threshold",

        "resting_mean",
        "thinking_mean",
        "thinking_minus_resting",

        "resting_group_rate",
        "thinking_group_rate",

        "resting_rate_ge_3",
        "thinking_rate_ge_3",

        "criterion_success",
        "strong_success",
    ]

    print(
        subject_df[
            columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # OVERALL
    # ========================================================

    print()
    print("=" * 80)
    print("OVERALL REAL DATA SUMMARY")
    print("=" * 80)

    print(
        "Mean resting detected:",
        f"{subject_df['resting_mean'].mean():.3f}",
    )

    print(
        "Mean thinking detected:",
        f"{subject_df['thinking_mean'].mean():.3f}",
    )

    print(
        "Mean thinking-resting difference:",
        f"{subject_df['thinking_minus_resting'].mean():+.3f}",
    )

    print(
        "Subjects with thinking > resting:",
        (
            subject_df[
                "thinking_minus_resting"
            ] > 0
        ).sum(),
        "/",
        len(subject_df),
    )

    print(
        "Subjects passing exploratory criterion:",
        subject_df[
            "criterion_success"
        ].sum(),
        "/",
        len(subject_df),
    )

    print(
        "Subjects passing strong criterion:",
        subject_df[
            "strong_success"
        ].sum(),
        "/",
        len(subject_df),
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

    epoch_path = (
        output_directory
        / "real_fc6_calibrated_epoch_results.csv"
    )

    subject_path = (
        output_directory
        / "real_fc6_calibrated_subject_summary.csv"
    )

    epoch_df.to_csv(
        epoch_path,
        index=False,
    )

    subject_df.to_csv(
        subject_path,
        index=False,
    )

    print()
    print(
        f"Epoch results saved to:\n{epoch_path}"
    )

    print()
    print(
        f"Subject summary saved to:\n{subject_path}"
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)