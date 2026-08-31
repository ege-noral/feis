import numpy as np
import pandas as pd
from pathlib import Path

from constants import EEG_CHANNELS

from validation.repetition_consistency.experiments import (
    validate_detector_calibrated_threshold as calibration
)


# ============================================================
# FIXED EXPERIMENT SETTINGS
# ============================================================

ALL_SUBJECTS = list(range(1, 9))

CHANNELS_TO_TEST = EEG_CHANNELS

CALIBRATION_EPOCHS = list(range(0, 40))
TEST_EPOCHS = list(range(40, 160))


# ============================================================
# LOCKED DETECTOR SETTINGS
# ============================================================

MIN_GROUP_SIZE = 4
MAX_GROUP_SIZE = 5

MIN_NEIGHBORS_PER_NODE = 2
MIN_GROUP_DENSITY = 0.0

TARGET_CALIBRATION_GROUP_RATE = 0.10


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIRECTORY = Path(
    "validation/repetition_consistency/archive"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

SUBJECT_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "real_all_channels_calibrated_subject_summary.csv"
)

EPOCH_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "real_all_channels_calibrated_epoch_results.csv"
)

CHANNEL_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "real_all_channels_calibrated_channel_summary.csv"
)


# ============================================================
# SYNC FIXED DETECTOR SETTINGS
# ============================================================

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
    channel_name,
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
                "channel": channel_name,
                "condition": condition,
                "epoch_idx": epoch_idx,
                "threshold": threshold,
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

        # Since MIN_GROUP_SIZE = 4,
        # group_rate and rate_ge_4
        # should normally be equivalent.
        "rate_ge_4": float(
            np.mean(
                detected_counts >= 4
            )
        ),

        "rate_eq_5": float(
            np.mean(
                detected_counts == 5
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
# SAVE CURRENT PROGRESS
# ============================================================

def save_progress(
    subject_rows,
    epoch_rows,
):

    if subject_rows:

        subject_df = pd.DataFrame(
            subject_rows
        )

        subject_df.to_csv(
            SUBJECT_OUTPUT_PATH,
            index=False,
        )

    if epoch_rows:

        epoch_df = pd.DataFrame(
            epoch_rows
        )

        epoch_df.to_csv(
            EPOCH_OUTPUT_PATH,
            index=False,
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 90)
    print(
        "REAL THINKING VS RESTING "
        "- ALL CHANNELS "
        "- CALIBRATED MOTIF DETECTOR"
    )
    print("=" * 90)

    print(
        f"Subjects: {ALL_SUBJECTS}"
    )

    print(
        f"Channels: {CHANNELS_TO_TEST}"
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

    subject_rows = []
    all_epoch_rows = []

    # ========================================================
    # CHANNEL LOOP
    # ========================================================

    for channel_number, channel_name in enumerate(
        CHANNELS_TO_TEST,
        start=1,
    ):

        print()
        print()
        print("#" * 90)

        print(
            f"CHANNEL "
            f"{channel_number}/{len(CHANNELS_TO_TEST)}: "
            f"{channel_name}"
        )

        print("#" * 90)

        # ----------------------------------------------------
        # VERY IMPORTANT:
        # both modules use CHANNEL_NAME as a global.
        # ----------------------------------------------------

        calibration.CHANNEL_NAME = channel_name
        calibration.base.CHANNEL_NAME = channel_name

        # ====================================================
        # SUBJECT LOOP
        # ====================================================

        for subject_id in ALL_SUBJECTS:

            print()
            print("=" * 80)

            print(
                f"{channel_name} | SUBJECT {subject_id}"
            )

            print("=" * 80)

            # =================================================
            # 1. SUBJECT + CHANNEL SPECIFIC RESTING CALIBRATION
            # =================================================

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
                f"Detector threshold: "
                f"{threshold:.4f}"
            )

            print(
                f"Calibration group rate: "
                f"{calibration_group_rate:.3f}"
            )

            # =================================================
            # 2. REAL RESTING
            # =================================================

            resting_summary, resting_rows = (
                evaluate_condition(
                    subject_id=subject_id,
                    channel_name=channel_name,
                    condition="resting",
                    threshold=threshold,
                )
            )

            # =================================================
            # 3. REAL THINKING
            # =================================================

            thinking_summary, thinking_rows = (
                evaluate_condition(
                    subject_id=subject_id,
                    channel_name=channel_name,
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

            # =================================================
            # 4. DIFFERENCES
            # =================================================

            mean_delta = (
                thinking_summary[
                    "mean_detected"
                ]
                -
                resting_summary[
                    "mean_detected"
                ]
            )

            group_rate_delta = (
                thinking_summary[
                    "group_rate"
                ]
                -
                resting_summary[
                    "group_rate"
                ]
            )

            five_rate_delta = (
                thinking_summary[
                    "rate_eq_5"
                ]
                -
                resting_summary[
                    "rate_eq_5"
                ]
            )

            # =================================================
            # PRINT
            # =================================================

            print()
            print("-" * 80)
            print("REAL RESTING")
            print("-" * 80)

            print(
                "Mean detected:",
                f"{resting_summary['mean_detected']:.3f}",
            )

            print(
                "Median:",
                f"{resting_summary['median_detected']:.3f}",
            )

            print(
                "Motif group rate:",
                f"{resting_summary['group_rate']:.3f}",
            )

            print(
                "Exactly 5 motif rate:",
                f"{resting_summary['rate_eq_5']:.3f}",
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
                "Median:",
                f"{thinking_summary['median_detected']:.3f}",
            )

            print(
                "Motif group rate:",
                f"{thinking_summary['group_rate']:.3f}",
            )

            print(
                "Exactly 5 motif rate:",
                f"{thinking_summary['rate_eq_5']:.3f}",
            )

            print()
            print(
                "Thinking - Resting mean:",
                f"{mean_delta:+.3f}",
            )

            print(
                "Thinking - Resting group rate:",
                f"{group_rate_delta:+.3f}",
            )

            print(
                "Thinking - Resting 5-motif rate:",
                f"{five_rate_delta:+.3f}",
            )

            # =================================================
            # STORE SUBJECT × CHANNEL RESULT
            # =================================================

            subject_rows.append(
                {
                    "subject_id": subject_id,
                    "channel": channel_name,

                    "threshold": threshold,

                    "calibration_group_rate": (
                        calibration_group_rate
                    ),

                    # -----------------------------------------
                    # RESTING
                    # -----------------------------------------

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

                    "resting_std": (
                        resting_summary[
                            "std_detected"
                        ]
                    ),

                    "resting_group_rate": (
                        resting_summary[
                            "group_rate"
                        ]
                    ),

                    "resting_rate_ge_4": (
                        resting_summary[
                            "rate_ge_4"
                        ]
                    ),

                    "resting_rate_eq_5": (
                        resting_summary[
                            "rate_eq_5"
                        ]
                    ),

                    # -----------------------------------------
                    # THINKING
                    # -----------------------------------------

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

                    "thinking_std": (
                        thinking_summary[
                            "std_detected"
                        ]
                    ),

                    "thinking_group_rate": (
                        thinking_summary[
                            "group_rate"
                        ]
                    ),

                    "thinking_rate_ge_4": (
                        thinking_summary[
                            "rate_ge_4"
                        ]
                    ),

                    "thinking_rate_eq_5": (
                        thinking_summary[
                            "rate_eq_5"
                        ]
                    ),

                    # -----------------------------------------
                    # DIFFERENCES
                    # -----------------------------------------

                    "mean_delta": (
                        mean_delta
                    ),

                    "group_rate_delta": (
                        group_rate_delta
                    ),

                    "five_rate_delta": (
                        five_rate_delta
                    ),

                    "thinking_gt_resting": (
                        mean_delta > 0
                    ),
                }
            )

        # ====================================================
        # SAVE AFTER EVERY CHANNEL
        # ====================================================

        save_progress(
            subject_rows=subject_rows,
            epoch_rows=all_epoch_rows,
        )

        print()
        print(
            f"Progress saved after "
            f"{channel_name}."
        )

    # ========================================================
    # SUBJECT × CHANNEL TABLE
    # ========================================================

    subject_df = pd.DataFrame(
        subject_rows
    )

    epoch_df = pd.DataFrame(
        all_epoch_rows
    )

    # ========================================================
    # CHANNEL-LEVEL SUMMARY
    # ========================================================

    channel_rows = []

    for channel_name in CHANNELS_TO_TEST:

        channel_df = subject_df[
            subject_df[
                "channel"
            ] == channel_name
        ]

        channel_rows.append(
            {
                "channel": channel_name,

                "resting_mean": (
                    channel_df[
                        "resting_mean"
                    ].mean()
                ),

                "thinking_mean": (
                    channel_df[
                        "thinking_mean"
                    ].mean()
                ),

                "mean_delta": (
                    channel_df[
                        "mean_delta"
                    ].mean()
                ),

                "resting_group_rate": (
                    channel_df[
                        "resting_group_rate"
                    ].mean()
                ),

                "thinking_group_rate": (
                    channel_df[
                        "thinking_group_rate"
                    ].mean()
                ),

                "group_rate_delta": (
                    channel_df[
                        "group_rate_delta"
                    ].mean()
                ),

                "resting_five_rate": (
                    channel_df[
                        "resting_rate_eq_5"
                    ].mean()
                ),

                "thinking_five_rate": (
                    channel_df[
                        "thinking_rate_eq_5"
                    ].mean()
                ),

                "five_rate_delta": (
                    channel_df[
                        "five_rate_delta"
                    ].mean()
                ),

                "subjects_thinking_gt_resting": int(
                    (
                        channel_df[
                            "mean_delta"
                        ] > 0
                    ).sum()
                ),

                "subjects_group_rate_gt_resting": int(
                    (
                        channel_df[
                            "group_rate_delta"
                        ] > 0
                    ).sum()
                ),
            }
        )

    channel_df = pd.DataFrame(
        channel_rows
    )

    # Sort by task-related group-rate difference
    channel_df = channel_df.sort_values(
        by="group_rate_delta",
        ascending=False,
    )

    # ========================================================
    # FINAL SAVE
    # ========================================================

    subject_df.to_csv(
        SUBJECT_OUTPUT_PATH,
        index=False,
    )

    epoch_df.to_csv(
        EPOCH_OUTPUT_PATH,
        index=False,
    )

    channel_df.to_csv(
        CHANNEL_OUTPUT_PATH,
        index=False,
    )

    # ========================================================
    # PRINT CHANNEL SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 110)

    print(
        "ALL CHANNELS - REAL DATA SUMMARY"
    )

    print("=" * 110)

    columns = [
        "channel",

        "resting_mean",
        "thinking_mean",
        "mean_delta",

        "resting_group_rate",
        "thinking_group_rate",
        "group_rate_delta",

        "resting_five_rate",
        "thinking_five_rate",
        "five_rate_delta",

        "subjects_thinking_gt_resting",
        "subjects_group_rate_gt_resting",
    ]

    print(
        channel_df[
            columns
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 90)

    print(
        "BEST CHANNELS BY GROUP-RATE DELTA"
    )

    print("=" * 90)

    print(
        channel_df[
            [
                "channel",
                "group_rate_delta",
                "mean_delta",
                "five_rate_delta",
                "subjects_thinking_gt_resting",
                "subjects_group_rate_gt_resting",
            ]
        ].head(
            14
        ).to_string(
            index=False
        )
    )

    print()
    print(
        f"Subject summary:\n"
        f"{SUBJECT_OUTPUT_PATH}"
    )

    print()
    print(
        f"Epoch results:\n"
        f"{EPOCH_OUTPUT_PATH}"
    )

    print()
    print(
        f"Channel summary:\n"
        f"{CHANNEL_OUTPUT_PATH}"
    )

    print()
    print("=" * 90)
    print("DONE")
    print("=" * 90)