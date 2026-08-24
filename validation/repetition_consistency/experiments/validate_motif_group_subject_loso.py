import numpy as np
import pandas as pd
from pathlib import Path

from validation.repetition_consistency.experiments import (
    validate_motif_group_detector as experiment
)


# ============================================================
# FIXED SETTINGS
# ============================================================

ALL_SUBJECTS = list(range(1, 9))

CHANNEL_NAME = "AF4"

RESTING_PERCENTILE = 95.0

MIN_GROUP_DENSITY = 0.0
MIN_NEIGHBORS_PER_NODE = 2


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("FC6 - SUBJECT-WISE LOSO SEMI-SYNTHETIC VALIDATION")
    print("=" * 70)

    print(f"Channel: {CHANNEL_NAME}")
    print(f"Resting percentile: {RESTING_PERCENTILE}")
    print(f"Minimum group density: {MIN_GROUP_DENSITY}")
    print(
        "Minimum neighbors per node:",
        MIN_NEIGHBORS_PER_NODE,
    )

    all_results = []

    # ========================================================
    # LOSO LOOP
    # ========================================================

    for fold_number, test_subject in enumerate(
        ALL_SUBJECTS,
        start=1,
    ):

        train_subjects = [
            subject
            for subject in ALL_SUBJECTS
            if subject != test_subject
        ]

        # ----------------------------------------------------
        # IMPORTANT:
        # Functions in validate_motif_group_detector.py
        # use module-level global variables.
        # So we update them inside that module.
        # ----------------------------------------------------

        experiment.CHANNEL_NAME = CHANNEL_NAME
        experiment.TRAIN_SUBJECTS = train_subjects
        experiment.TEST_SUBJECT = test_subject

        print()
        print()
        print("=" * 70)
        print(
            f"FOLD {fold_number}/{len(ALL_SUBJECTS)}"
        )
        print("=" * 70)

        print(
            f"Held-out subject: {test_subject}"
        )

        print(
            f"Train subjects: {train_subjects}"
        )

        # ====================================================
        # LEARN RESTING DISTRIBUTION FROM TRAIN SUBJECTS ONLY
        # ====================================================

        resting_similarities = (
            experiment.learn_resting_similarity_distribution()
        )

        print()
        print(
            "Number of training resting similarities:",
            len(resting_similarities),
        )

        # ====================================================
        # LEARN THRESHOLD FROM TRAIN SUBJECTS ONLY
        # ====================================================

        similarity_threshold = np.percentile(
            resting_similarities,
            RESTING_PERCENTILE,
        )

        print(
            f"{RESTING_PERCENTILE:.1f}th percentile "
            f"threshold = {similarity_threshold:.4f}"
        )

        # ====================================================
        # HELD-OUT SUBJECT TEST
        # ====================================================

        result = experiment.evaluate_threshold(
            similarity_threshold=similarity_threshold,
            min_group_density=MIN_GROUP_DENSITY,
            min_neighbors_per_node=(
                MIN_NEIGHBORS_PER_NODE
            ),
        )

        # ====================================================
        # EXTRA USEFUL METRIC
        # ====================================================

        if result["five_mean_detected"] > 0:

            aggregate_precision = (
                result["five_mean_matched"]
                / result["five_mean_detected"]
            )

        else:
            aggregate_precision = np.nan

        # ====================================================
        # PRINT FOLD RESULT
        # ====================================================

        print()
        print("-" * 70)
        print("0 REPEAT")
        print("-" * 70)

        print(
            "Epochs with detected motif group:",
            f"{result['zero_group_rate']:.3f}",
        )

        print(
            "Mean detected repeats:",
            f"{result['zero_mean_detected']:.3f}",
        )

        print()
        print("-" * 70)
        print("5 REPEATS")
        print("-" * 70)

        print(
            "Mean detected:",
            f"{result['five_mean_detected']:.3f}",
        )

        print(
            "Mean matched:",
            f"{result['five_mean_matched']:.3f}",
        )

        print(
            "Mean false positives:",
            f"{result['five_mean_false_positives']:.3f}",
        )

        print(
            "Mean missed:",
            f"{result['five_mean_missed']:.3f}",
        )

        print(
            "Mean recall:",
            f"{result['five_mean_recall']:.3f}",
        )

        print(
            "Mean F1:",
            f"{result['five_mean_f1']:.3f}",
        )

        print(
            "Aggregate matched / detected:",
            f"{aggregate_precision:.3f}",
        )

        print(
            "Mean timing error:",
            f"{result['five_mean_timing_error']:.3f} s",
        )

        # ====================================================
        # STORE RESULT
        # ====================================================

        all_results.append(
            {
                "test_subject": test_subject,

                "train_subjects": ",".join(
                    map(str, train_subjects)
                ),

                "channel": CHANNEL_NAME,

                "similarity_threshold": (
                    similarity_threshold
                ),

                "zero_group_rate": result[
                    "zero_group_rate"
                ],

                "zero_mean_detected": result[
                    "zero_mean_detected"
                ],

                "five_mean_detected": result[
                    "five_mean_detected"
                ],

                "five_mean_matched": result[
                    "five_mean_matched"
                ],

                "five_mean_false_positives": result[
                    "five_mean_false_positives"
                ],

                "five_mean_missed": result[
                    "five_mean_missed"
                ],

                "five_mean_precision": result[
                    "five_mean_precision"
                ],

                "five_aggregate_precision": (
                    aggregate_precision
                ),

                "five_mean_recall": result[
                    "five_mean_recall"
                ],

                "five_mean_f1": result[
                    "five_mean_f1"
                ],

                "five_mean_timing_error": result[
                    "five_mean_timing_error"
                ],
            }
        )

    # ========================================================
    # FINAL DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    print()
    print()
    print("=" * 70)
    print("SUBJECT-WISE SUMMARY")
    print("=" * 70)

    summary_columns = [
        "test_subject",
        "similarity_threshold",
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
            summary_columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("OVERALL FC6 SUMMARY")
    print("=" * 70)

    print(
        "Mean 0-repeat group rate:",
        f"{results_df['zero_group_rate'].mean():.3f}",
    )

    print(
        "Mean 0-repeat detected:",
        f"{results_df['zero_mean_detected'].mean():.3f}",
    )

    print(
        "Mean 5-repeat detected:",
        f"{results_df['five_mean_detected'].mean():.3f}",
    )

    print(
        "Mean 5-repeat matched:",
        f"{results_df['five_mean_matched'].mean():.3f}",
    )

    print(
        "Mean 5-repeat false positives:",
        f"{results_df['five_mean_false_positives'].mean():.3f}",
    )

    print(
        "Mean 5-repeat recall:",
        f"{results_df['five_mean_recall'].mean():.3f}",
    )

    print(
        "Mean 5-repeat F1:",
        f"{results_df['five_mean_f1'].mean():.3f}",
    )

    print(
        "Mean matched / detected:",
        f"{results_df['five_aggregate_precision'].mean():.3f}",
    )

    print(
        "Mean timing error:",
        f"{results_df['five_mean_timing_error'].mean():.3f} s",
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
        / "motif_group_fc6_subject_loso_sanity.csv"
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