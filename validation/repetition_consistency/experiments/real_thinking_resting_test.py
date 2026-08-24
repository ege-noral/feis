import json
from pathlib import Path

import numpy as np
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


# ======================================================
# GENERAL SETTINGS
# ======================================================

CHANNEL_NAME = "P7"
FS = 256

# Subjects 6 and 7 were NOT used in the parameter search:
# train subjects      = 1, 2, 3, 4
# validation subjects = 5, 8
#
# Therefore 6 and 7 are our untouched real-data test subjects.
TEST_SUBJECTS = [
    6,
    7,
]

CONDITIONS = [
    "thinking",
    "resting",
]

# FEIS data used in the project has 160 epochs per condition.
# If one subject/condition has fewer epochs, the loader error is
# caught and that epoch is skipped.
EPOCH_INDICES = list(
    range(160)
)

PARAMS_PATH = Path(
    "validation/"
    "repetition_consistency/"
    "best_repetition_params.json"
)

OUTPUT_PATH = Path(
    "validation/"
    "repetition_consistency/"
    "real_thinking_resting_test_results.csv"
)


# ======================================================
# LOAD FIXED PARAMETERS
# ======================================================

def load_best_params():
    """
    Load the parameters found by the semi-synthetic
    Optuna search.

    IMPORTANT:
    These parameters are fixed here.
    No optimization is performed on real thinking/resting data.
    """

    if not PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"Best parameter file not found:\n{PARAMS_PATH}"
        )

    with open(
        PARAMS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        params = json.load(
            file
        )

    return params


# ======================================================
# DETECT REPETITIONS IN ONE REAL EPOCH
# ======================================================

def detect_one_real_epoch(
    subject_id,
    condition,
    epoch_idx,
    params,
):
    """
    Load one real FEIS epoch and detect repetition peaks.

    There are NO synthetic events here.

    For real thinking EEG we know that the protocol contains
    five repetitions, but we do not have exact event-center
    timestamps inside the 5-second epoch.

    Therefore this test compares:
        thinking detected-repeat distribution
        versus
        resting detected-repeat distribution

    It does NOT calculate matched / missed / F1 using timing
    ground truth.
    """

    channel_idx = (
        EEG_CHANNELS.index(
            CHANNEL_NAME
        )
    )

    epoch = (
        get_one_epoch(
            subject_id=subject_id,
            condition=condition,
            epoch_idx=epoch_idx,
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
# RUN COMPLETE REAL-DATA TEST
# ======================================================

def run_real_test(
    params,
):
    """
    Apply the already-fixed parameters to untouched real
    thinking and resting EEG.
    """

    rows = []

    for subject_id in TEST_SUBJECTS:

        print(
            "\n"
            "========================================"
        )

        print(
            f"SUBJECT {subject_id}"
        )

        print(
            "========================================"
        )

        for condition in CONDITIONS:

            print(
                f"\nCondition: {condition}"
            )

            successful_epochs = 0

            for epoch_idx in EPOCH_INDICES:

                try:

                    peak_times = (
                        detect_one_real_epoch(
                            subject_id=(
                                subject_id
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

                    # Reached beyond available epochs.
                    break

                except Exception as error:

                    print(
                        f"Skipped "
                        f"subject={subject_id}, "
                        f"condition={condition}, "
                        f"epoch={epoch_idx}: "
                        f"{error}"
                    )

                    continue

                successful_epochs += 1

                rows.append({
                    "subject_id": (
                        subject_id
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
                "Processed epochs:",
                successful_epochs,
            )

    results = pd.DataFrame(
        rows
    )

    return results


# ======================================================
# PRINT SUMMARY
# ======================================================

def print_condition_summary(
    results,
):
    """
    Main comparison:
    How many repetition peaks are detected in real thinking
    versus real resting EEG?
    """

    print(
        "\n\n"
        "========================================"
    )

    print(
        "OVERALL THINKING VS RESTING SUMMARY"
    )

    print(
        "========================================"
    )

    summary = (
        results
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
        summary.to_string(
            index=False
        )
    )


# ======================================================
# PRINT SUMMARY BY SUBJECT
# ======================================================

def print_subject_summary(
    results,
):
    print(
        "\n\n"
        "========================================"
    )

    print(
        "SUMMARY BY SUBJECT"
    )

    print(
        "========================================"
    )

    summary = (
        results
        .groupby(
            [
                "subject_id",
                "condition",
            ]
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
        )
        .reset_index()
    )

    print(
        summary.to_string(
            index=False
        )
    )


# ======================================================
# PRINT DETECTION-COUNT DISTRIBUTION
# ======================================================

def print_count_distribution(
    results,
):
    """
    Show how often the detector returns:
        0 peaks
        1 peak
        2 peaks
        ...
    in thinking and resting.
    """

    print(
        "\n\n"
        "========================================"
    )

    print(
        "DETECTED-REPEAT COUNT DISTRIBUTION"
    )

    print(
        "========================================"
    )

    distribution = (
        results
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
        results
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
# SIMPLE 4+ PEAK CHECK
# ======================================================

def print_four_plus_rate(
    results,
):
    """
    The semi-synthetic validation showed that real 5-repeat
    patterns are often detected as roughly 4 peaks.

    Therefore, as a simple descriptive check, report the
    percentage of epochs with at least 4 detected peaks.

    This is NOT a classifier accuracy score.
    """

    temp = (
        results.copy()
    )

    temp[
        "four_or_more"
    ] = (
        temp[
            "detected_repeats"
        ]
        >= 4
    )

    rates = (
        temp
        .groupby(
            "condition"
        )[
            "four_or_more"
        ]
        .mean()
        * 100.0
    )

    print(
        "\n\n"
        "========================================"
    )

    print(
        "EPOCHS WITH >= 4 DETECTED PEAKS"
    )

    print(
        "========================================"
    )

    for condition, rate in rates.items():

        print(
            f"{condition}: "
            f"{rate:.2f}%"
        )


# ======================================================
# MAIN
# ======================================================

def main():

    print(
        "\nLoading fixed parameters..."
    )

    params = (
        load_best_params()
    )

    print(
        "\n========================================"
    )

    print(
        "FIXED PARAMETERS"
    )

    print(
        "========================================"
    )

    for key, value in params.items():

        print(
            f"{key}: {value}"
        )

    print(
        "\nStarting REAL thinking/resting test..."
    )

    results = (
        run_real_test(
            params=params
        )
    )

    if len(
        results
    ) == 0:

        raise RuntimeError(
            "No real EEG epochs were processed."
        )

    print_condition_summary(
        results
    )

    print_subject_summary(
        results
    )

    print_count_distribution(
        results
    )

    print_four_plus_rate(
        results
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\n\nResults saved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()
