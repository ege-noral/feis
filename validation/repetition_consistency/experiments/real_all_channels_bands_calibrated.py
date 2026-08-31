from pathlib import Path

import numpy as np
import pandas as pd

from constants import EEG_CHANNELS
from data_loader import get_one_epoch

from preprocessing.demean import demean
from quality_control.repetition_consistency.morlet_tfr import compute_morlet_tfr
from quality_control.repetition_consistency.self_similarity import (
    compute_tfr_self_similarity,
)
from quality_control.repetition_consistency.detectors.motif_group_detector import (
    detect_motif_groups,
)


# ============================================================
# FIXED SETTINGS
# ============================================================

FS = 256

SUBJECTS = list(range(1, 9))
CHANNELS = list(EEG_CHANNELS)

BANDS = {
    "theta": (4.0, 7.0),
    "alpha": (8.0, 12.0),
    "low_beta": (13.0, 20.0),
    "high_beta": (21.0, 30.0),
    "low_gamma": (31.0, 40.0),
}

CALIBRATION_RESTING_EPOCHS = list(range(0, 40))
TEST_EPOCHS = list(range(40, 160))

WINDOW_SECONDS = 0.40
STEP_SECONDS = 0.15

MIN_TIME_SEPARATION_SECONDS = 0.50

MIN_GROUP_SIZE = 4
MAX_GROUP_SIZE = 5

MIN_GROUP_DENSITY = 0.0
MIN_NEIGHBORS_PER_NODE = 2

MERGE_TOLERANCE_SECONDS = 0.20

TARGET_CALIBRATION_GROUP_RATE = 0.10

THRESHOLD_SEARCH_STEPS = 25


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = Path(
    "validation/repetition_consistency/archive"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT_SUMMARY_PATH = (
    OUTPUT_DIR
    / "real_all_channels_bands_calibrated_subject_summary.csv"
)

EPOCH_RESULTS_PATH = (
    OUTPUT_DIR
    / "real_all_channels_bands_calibrated_epoch_results.csv"
)

CHANNEL_BAND_SUMMARY_PATH = (
    OUTPUT_DIR
    / "real_all_channels_bands_calibrated_channel_band_summary.csv"
)

BAND_SUMMARY_PATH = (
    OUTPUT_DIR
    / "real_all_channels_bands_calibrated_band_summary.csv"
)


# ============================================================
# DETECTOR
# ============================================================

def run_detector(
    similarity_matrix,
    window_centers,
    similarity_threshold,
):
    """
    Run the already-fixed motif-group detector.
    """

    result = detect_motif_groups(
        similarity_matrix=similarity_matrix,
        window_centers_seconds=window_centers,
        similarity_threshold=similarity_threshold,
        min_time_separation_seconds=MIN_TIME_SEPARATION_SECONDS,
        min_group_size=MIN_GROUP_SIZE,
        max_group_size=MAX_GROUP_SIZE,
        min_group_density=MIN_GROUP_DENSITY,
        min_neighbors_per_node=MIN_NEIGHBORS_PER_NODE,
        merge_tolerance_seconds=MERGE_TOLERANCE_SECONDS,
    )

    return int(result["detected_repeats"])


# ============================================================
# BAND SELF-SIMILARITY
# ============================================================

def compute_band_similarities(epoch):
    """
    epoch shape:
        (n_channels, n_samples)

    Morlet is calculated ONCE per channel.

    Then the TFR rows are sliced into:
        theta
        alpha
        low beta
        high beta
        low gamma

    Returns
    -------
    results[channel_name][band_name] = similarity_matrix

    window_centers are the same for all channels/bands.
    """

    epoch = np.asarray(epoch, dtype=float)

    if epoch.ndim != 2:
        raise ValueError(
            f"Expected epoch shape (channels, samples), got {epoch.shape}"
        )

    epoch = demean(epoch, axis=-1)

    results = {}
    shared_window_centers = None

    for channel_idx, channel_name in enumerate(CHANNELS):

        signal = epoch[channel_idx]

        freqs, power = compute_morlet_tfr(
            signal,
            fs=FS,
        )

        results[channel_name] = {}

        for band_name, (f_low, f_high) in BANDS.items():

            band_mask = (
                (freqs >= f_low)
                & (freqs <= f_high)
            )

            if not np.any(band_mask):
                raise RuntimeError(
                    f"No frequencies found for {band_name}: "
                    f"{f_low}-{f_high} Hz"
                )

            band_power = power[band_mask, :]

            similarity_matrix, window_centers = (
                compute_tfr_self_similarity(
                    power=band_power,
                    fs=FS,
                    window_seconds=WINDOW_SECONDS,
                    step_seconds=STEP_SECONDS,
                )
            )

            # float32 saves quite a lot of memory.
            results[channel_name][band_name] = (
                similarity_matrix.astype(np.float32)
            )

            if shared_window_centers is None:
                shared_window_centers = np.asarray(
                    window_centers,
                    dtype=float,
                )

    return results, shared_window_centers


# ============================================================
# CALIBRATION
# ============================================================

def calculate_group_rate(
    similarity_matrices,
    window_centers,
    threshold,
):
    """
    Fraction of epochs where at least one valid motif group
    is detected.
    """

    detections = []

    for similarity_matrix in similarity_matrices:

        detected = run_detector(
            similarity_matrix,
            window_centers,
            threshold,
        )

        detections.append(detected > 0)

    return float(np.mean(detections))


def calibrate_similarity_threshold(
    calibration_similarity_matrices,
    window_centers,
):
    """
    Find the LOWEST similarity threshold such that
    calibration resting group-rate <= 10%.

    Higher threshold -> fewer similarity edges -> fewer groups.
    """

    low = -1.0
    high = 1.0

    for _ in range(THRESHOLD_SEARCH_STEPS):

        middle = (low + high) / 2.0

        group_rate = calculate_group_rate(
            calibration_similarity_matrices,
            window_centers,
            middle,
        )

        if group_rate <= TARGET_CALIBRATION_GROUP_RATE:
            # This threshold is strict enough.
            # Try slightly lower.
            high = middle
        else:
            # Too many resting groups.
            # Threshold must become stricter.
            low = middle

    final_threshold = high

    final_group_rate = calculate_group_rate(
        calibration_similarity_matrices,
        window_centers,
        final_threshold,
    )

    return final_threshold, final_group_rate


# ============================================================
# TEST CONDITION
# ============================================================

def evaluate_similarity_matrices(
    similarity_matrices,
    epoch_indices,
    window_centers,
    threshold,
    subject_id,
    channel_name,
    band_name,
    condition,
):
    detected_counts = []
    epoch_rows = []

    for epoch_idx, similarity_matrix in zip(
        epoch_indices,
        similarity_matrices,
    ):

        detected = run_detector(
            similarity_matrix,
            window_centers,
            threshold,
        )

        detected_counts.append(detected)

        epoch_rows.append(
            {
                "subject": subject_id,
                "channel": channel_name,
                "band": band_name,
                "condition": condition,
                "epoch_idx": epoch_idx,
                "similarity_threshold": threshold,
                "detected_repeats": detected,
                "has_group": int(detected > 0),
                "is_four": int(detected == 4),
                "is_five": int(detected == 5),
            }
        )

    detected_counts = np.asarray(
        detected_counts,
        dtype=int,
    )

    summary = {
        "mean_detected": float(
            np.mean(detected_counts)
        ),
        "median_detected": float(
            np.median(detected_counts)
        ),
        "std_detected": float(
            np.std(detected_counts)
        ),
        "group_rate": float(
            np.mean(detected_counts > 0)
        ),
        "four_rate": float(
            np.mean(detected_counts == 4)
        ),
        "five_rate": float(
            np.mean(detected_counts == 5)
        ),
        "max_detected": int(
            np.max(detected_counts)
        ),
    }

    return summary, epoch_rows


# ============================================================
# SAVE
# ============================================================

def save_progress(
    subject_rows,
    epoch_rows,
):
    pd.DataFrame(subject_rows).to_csv(
        SUBJECT_SUMMARY_PATH,
        index=False,
    )

    pd.DataFrame(epoch_rows).to_csv(
        EPOCH_RESULTS_PATH,
        index=False,
    )


# ============================================================
# SUBJECT PROCESSING
# ============================================================

def process_subject(subject_id):
    """
    Important efficiency detail:

    Each real EEG epoch is loaded ONCE.
    Morlet TFR is calculated ONCE per channel.

    The five frequency bands are then sliced from the
    SAME TFR.

    So we do NOT calculate Morlet five separate times.
    """

    print()
    print("=" * 70)
    print(f"SUBJECT {subject_id}")
    print("=" * 70)

    cache = {
        channel: {
            band: {
                "calibration_resting": [],
                "test_resting": [],
                "test_thinking": [],
            }
            for band in BANDS
        }
        for channel in CHANNELS
    }

    window_centers = None

    # --------------------------------------------------------
    # RESTING: epochs 0-159
    # --------------------------------------------------------

    print()
    print("Preparing RESTING TFR similarities...")

    for epoch_idx in range(160):

        print(
            f"\rResting epoch {epoch_idx + 1}/160",
            end="",
            flush=True,
        )

        epoch = get_one_epoch(
            subject_id=subject_id,
            condition="resting",
            epoch_idx=epoch_idx,
        )

        band_results, centers = (
            compute_band_similarities(epoch)
        )

        if window_centers is None:
            window_centers = centers

        for channel_name in CHANNELS:

            for band_name in BANDS:

                sim = band_results[
                    channel_name
                ][band_name]

                if epoch_idx in CALIBRATION_RESTING_EPOCHS:

                    cache[channel_name][band_name][
                        "calibration_resting"
                    ].append(sim)

                elif epoch_idx in TEST_EPOCHS:

                    cache[channel_name][band_name][
                        "test_resting"
                    ].append(sim)

    print()

    # --------------------------------------------------------
    # THINKING: epochs 40-159
    # --------------------------------------------------------

    print()
    print("Preparing THINKING TFR similarities...")

    for counter, epoch_idx in enumerate(
        TEST_EPOCHS,
        start=1,
    ):

        print(
            f"\rThinking epoch {counter}/{len(TEST_EPOCHS)}",
            end="",
            flush=True,
        )

        epoch = get_one_epoch(
            subject_id=subject_id,
            condition="thinking",
            epoch_idx=epoch_idx,
        )

        band_results, _ = (
            compute_band_similarities(epoch)
        )

        for channel_name in CHANNELS:

            for band_name in BANDS:

                cache[channel_name][band_name][
                    "test_thinking"
                ].append(
                    band_results[
                        channel_name
                    ][band_name]
                )

    print()

    # --------------------------------------------------------
    # CALIBRATE + TEST
    # --------------------------------------------------------

    subject_rows = []
    epoch_rows = []

    print()
    print("Calibrating and evaluating bands...")

    for channel_name in CHANNELS:

        print()
        print(f"Channel: {channel_name}")

        for band_name in BANDS:

            band_cache = (
                cache[channel_name][band_name]
            )

            threshold, calibration_group_rate = (
                calibrate_similarity_threshold(
                    band_cache[
                        "calibration_resting"
                    ],
                    window_centers,
                )
            )

            resting_summary, resting_epoch_rows = (
                evaluate_similarity_matrices(
                    band_cache["test_resting"],
                    TEST_EPOCHS,
                    window_centers,
                    threshold,
                    subject_id,
                    channel_name,
                    band_name,
                    "resting",
                )
            )

            thinking_summary, thinking_epoch_rows = (
                evaluate_similarity_matrices(
                    band_cache["test_thinking"],
                    TEST_EPOCHS,
                    window_centers,
                    threshold,
                    subject_id,
                    channel_name,
                    band_name,
                    "thinking",
                )
            )

            row = {
                "subject": subject_id,
                "channel": channel_name,
                "band": band_name,

                "similarity_threshold": threshold,
                "calibration_group_rate": calibration_group_rate,

                "resting_mean": resting_summary[
                    "mean_detected"
                ],
                "thinking_mean": thinking_summary[
                    "mean_detected"
                ],

                "mean_delta": (
                    thinking_summary["mean_detected"]
                    - resting_summary["mean_detected"]
                ),

                "resting_group_rate": resting_summary[
                    "group_rate"
                ],
                "thinking_group_rate": thinking_summary[
                    "group_rate"
                ],

                "group_rate_delta": (
                    thinking_summary["group_rate"]
                    - resting_summary["group_rate"]
                ),

                "resting_four_rate": resting_summary[
                    "four_rate"
                ],
                "thinking_four_rate": thinking_summary[
                    "four_rate"
                ],

                "four_rate_delta": (
                    thinking_summary["four_rate"]
                    - resting_summary["four_rate"]
                ),

                "resting_five_rate": resting_summary[
                    "five_rate"
                ],
                "thinking_five_rate": thinking_summary[
                    "five_rate"
                ],

                "five_rate_delta": (
                    thinking_summary["five_rate"]
                    - resting_summary["five_rate"]
                ),
            }

            subject_rows.append(row)

            epoch_rows.extend(
                resting_epoch_rows
            )

            epoch_rows.extend(
                thinking_epoch_rows
            )

            print(
                f"  {band_name:10s} | "
                f"thr={threshold:.4f} | "
                f"R={resting_summary['group_rate']:.3f} | "
                f"T={thinking_summary['group_rate']:.3f} | "
                f"Δ={row['group_rate_delta']:+.3f} | "
                f"5Δ={row['five_rate_delta']:+.3f}"
            )

    return subject_rows, epoch_rows


# ============================================================
# FINAL SUMMARIES
# ============================================================

def build_channel_band_summary(subject_df):

    rows = []

    grouped = subject_df.groupby(
        ["channel", "band"],
        sort=False,
    )

    for (channel, band), group in grouped:

        rows.append(
            {
                "channel": channel,
                "band": band,

                "resting_mean": group[
                    "resting_mean"
                ].mean(),

                "thinking_mean": group[
                    "thinking_mean"
                ].mean(),

                "mean_delta": group[
                    "mean_delta"
                ].mean(),

                "resting_group_rate": group[
                    "resting_group_rate"
                ].mean(),

                "thinking_group_rate": group[
                    "thinking_group_rate"
                ].mean(),

                "group_rate_delta": group[
                    "group_rate_delta"
                ].mean(),

                "resting_five_rate": group[
                    "resting_five_rate"
                ].mean(),

                "thinking_five_rate": group[
                    "thinking_five_rate"
                ].mean(),

                "five_rate_delta": group[
                    "five_rate_delta"
                ].mean(),

                "subjects_thinking_gt_resting": int(
                    np.sum(
                        group["mean_delta"] > 0
                    )
                ),

                "subjects_group_rate_gt_resting": int(
                    np.sum(
                        group["group_rate_delta"] > 0
                    )
                ),

                "subjects_five_rate_gt_resting": int(
                    np.sum(
                        group["five_rate_delta"] > 0
                    )
                ),
            }
        )

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "group_rate_delta",
        ascending=False,
    )

    return result


def build_band_summary(subject_df):

    rows = []

    for band_name in BANDS:

        band_df = subject_df[
            subject_df["band"] == band_name
        ]

        rows.append(
            {
                "band": band_name,

                "resting_mean": band_df[
                    "resting_mean"
                ].mean(),

                "thinking_mean": band_df[
                    "thinking_mean"
                ].mean(),

                "mean_delta": band_df[
                    "mean_delta"
                ].mean(),

                "resting_group_rate": band_df[
                    "resting_group_rate"
                ].mean(),

                "thinking_group_rate": band_df[
                    "thinking_group_rate"
                ].mean(),

                "group_rate_delta": band_df[
                    "group_rate_delta"
                ].mean(),

                "resting_five_rate": band_df[
                    "resting_five_rate"
                ].mean(),

                "thinking_five_rate": band_df[
                    "thinking_five_rate"
                ].mean(),

                "five_rate_delta": band_df[
                    "five_rate_delta"
                ].mean(),

                "positive_subject_channel_pairs": int(
                    np.sum(
                        band_df[
                            "group_rate_delta"
                        ] > 0
                    )
                ),

                "total_subject_channel_pairs": int(
                    len(band_df)
                ),
            }
        )

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "group_rate_delta",
        ascending=False,
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    all_subject_rows = []
    all_epoch_rows = []

    for subject_id in SUBJECTS:

        subject_rows, epoch_rows = (
            process_subject(subject_id)
        )

        all_subject_rows.extend(
            subject_rows
        )

        all_epoch_rows.extend(
            epoch_rows
        )

        # Save after every subject.
        save_progress(
            all_subject_rows,
            all_epoch_rows,
        )

        print()
        print(
            f"Progress saved after Subject {subject_id}"
        )

    subject_df = pd.DataFrame(
        all_subject_rows
    )

    channel_band_summary = (
        build_channel_band_summary(
            subject_df
        )
    )

    band_summary = build_band_summary(
        subject_df
    )

    channel_band_summary.to_csv(
        CHANNEL_BAND_SUMMARY_PATH,
        index=False,
    )

    band_summary.to_csv(
        BAND_SUMMARY_PATH,
        index=False,
    )

    print()
    print("=" * 70)
    print("BAND SUMMARY")
    print("=" * 70)

    print(
        band_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("TOP CHANNEL × BAND RESULTS")
    print("=" * 70)

    print(
        channel_band_summary.head(25).to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    print(SUBJECT_SUMMARY_PATH)
    print(EPOCH_RESULTS_PATH)
    print(CHANNEL_BAND_SUMMARY_PATH)
    print(BAND_SUMMARY_PATH)


if __name__ == "__main__":
    main()