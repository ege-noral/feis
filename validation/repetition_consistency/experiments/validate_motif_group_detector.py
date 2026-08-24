import numpy as np

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
# SETTINGS
# ============================================================

FS = 256

# Önce F3'te deneyelim.
# Sonra sadece bunu "FC5" yaparak FC5'i deneyebiliriz.
CHANNEL_NAME = "F3"

CHANNELS_TO_TEST = EEG_CHANNELS

# Subject 8 tamamen görülmemiş test subject olsun.
TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7]
TEST_SUBJECT = 8

# Threshold öğrenmek için her training subject'ten 10 resting epoch.
TRAIN_BACKGROUND_EPOCHS = list(range(10))

# Subject 8 üzerinde 20 farklı resting background.
TEST_BACKGROUND_EPOCHS = list(range(20))

# Şimdilik eski deneylerde iyi çalışan ölçeğe yakın kalıyoruz.
WINDOW_SECONDS = 0.40
STEP_SECONDS = 0.15

# Birbirine bundan daha yakın pencereleri
# ayrı repetition olarak karşılaştırmayacağız.
MIN_TIME_SEPARATION_SECONDS = 0.50

# En az 3 farklı zaman birbirini karşılıklı desteklemeli.
MIN_GROUP_SIZE = 3

MERGE_TOLERANCE_SECONDS = 0.20

# Bir detected time'ı gerçek injected motif ile
# eşleştirmek için tolerans.
MATCH_TOLERANCE_SECONDS = 0.35

# Semi-synthetic ayarlarımız
DURATION_RANGE = (0.40, 0.80)
MOTIF_STD_RATIO = 0.50
MIN_GAP_SECONDS = 0.10
EDGE_MARGIN_SECONDS = 0.10

# Tek bir percentile seçmeyelim.
# Resting threshold arttıkça sonucu görelim.
RESTING_PERCENTILE = 95.0

GROUP_CONFIGS = [
    {
        "name": "FULL CLIQUE",
        "min_group_density": 1.0,
        "min_neighbors_per_node": None,
    },
    {
        "name": "DENSITY 0.50",
        "min_group_density": 0.50,
        "min_neighbors_per_node": None,
    },
    {
        "name": "MIN 3 NEIGHBORS",
        "min_group_density": 0.0,
        "min_neighbors_per_node": 3,
    },
    {
        "name": "MIN 2 NEIGHBORS",
        "min_group_density": 0.0,
        "min_neighbors_per_node": 2,
    },
    {
        "name": "DENSITY 0.50 + MIN 2 NEIGHBORS",
        "min_group_density": 0.50,
        "min_neighbors_per_node": 2,
    },
]

RANDOM_SEED = 42


# ============================================================
# HELPERS
# ============================================================

def compute_similarity_matrix(signal):
    """
    1D EEG -> demean -> Morlet TFR -> self-similarity
    """

    signal = demean(signal)

    _, power = compute_morlet_tfr(
        signal,
        fs=FS,
    )

    similarity_matrix, window_centers = compute_tfr_self_similarity(
        power,
        fs=FS,
        window_seconds=WINDOW_SECONDS,
        step_seconds=STEP_SECONDS,
    )

    return similarity_matrix, window_centers


def get_valid_distant_similarities(
    similarity_matrix,
    window_centers,
):
    """
    Self-similarity matrisinden sadece:
      - diagonal dışındaki
      - birbirinden yeterince uzak
    pencere çiftlerini alır.
    """

    time_difference = np.abs(
        window_centers[:, None]
        - window_centers[None, :]
    )

    upper_triangle = np.triu(
        np.ones_like(
            similarity_matrix,
            dtype=bool,
        ),
        k=1,
    )

    far_enough = (
        time_difference
        >= MIN_TIME_SEPARATION_SECONDS
    )

    valid_mask = upper_triangle & far_enough

    values = similarity_matrix[valid_mask]

    values = values[
        np.isfinite(values)
    ]

    return values


def learn_resting_similarity_distribution():
    """
    SADECE training subject'larının gerçek resting EEG'sini kullanır.

    Amaç:
        normal resting EEG'de distant TFR windows
        ne kadar birbirine benziyor?
    """

    channel_idx = EEG_CHANNELS.index(
        CHANNEL_NAME
    )

    all_similarity_values = []

    print("\nLearning resting similarity distribution...")

    for subject_id in TRAIN_SUBJECTS:

        print(
            f"  Subject {subject_id}"
        )

        for epoch_idx in TRAIN_BACKGROUND_EPOCHS:

            epoch = get_one_epoch(
                subject_id=subject_id,
                condition="resting",
                epoch_idx=epoch_idx,
            )

            signal = epoch[channel_idx]

            similarity_matrix, window_centers = (
                compute_similarity_matrix(signal)
            )

            values = get_valid_distant_similarities(
                similarity_matrix,
                window_centers,
            )

            all_similarity_values.extend(
                values.tolist()
            )

    all_similarity_values = np.asarray(
        all_similarity_values,
        dtype=float,
    )

    return all_similarity_values


def match_detected_to_true(
    detected_times,
    true_times,
):
    """
    Detected motif times ile injected gerçek motif merkezlerini
    one-to-one eşleştirir.

    En yakın çiftler önce eşleştirilir.
    """

    detected_times = np.asarray(
        detected_times,
        dtype=float,
    )

    true_times = np.asarray(
        true_times,
        dtype=float,
    )

    pairs = []

    for detected_index, detected_time in enumerate(
        detected_times
    ):
        for true_index, true_time in enumerate(
            true_times
        ):
            error = abs(
                detected_time - true_time
            )

            if error <= MATCH_TOLERANCE_SECONDS:
                pairs.append(
                    (
                        error,
                        detected_index,
                        true_index,
                    )
                )

    pairs.sort(
        key=lambda item: item[0]
    )

    used_detected = set()
    used_true = set()

    timing_errors = []

    for error, detected_index, true_index in pairs:

        if detected_index in used_detected:
            continue

        if true_index in used_true:
            continue

        used_detected.add(
            detected_index
        )

        used_true.add(
            true_index
        )

        timing_errors.append(
            error
        )

    matched = len(
        timing_errors
    )

    false_positives = (
        len(detected_times)
        - matched
    )

    missed = (
        len(true_times)
        - matched
    )

    return {
        "matched": matched,
        "false_positives": false_positives,
        "missed": missed,
        "timing_errors": timing_errors,
    }


def evaluate_threshold(
    similarity_threshold,
    min_group_density,
    min_neighbors_per_node,
):
    """
    Subject 8'in aynı resting background'ları üzerinde:

        0 injected motif
        5 injected motif

    test edilir.
    """

    channel_idx = EEG_CHANNELS.index(
        CHANNEL_NAME
    )

    zero_detected_counts = []
    zero_has_group = []

    five_detected_counts = []
    five_matched_counts = []
    five_false_positives = []
    five_missed_counts = []

    five_precisions = []
    five_recalls = []
    five_f1_scores = []
    five_timing_errors = []

    for epoch_idx in TEST_BACKGROUND_EPOCHS:

        resting_epoch = get_one_epoch(
            subject_id=TEST_SUBJECT,
            condition="resting",
            epoch_idx=epoch_idx,
        )

        background_signal = resting_epoch[
            channel_idx
        ]

        # ====================================================
        # 0 REPEAT
        # ====================================================

        zero_signal, zero_events = (
            create_semi_synthetic_epoch(
                background_signal=background_signal,
                fs=FS,
                n_repeats=0,
                duration_range=DURATION_RANGE,
                motif_std_ratio=MOTIF_STD_RATIO,
                min_gap_seconds=MIN_GAP_SECONDS,
                edge_margin_seconds=EDGE_MARGIN_SECONDS,
                seed=(
                    RANDOM_SEED
                    + epoch_idx * 100
                ),
            )
        )

        zero_similarity, zero_centers = (
            compute_similarity_matrix(
                zero_signal
            )
        )

        zero_result = detect_motif_groups(
            similarity_matrix=zero_similarity,
            window_centers_seconds=zero_centers,
            similarity_threshold=similarity_threshold,
            min_time_separation_seconds=(
                MIN_TIME_SEPARATION_SECONDS
            ),
            min_group_size=MIN_GROUP_SIZE,
            merge_tolerance_seconds=(
                MERGE_TOLERANCE_SECONDS
            ),
            min_group_density=min_group_density,
            min_neighbors_per_node=min_neighbors_per_node,
        )

        zero_detected = (
            zero_result[
                "detected_repeats"
            ]
        )

        zero_detected_counts.append(
            zero_detected
        )

        zero_has_group.append(
            zero_detected > 0
        )

        # ====================================================
        # 5 REPEATS
        # ====================================================

        five_signal, five_events = (
            create_semi_synthetic_epoch(
                background_signal=background_signal,
                fs=FS,
                n_repeats=5,
                duration_range=DURATION_RANGE,
                motif_std_ratio=MOTIF_STD_RATIO,
                min_gap_seconds=MIN_GAP_SECONDS,
                edge_margin_seconds=EDGE_MARGIN_SECONDS,
                seed=(
                    RANDOM_SEED
                    + 10000
                    + epoch_idx * 100
                ),
            )
        )

        true_times = np.asarray(
            [
                event["center_seconds"]
                for event in five_events
            ],
            dtype=float,
        )

        five_similarity, five_centers = (
            compute_similarity_matrix(
                five_signal
            )
        )

        five_result = detect_motif_groups(
            similarity_matrix=five_similarity,
            window_centers_seconds=five_centers,
            similarity_threshold=similarity_threshold,
            min_time_separation_seconds=(
                MIN_TIME_SEPARATION_SECONDS
            ),
            min_group_size=MIN_GROUP_SIZE,
            merge_tolerance_seconds=(
                MERGE_TOLERANCE_SECONDS
            ),
            min_group_density=min_group_density,
            min_neighbors_per_node=min_neighbors_per_node,
        )

        detected_times = five_result[
            "peak_times"
        ]

        match_result = match_detected_to_true(
            detected_times=detected_times,
            true_times=true_times,
        )

        detected_count = len(
            detected_times
        )

        matched = match_result[
            "matched"
        ]

        false_positives = match_result[
            "false_positives"
        ]

        missed = match_result[
            "missed"
        ]

        if detected_count > 0:
            precision = (
                matched
                / detected_count
            )
        else:
            precision = 0.0

        recall = (
            matched
            / len(true_times)
        )

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

        five_detected_counts.append(
            detected_count
        )

        five_matched_counts.append(
            matched
        )

        five_false_positives.append(
            false_positives
        )

        five_missed_counts.append(
            missed
        )

        five_precisions.append(
            precision
        )

        five_recalls.append(
            recall
        )

        five_f1_scores.append(
            f1
        )

        five_timing_errors.extend(
            match_result[
                "timing_errors"
            ]
        )

    return {
        "zero_group_rate": np.mean(
            zero_has_group
        ),
        "zero_mean_detected": np.mean(
            zero_detected_counts
        ),

        "five_mean_detected": np.mean(
            five_detected_counts
        ),
        "five_mean_matched": np.mean(
            five_matched_counts
        ),
        "five_mean_false_positives": np.mean(
            five_false_positives
        ),
        "five_mean_missed": np.mean(
            five_missed_counts
        ),
        "five_mean_precision": np.mean(
            five_precisions
        ),
        "five_mean_recall": np.mean(
            five_recalls
        ),
        "five_mean_f1": np.mean(
            five_f1_scores
        ),
        "five_mean_timing_error": (
            np.mean(
                five_timing_errors
            )
            if five_timing_errors
            else np.nan
        ),
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import pandas as pd
    from pathlib import Path

    print("=" * 70)
    print("ALL CHANNELS - MOTIF GROUP CONSENSUS SANITY TEST")
    print("=" * 70)

    print(f"Train subjects: {TRAIN_SUBJECTS}")
    print(f"Held-out test subject: {TEST_SUBJECT}")
    print(f"Channels: {CHANNELS_TO_TEST}")

    all_results = []

    # ========================================================
    # CHANNEL LOOP
    # ========================================================

    for channel_number, channel_name in enumerate(
        CHANNELS_TO_TEST,
        start=1,
    ):

        # Fonksiyonlarımız CHANNEL_NAME global değişkenini
        # kullandığı için burada güncelliyoruz.
        CHANNEL_NAME = channel_name

        print()
        print()
        print("=" * 70)
        print(
            f"CHANNEL {channel_number}/{len(CHANNELS_TO_TEST)}: "
            f"{CHANNEL_NAME}"
        )
        print("=" * 70)

        # ====================================================
        # 1. TRAINING RESTING DISTRIBUTION
        # ====================================================

        resting_similarities = (
            learn_resting_similarity_distribution()
        )

        print()
        print(
            "Number of training resting similarities:",
            len(resting_similarities),
        )

        # ====================================================
        # 2. CHANNEL-SPECIFIC RESTING THRESHOLD
        # ====================================================

        similarity_threshold = np.percentile(
            resting_similarities,
            RESTING_PERCENTILE,
        )

        print()
        print(
            f"{RESTING_PERCENTILE:.1f}th percentile "
            f"threshold for {CHANNEL_NAME}: "
            f"{similarity_threshold:.4f}"
        )

        # ====================================================
        # 3. TEST CONFIGURATIONS
        # ====================================================

        for config in GROUP_CONFIGS:

            print()
            print("-" * 70)
            print(
                f"{CHANNEL_NAME} | "
                f"{config['name']}"
            )
            print("-" * 70)

            result = evaluate_threshold(
                similarity_threshold=similarity_threshold,
                min_group_density=config[
                    "min_group_density"
                ],
                min_neighbors_per_node=config[
                    "min_neighbors_per_node"
                ],
            )

            # -----------------------------------------------
            # PRINT RESULT
            # -----------------------------------------------

            print()
            print("0 REPEAT")

            print(
                "Epochs with detected motif group:",
                f"{result['zero_group_rate']:.3f}",
            )

            print(
                "Mean detected repeats:",
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
                "Mean false positives:",
                f"{result['five_mean_false_positives']:.3f}",
            )

            print(
                "Mean missed:",
                f"{result['five_mean_missed']:.3f}",
            )

            print(
                "Mean precision:",
                f"{result['five_mean_precision']:.3f}",
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
                "Mean timing error:",
                f"{result['five_mean_timing_error']:.3f} s",
            )

            # -----------------------------------------------
            # STORE RESULT
            # -----------------------------------------------

            all_results.append(
                {
                    "channel": CHANNEL_NAME,
                    "configuration": config["name"],
                    "similarity_threshold": similarity_threshold,

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
    # FINAL SUMMARY
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    print()
    print()
    print("=" * 70)
    print("ALL CHANNELS SUMMARY")
    print("=" * 70)

    summary_columns = [
        "channel",
        "configuration",
        "similarity_threshold",
        "zero_mean_detected",
        "five_mean_detected",
        "five_mean_matched",
        "five_mean_false_positives",
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
    # MIN 2 NEIGHBORS RANKING
    # ========================================================

    neighbor_results = (
        results_df[
            results_df["configuration"]
            == "MIN 2 NEIGHBORS"
        ]
        .sort_values(
            by="five_mean_f1",
            ascending=False,
        )
    )

    print()
    print("=" * 70)
    print("MIN 2 NEIGHBORS - CHANNEL COMPARISON")
    print("=" * 70)

    print(
        neighbor_results[
            [
                "channel",
                "zero_group_rate",
                "zero_mean_detected",
                "five_mean_matched",
                "five_mean_false_positives",
                "five_mean_recall",
                "five_mean_f1",
                "five_mean_timing_error",
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE CSV
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
        / "motif_group_all_channels_sanity.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Results saved to:\n{output_path}"
    )