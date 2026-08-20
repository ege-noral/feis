import numpy as np
import pandas as pd

from constants import EEG_CHANNELS
from data_loader import get_one_epoch

from preprocessing.artifact_removal import (
    build_zscore_table,
    build_artifact_score_table,
    extract_dataset_features
)

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

SUBJECT_IDS = [3]

CONDITION = "thinking"
N_EPOCHS = 160
FS = 256
THRESHOLD = 3.5

N_CHANNELS = len(EEG_CHANNELS)


# --------------------------------------------------
# Tek bir subject'i analiz et
# --------------------------------------------------

def analyze_subject(subject_id):

    print(
        f"\n"
        f"========================================\n"
        f"SUBJECT {subject_id}\n"
        f"========================================"
    )

    # ----------------------------------------------
    # 1. Tüm thinking epochlarını yükle
    # ----------------------------------------------

    epochs = np.stack([
        get_one_epoch(
            subject_id=subject_id,
            condition=CONDITION,
            epoch_idx=epoch_idx,
        )
        for epoch_idx in range(N_EPOCHS)
    ])

    print("Epochs shape:", epochs.shape)

    # ----------------------------------------------
    # 2. Feature table
    # ----------------------------------------------

    feature_df = extract_dataset_features(
        epochs,
        channel_names=EEG_CHANNELS,
        fs=FS,
    )

    # ----------------------------------------------
    # 3. Robust z-score table
    # ----------------------------------------------

    zscore_df = build_zscore_table(
        feature_df
    )
    trial_107 = zscore_df[
        zscore_df["epoch"] == 107
    ]

    features = [
        "peak_to_peak",
        "variance",
        "kurtosis",
        "max_gradient",
        "hf_power_ratio",
        "mean_amplitude_deviation",
    ]

    threshold = 3.5

    # ----------------------------------------------
    # 4. Artifact score table
    # ----------------------------------------------

    artifact_scores = build_artifact_score_table(
        zscore_df,
        threshold=THRESHOLD,
    )

    # ----------------------------------------------
    # 5. En şüpheli epochları sırala
    # ----------------------------------------------

    ranked_epochs = artifact_scores.sort_values(
        by=[
            "suspicious_channels",
            "total_flags",
            "max_flags_in_channel",
        ],
        ascending=False,
    ).reset_index(drop=True)

    print("\n--- TOP 10 SUSPICIOUS EPOCHS ---")

    print(
        ranked_epochs.head(10).to_string(
            index=False
        )
    )

    # ----------------------------------------------
    # 6. Suspicious channel dağılımı
    # ----------------------------------------------

    distribution = (
        artifact_scores["suspicious_channels"]
        .value_counts()
        .sort_index()
    )

    print(
        "\n--- SUSPICIOUS CHANNEL DISTRIBUTION ---"
    )

    for suspicious_channels, epoch_count in distribution.items():

        print(
            f"{suspicious_channels:2d} suspicious channels "
            f"-> {epoch_count:3d} epochs"
        )

    # ----------------------------------------------
    # 7. Subject-level özet
    # ----------------------------------------------

    zero_epochs = (
        artifact_scores["suspicious_channels"] == 0
    ).sum()

    one_or_two_channels = (
        artifact_scores["suspicious_channels"] <= 2
    ).sum()

    seven_or_more = (
        artifact_scores["suspicious_channels"] >= 7
    ).sum()

    ten_or_more = (
        artifact_scores["suspicious_channels"] >= 10
    ).sum()

    fourteen_channels = (
        artifact_scores["suspicious_channels"] == N_CHANNELS
    ).sum()

    max_suspicious_channels = (
        artifact_scores["suspicious_channels"].max()
    )

    max_total_flags = (
        artifact_scores["total_flags"].max()
    )

    most_suspicious_epoch = int(
        ranked_epochs.iloc[0]["epoch"]
    )

    summary = {
        "subject": subject_id,

        "zero_epochs": zero_epochs,

        "zero_to_two_channels": one_or_two_channels,

        "seven_or_more_channels": seven_or_more,

        "ten_or_more_channels": ten_or_more,

        "all_14_channels": fourteen_channels,

        "max_suspicious_channels": max_suspicious_channels,

        "max_total_flags": max_total_flags,

        "most_suspicious_epoch": most_suspicious_epoch,
    }

    return summary


# --------------------------------------------------
# TÜM SUBJECTLER
# --------------------------------------------------

subject_summaries = []

for subject_id in SUBJECT_IDS:

    summary = analyze_subject(
        subject_id
    )

    subject_summaries.append(
        summary
    )


# --------------------------------------------------
# Genel summary table
# --------------------------------------------------

summary_df = pd.DataFrame(
    subject_summaries
)

print(
    "\n\n"
    "========================================\n"
    "ALL SUBJECTS ARTIFACT SUMMARY\n"
    "========================================"
)

print(
    summary_df.to_string(
        index=False
    )
)