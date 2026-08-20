import numpy as np

from constants import EEG_CHANNELS
from data_loader import get_one_epoch

from preprocessing.artifact_removal import (
    build_zscore_table,
    build_artifact_score_table,
    extract_dataset_features
)

# Buradaki import yolunu kendi klasör yapına göre düzelt
from preprocessing.artifact_removal import (
    bandpass_filter
)



SUBJECT_ID = 3
CONDITION = "thinking"
N_EPOCHS = 160
FS = 256
THRESHOLD = 3.5


# --------------------------------------------------
# Yardımcı fonksiyon:
# Bir epoch setini baştan sona artifact pipeline'dan geçir
# --------------------------------------------------

def analyze_epochs(epochs, label):
    print(f"\n================ {label} ================")

    # 1. Feature table
    feature_df = extract_dataset_features(
        epochs,
        channel_names=EEG_CHANNELS,
        fs=FS,
    )

    print(
        f"{label} feature table shape:",
        feature_df.shape
    )

    # 2. Robust z-score table
    zscore_df = build_zscore_table(
        feature_df
    )

    # 3. Artifact score table
    artifact_scores = build_artifact_score_table(
        zscore_df,
        threshold=THRESHOLD,
    )

    # 4. En şüpheliden en az şüpheliye sırala
    ranked_epochs = artifact_scores.sort_values(
        by=[
            "suspicious_channels",
            "total_flags",
            "max_flags_in_channel",
        ],
        ascending=False,
    ).reset_index(drop=True)

    # 5. En şüpheli 20 epoch
    print(f"\n--- {label}: MOST SUSPICIOUS EPOCHS ---")

    print(
        ranked_epochs.head(20).to_string(
            index=False
        )
    )

    # 6. Suspicious channel distribution
    channel_count_distribution = (
        artifact_scores["suspicious_channels"]
        .value_counts()
        .sort_index()
    )

    print(
        f"\n--- {label}: "
        f"SUSPICIOUS CHANNEL COUNT DISTRIBUTION ---"
    )

    for suspicious_channels, epoch_count in (
        channel_count_distribution.items()
    ):
        print(
            f"{suspicious_channels:2d} suspicious channels "
            f"-> {epoch_count:3d} epochs"
        )

    # 7. Sıfır skorlu epoch sayısı
    zero_score_count = (
        artifact_scores["suspicious_channels"] == 0
    ).sum()

    print(
        f"\n{label} zero-suspicious epochs: "
        f"{zero_score_count} / {len(artifact_scores)}"
    )

    return (
        feature_df,
        zscore_df,
        artifact_scores,
        ranked_epochs,
    )


# --------------------------------------------------
# 1. Subject 8'in tüm thinking epochlarını yükle
# --------------------------------------------------

raw_epochs = np.stack([
    get_one_epoch(
        subject_id=SUBJECT_ID,
        condition=CONDITION,
        epoch_idx=epoch_idx,
    )
    for epoch_idx in range(N_EPOCHS)
])

print("Raw epochs shape:", raw_epochs.shape)
# Beklenen:
# (160, 14, 1280)


# --------------------------------------------------
# 2. Aynı verinin filtrelenmiş halini oluştur
# --------------------------------------------------

filtered_epochs = bandpass_filter(
    raw_epochs,
    sfreq=FS,
    lowcut=0.5,
    highcut=40.0,
    order=4,
)

print(
    "Filtered epochs shape:",
    filtered_epochs.shape
)


# --------------------------------------------------
# 3. RAW veriyi analiz et
# --------------------------------------------------

(
    raw_feature_df,
    raw_zscore_df,
    raw_artifact_scores,
    raw_ranked_epochs,
) = analyze_epochs(
    raw_epochs,
    label="RAW"
)


# --------------------------------------------------
# 4. FILTERED veriyi analiz et
# --------------------------------------------------

(
    filtered_feature_df,
    filtered_zscore_df,
    filtered_artifact_scores,
    filtered_ranked_epochs,
) = analyze_epochs(
    filtered_epochs,
    label="FILTERED"
)


# --------------------------------------------------
# 5. RAW ve FILTERED sonuçlarını epoch bazında birleştir
# --------------------------------------------------

comparison = raw_artifact_scores.merge(
    filtered_artifact_scores,
    on="epoch",
    suffixes=("_raw", "_filtered"),
)


# --------------------------------------------------
# 6. Farkları da hesapla
# --------------------------------------------------

comparison["channel_difference"] = (
    comparison["suspicious_channels_filtered"]
    - comparison["suspicious_channels_raw"]
)

comparison["flag_difference"] = (
    comparison["total_flags_filtered"]
    - comparison["total_flags_raw"]
)


# --------------------------------------------------
# 7. Özellikle daha önce incelediğimiz epochları göster
# --------------------------------------------------

epochs_of_interest = [
    149,
    128,
    95,
    83,
    44,
]

selected_comparison = comparison[
    comparison["epoch"].isin(
        epochs_of_interest
    )
].sort_values(
    by="epoch",
    ascending=False,
)


print(
    "\n--- RAW vs FILTERED: "
    "SELECTED EPOCHS ---"
)

print(
    selected_comparison[
        [
            "epoch",
            "suspicious_channels_raw",
            "suspicious_channels_filtered",
            "total_flags_raw",
            "total_flags_filtered",
            "channel_difference",
            "flag_difference",
        ]
    ].to_string(
        index=False
    )
)


# --------------------------------------------------
# 8. Filtre sonrası suspicious channel sayısı
#    en fazla düşen epochları göster
# --------------------------------------------------

most_reduced = comparison.sort_values(
    by="channel_difference",
    ascending=True,
)

print(
    "\n--- EPOCHS WITH LARGEST REDUCTION "
    "AFTER FILTERING ---"
)

print(
    most_reduced[
        [
            "epoch",
            "suspicious_channels_raw",
            "suspicious_channels_filtered",
            "channel_difference",
            "total_flags_raw",
            "total_flags_filtered",
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)