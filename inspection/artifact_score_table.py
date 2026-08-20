import numpy as np

from constants import EEG_CHANNELS
from data_loader import get_one_epoch

from preprocessing.artifact_removal.dataset_feature_extraction import (
    extract_dataset_features
)

from preprocessing.artifact_removal.outlier_detection.zscore_table import (
    build_zscore_table
)

from preprocessing.artifact_removal.outlier_detection.artifact_score import (
    build_artifact_score_table
)


SUBJECT_ID = 2
CONDITION = "thinking"
N_EPOCHS = 160
FS = 256
THRESHOLD = 3.5


# --------------------------------------------------
# 1. Subject 8'in tüm thinking epochlarını yükle
# --------------------------------------------------

epochs = np.stack([
    get_one_epoch(
        subject_id=SUBJECT_ID,
        condition=CONDITION,
        epoch_idx=epoch_idx,
    )
    for epoch_idx in range(N_EPOCHS)
])

print("Epochs shape:", epochs.shape)
# Beklenen:
# (160, 14, 1280)


# --------------------------------------------------
# 2. Feature table oluştur
# --------------------------------------------------

feature_df = extract_dataset_features(
    epochs,
    channel_names=EEG_CHANNELS,
    fs=FS,
)

print("Feature table shape:", feature_df.shape)
# Beklenen:
# (2240, 9)


# --------------------------------------------------
# 3. Robust z-score table oluştur
# --------------------------------------------------

zscore_df = build_zscore_table(
    feature_df
)


# --------------------------------------------------
# 4. Her epoch için artifact score hesapla
# --------------------------------------------------

artifact_scores = build_artifact_score_table(
    zscore_df,
    threshold=THRESHOLD,
)


# --------------------------------------------------
# 5. Epochları en şüpheliden en az şüpheliye sırala
# --------------------------------------------------

ranked_epochs = artifact_scores.sort_values(
    by=[
        "total_flags",
        "suspicious_channels",
        "max_flags_in_channel",
    ],
    ascending=False,
).reset_index(drop=True)


# --------------------------------------------------
# 6. En şüpheli 20 epochu göster
# --------------------------------------------------

print("\n--- MOST SUSPICIOUS EPOCHS ---")

print(
    ranked_epochs.head(20).to_string(
        index=False
    )
)

# print("\n--- LEAST SUSPICIOUS EPOCHS ---")

# print(
#     ranked_epochs.tail(20).to_string(
#         index=False
#     )
# )


# --------------------------------------------------
# 7. Epoch 128'in sırasını bul
# --------------------------------------------------

epoch_128_indices = ranked_epochs.index[
    ranked_epochs["epoch"] == 128
]

