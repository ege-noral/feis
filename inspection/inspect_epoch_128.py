import numpy as np

from constants import EEG_CHANNELS
from data_loader import get_one_epoch

from preprocessing.artifact_removal.dataset_feature_extraction import (
    extract_dataset_features
)

from preprocessing.artifact_removal.outlier_detection.zscore_table import (
    build_zscore_table
)


# 1. Subject 8'in TÜM thinking epochlarını yükle
epochs = np.stack([
    get_one_epoch(
        subject_id=8,
        condition="thinking",
        epoch_idx=epoch_idx,
    )
    for epoch_idx in range(160)
])

print("Epochs shape:", epochs.shape)
# Beklenen:
# (160, 14, 1280)


# 2. Feature table oluştur
feature_df = extract_dataset_features(
    epochs,
    channel_names=EEG_CHANNELS,
    fs=256,
)

print("Feature table shape:", feature_df.shape)


# 3. Robust z-score table oluştur
zscore_df = build_zscore_table(feature_df)


# 4. Epoch ID = 128'i seç
epoch_128 = zscore_df[
    zscore_df["epoch"] == 128
]


# 5. Şüpheli feature'ları göster
threshold = 3.5

features = [
    "peak_to_peak",
    "variance",
    "kurtosis",
    "max_gradient",
    "hf_power_ratio",
    "mean_amplitude_deviation",
]


print("\n--- SUBJECT 8 / EPOCH 128 ---")

for _, row in epoch_128.iterrows():

    suspicious = []

    for feature in features:

        z = row[f"{feature}_rz"]

        if abs(z) > threshold:
            suspicious.append(
                f"{feature} ({z:.2f})"
            )

    if suspicious:
        print(
            f"{row['channel']} -> "
            + ", ".join(suspicious)
        )