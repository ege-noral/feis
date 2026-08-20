from preprocessing.artifact_removal import (
    reject_bad_epochs,
    build_zscore_table,
    build_artifact_score_table,
    extract_dataset_features

)

from constants import EEG_CHANNELS
from data_loader import get_one_epoch

import numpy as np

SUBJECT_ID = 8

CONDITION = "thinking"
N_EPOCHS = 160
FS = 256
THRESHOLD = 3.5

N_CHANNELS = len(EEG_CHANNELS)

epochs = np.stack([
    get_one_epoch(
        subject_id=SUBJECT_ID,
        condition=CONDITION,
        epoch_idx=epoch_idx,
    )
    for epoch_idx in range(N_EPOCHS)
])

print("Epochs shape:", epochs.shape)

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

clean_epochs, bad_epoch_indices = reject_bad_epochs(
    epochs,
    artifact_scores,
    min_suspicious_channel_ratio=0.5,
)

print("Original epochs:", epochs.shape)
print("Clean epochs:", clean_epochs.shape)

print("Rejected epochs:", len(bad_epoch_indices))
print("Rejected epoch indices:", bad_epoch_indices)