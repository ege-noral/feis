import numpy as np


EEG_CHANNELS = [
    "F3", "FC5", "AF3", "F7",
    "T7", "P7", "O1", "O2",
    "P8", "T8", "F8", "AF4",
    "FC6", "F4"
]


def split_into_epochs(df):

    epochs = {}

    for epoch_id in sorted(df["Epoch"].unique()):

        epoch_df = df[df["Epoch"] == epoch_id]

        eeg = epoch_df[EEG_CHANNELS].to_numpy().T

        epochs[epoch_id] = {
            "eeg": eeg,
            "label": epoch_df["Label"].iloc[0],
            "stage": epoch_df["Stage"].iloc[0],
            "flag": epoch_df["Flag"].iloc[0]
        }

    return epochs