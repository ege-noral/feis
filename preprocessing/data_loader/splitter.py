import numpy as np


EEG_CHANNELS = [
    "F3", "FC5", "AF3", "F7",
    "T7", "P7", "O1", "O2",
    "P8", "T8", "F8", "AF4",
    "FC6", "F4"
]


def split_into_epochs(df):
    """
    Split FEIS dataframe into epoch-wise EEG arrays.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    np.ndarray
        Shape:
        (n_epochs, n_channels, n_samples)
    """

    epochs = []

    for epoch_id in sorted(df["Epoch"].unique()):

        epoch_df = df[df["Epoch"] == epoch_id]

        eeg = epoch_df[EEG_CHANNELS].to_numpy().T

        epochs.append(eeg)

    return np.array(epochs)