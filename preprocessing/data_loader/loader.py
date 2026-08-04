from pathlib import Path

import pandas as pd
import numpy as np


DATA_DIR = Path("data/FEIS")

EEG_CHANNELS = [
    "F3",
    "FC5",
    "AF3",
    "F7",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "F8",
    "AF4",
    "FC6",
    "F4",
]


def load_subject(subject_id: int, condition: str) -> np.ndarray:
    """
    Load FEIS EEG recording.

    Returns
    -------
    np.ndarray
        Shape: (n_channels, n_samples)
    """

    subject_folder = DATA_DIR / f"subj_{subject_id:02d}"
    file_path = subject_folder / condition / f"{condition}.csv"

    df = pd.read_csv(file_path)

    print("DataFrame shape:", df.shape)
    print("Columns:", df.columns.tolist())

    eeg = df[EEG_CHANNELS].to_numpy()

    return eeg.T

def get_eeg_channels():
    return EEG_CHANNELS.copy()

def get_sampling_rate():
    return 256