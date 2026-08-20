from pathlib import Path

import pandas as pd
import numpy as np

from constants import EEG_CHANNELS


DATA_DIR = Path("data/FEIS")
EPOCH_LENGTH = 1280


def load_subject_dataframe(
    subject_id,
    condition,
    data_dir=DATA_DIR,
):
    """
    Load complete FEIS CSV as pandas DataFrame.
    """

    file_path = (
        Path(data_dir)
        / f"subj_{subject_id:02d}"
        / condition
        / f"{condition}.csv"
    )

    return pd.read_csv(file_path)


def get_one_epoch(
    subject_id,
    condition,
    epoch_idx=0,
    data_dir=DATA_DIR,
):
    """
    Read a single FEIS epoch directly from CSV.

    Returns
    -------
    np.ndarray
        Shape: (n_channels, 1280)
    """

    file_path = (
        Path(data_dir)
        / f"subj_{subject_id:02d}"
        / condition
        / f"{condition}.csv"
    )

    start_row = epoch_idx * EPOCH_LENGTH

    df = pd.read_csv(
        file_path,
        skiprows=range(1, start_row + 1),
        nrows=EPOCH_LENGTH,
    )

    epoch = df[EEG_CHANNELS].to_numpy().T

    return epoch