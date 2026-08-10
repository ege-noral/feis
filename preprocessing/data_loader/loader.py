from pathlib import Path

import pandas as pd
import numpy as np


DATA_DIR = Path("data/FEIS")

from constants import EEG_CHANNELS

def load_subject_dataframe(
    subject_id,
    condition,
    data_dir="data/FEIS"
):
    """
    Load FEIS CSV as pandas DataFrame.
    """

    file_path = (
        Path(data_dir)
        / f"subj_{subject_id:02d}"
        / condition
        / f"{condition}.csv"
    )

    return pd.read_csv(file_path)
