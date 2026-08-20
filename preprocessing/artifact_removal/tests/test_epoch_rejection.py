import numpy as np
import pandas as pd

from preprocessing.artifact_removal import (
    reject_bad_epochs
)

def test_reject_bad_epochs():

    # 4 mock epoch
    epochs = np.array([
        [[1, 1, 1]],
        [[2, 2, 2]],
        [[3, 3, 3]],
        [[4, 4, 4]],
    ], dtype=float)

    artifact_scores = pd.DataFrame({
        "epoch": [0, 1, 2, 3],

        "suspicious_channel_ratio": [
            0.00,
            0.50,
            0.25,
            1.00,
        ],
    })

    clean_epochs, bad_epoch_indices = reject_bad_epochs(
        epochs,
        artifact_scores,
        min_suspicious_channel_ratio=0.5,
    )

    # Epoch 1 ve 3 reject edilmeli
    expected_bad_indices = np.array([
        1,
        3,
    ])

    assert np.array_equal(
        bad_epoch_indices,
        expected_bad_indices
    )

    # Geriye epoch 0 ve 2 kalmalı
    assert clean_epochs.shape[0] == 2

    assert np.array_equal(
        clean_epochs,
        epochs[[0, 2]]
    )