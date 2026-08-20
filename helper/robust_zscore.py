import numpy as np


def compute_robust_zscore(values):
    median = np.median(values)
    mad = np.median(
        np.abs(values - median)
    )

    if mad == 0:
        return np.zeros_like(values, dtype=float)

    return (
        0.6745
        * (values - median)
        / mad
    )