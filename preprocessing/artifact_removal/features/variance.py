import numpy as np


def compute_variance(signal):
    """
    Compute the variance of a 1D EEG signal.
    """
    return np.var(signal)