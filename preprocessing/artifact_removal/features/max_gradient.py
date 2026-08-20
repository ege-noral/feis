import numpy as np


def compute_max_gradient(signal):
    gradients = np.diff(signal)
    return np.max(np.abs(gradients))