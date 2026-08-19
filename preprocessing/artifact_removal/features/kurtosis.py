from scipy.stats import kurtosis


def compute_kurtosis(signal):
    return kurtosis(signal, fisher=True, bias=False)