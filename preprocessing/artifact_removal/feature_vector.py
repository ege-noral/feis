from preprocessing.artifact_removal.features.peak_to_peak import (
    compute_peak_to_peak
)
from preprocessing.artifact_removal.features.variance import (
    compute_variance
)
from preprocessing.artifact_removal.features.kurtosis import (
    compute_kurtosis
)
from preprocessing.artifact_removal.features.max_gradient import (
    compute_max_gradient
)
from preprocessing.artifact_removal.features.hf_power_ratio import (
    compute_hf_power_ratio
)
from preprocessing.artifact_removal.features.mean_amplitude_deviation import (
    compute_mean_amplitude_deviation
)


def artifact_feature_vector(
    signal,
    reference_mean,
    fs=256,
):
    """
    Extract artifact-related features from one EEG channel
    within one epoch.

    Parameters
    ----------
    signal : np.ndarray
        1D EEG signal.

    reference_mean : float
        Reference mean amplitude of the same EEG channel.

    fs : int
        Sampling frequency.

    Returns
    -------
    dict
        Artifact-related feature values.
    """

    return {
        "peak_to_peak": compute_peak_to_peak(signal),
        "variance": compute_variance(signal),
        "kurtosis": compute_kurtosis(signal),
        "max_gradient": compute_max_gradient(signal),
        "hf_power_ratio": compute_hf_power_ratio(signal, fs=fs),
        "mean_amplitude_deviation": compute_mean_amplitude_deviation(
            signal,
            reference_mean
        ),
    }