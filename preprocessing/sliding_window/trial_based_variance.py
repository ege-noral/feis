def compute_trial_variance(trials):
    """
    Parameters
    ----------
    trials : np.ndarray
        Shape (n_trials, n_channels, n_samples)

    Returns
    -------
    np.ndarray
        Shape (n_trials, n_channels)
    """
    
    return np.var(trials, axis=2)