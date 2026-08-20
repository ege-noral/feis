from helper import compute_robust_zscore



FEATURE_COLUMNS = [
    "peak_to_peak",
    "variance",
    "kurtosis",
    "max_gradient",
    "hf_power_ratio",
    "mean_amplitude_deviation",
]


def build_zscore_table(
    feature_df,
    feature_columns=FEATURE_COLUMNS,
):
    """
    Compute robust z-scores for each artifact feature,
    separately for each EEG channel.

    Parameters
    ----------
    feature_df : pandas.DataFrame
        Feature table containing one row per epoch-channel pair.

    feature_columns : list[str]
        Feature columns for which robust z-scores will be computed.

    Returns
    -------
    pandas.DataFrame
        Copy of the input DataFrame with additional
        '<feature>_rz' columns.
    """

    result = feature_df.copy()

    for feature in feature_columns:

        result[f"{feature}_rz"] = (
            result
            .groupby("channel")[feature]
            .transform(
                lambda values: compute_robust_zscore(
                    values.to_numpy()
                )
            )
        )

    return result