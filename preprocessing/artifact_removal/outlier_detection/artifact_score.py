FEATURES = [
    "peak_to_peak",
    "variance",
    "kurtosis",
    "max_gradient",
    "hf_power_ratio",
    "mean_amplitude_deviation",
]


def build_artifact_score_table(
    zscore_df,
    threshold=3.5,
    features=FEATURES,
):
    """
    Build an epoch-level artifact summary using a
    FASTER-inspired outlier decision strategy.

    A channel within an epoch is marked suspicious if
    at least one artifact feature is an outlier.

    Robust z-scores are used instead of classical z-scores.

    Parameters
    ----------
    zscore_df : pandas.DataFrame
        One row per epoch-channel pair containing robust
        z-score columns (<feature>_rz).

    threshold : float
        Robust z-score threshold.

    features : list[str]
        Features used for artifact detection.

    Returns
    -------
    pandas.DataFrame
        One row per epoch containing:
        - suspicious_channels
        - suspicious_channel_ratio
        - total_flags
        - max_flags_in_channel
    """

    df = zscore_df.copy()

    flag_columns = []

    for feature in features:

        z_col = f"{feature}_rz"
        flag_col = f"{feature}_flag"

        # FASTER-like statistical features:
        # both unusually HIGH and unusually LOW values
        # may represent abnormal channels.
        if feature in [
            "peak_to_peak",
            "variance",
            "max_gradient",
            "mean_amplitude_deviation",
        ]:
            df[flag_col] = df[z_col].abs() > threshold

        # Our additional features:
        # only unusually HIGH values are interpreted
        # as artifact evidence.
        elif feature in [
            "kurtosis",
            "hf_power_ratio",
        ]:
            df[flag_col] = df[z_col] > threshold

        flag_columns.append(flag_col)

    # Number of abnormal features for each channel × epoch
    df["flag_count"] = df[flag_columns].sum(axis=1)

    # FASTER-like decision:
    # at least one abnormal feature -> suspicious channel
    df["suspicious_channel"] = df["flag_count"] > 0

    n_channels = df["channel"].nunique()

    # Epoch-level summary
    epoch_summary = (
        df.groupby("epoch")
        .agg(
            suspicious_channels=(
                "suspicious_channel",
                "sum"
            ),
            total_flags=(
                "flag_count",
                "sum"
            ),
            max_flags_in_channel=(
                "flag_count",
                "max"
            ),
        )
        .reset_index()
    )

    epoch_summary["suspicious_channel_ratio"] = (
        epoch_summary["suspicious_channels"]
        / n_channels
    )

    return epoch_summary