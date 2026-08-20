import pandas as pd


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
    Build an epoch-level artifact score table from robust z-scores.

    Parameters
    ----------
    zscore_df : pd.DataFrame
        Table containing one row per epoch-channel pair
        and robust z-score columns (<feature>_rz).

    threshold : float
        Robust z-score threshold for an outlier.

    features : list[str]
        Artifact features to evaluate.

    Returns
    -------
    pd.DataFrame
        One row per epoch containing:
        - suspicious_channels
        - total_flags
        - max_flags_in_channel
    """

    df = zscore_df.copy()

    flag_columns = []

    for feature in features:
        z_col = f"{feature}_rz"
        flag_col = f"{feature}_flag"

        # Mean amplitude deviation can be abnormal
        # in either positive or negative direction.
        if feature == "mean_amplitude_deviation":
            df[flag_col] = df[z_col].abs() > threshold

        # For the remaining features,
        # unusually HIGH values are suspicious.
        else:
            df[flag_col] = df[z_col] > threshold

        flag_columns.append(flag_col)

    # Number of suspicious features for each channel × epoch
    df["flag_count"] = df[flag_columns].sum(axis=1)

    # Epoch-level summary
    epoch_summary = (
        df.groupby("epoch")
        .agg(
            suspicious_channels=(
                "flag_count",
                lambda x: (x > 0).sum()
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

    return epoch_summary