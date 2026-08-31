from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
)

from sklearn.model_selection import (
    StratifiedGroupKFold,
)

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from constants import EEG_CHANNELS


# ============================================================
# SETTINGS
# ============================================================

SUBJECTS = list(range(1, 9))

N_SPLITS = 5
RANDOM_STATE = 42
SVM_C = 1.0

# 20 yeterli ilk kontrol için.
# Daha sonra 100+ yapabiliriz.
N_IMPORTANCE_REPEATS = 20


# ============================================================
# FEATURE STRUCTURE
# ============================================================

FREQS = np.arange(
    4,
    41,
)

N_CHANNELS = len(
    EEG_CHANNELS
)

N_TIME_BINS = 10
N_FREQS = len(FREQS)

TIME_BIN_SECONDS = 0.5


BANDS = {
    "theta": (4, 7),
    "alpha": (8, 12),
    "low_beta": (13, 20),
    "high_beta": (21, 30),
    "low_gamma": (31, 40),
}


# ============================================================
# EXISTING CACHE
# ============================================================

CACHE_DIR = Path(
    "validation/repetition_consistency/archive/"
    "tfr_svm_thinking_resting/"
    "feature_cache"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = Path(
    "validation/repetition_consistency/archive/"
    "tfr_svm_feature_importance"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


BAND_PATH = (
    OUTPUT_DIR
    / "subject_band_importance.csv"
)

CHANNEL_PATH = (
    OUTPUT_DIR
    / "subject_channel_importance.csv"
)

TIME_PATH = (
    OUTPUT_DIR
    / "subject_time_importance.csv"
)

CHANNEL_BAND_PATH = (
    OUTPUT_DIR
    / "subject_channel_band_importance.csv"
)

GROUP_BAND_PATH = (
    OUTPUT_DIR
    / "group_band_importance.csv"
)

GROUP_CHANNEL_PATH = (
    OUTPUT_DIR
    / "group_channel_importance.csv"
)

GROUP_TIME_PATH = (
    OUTPUT_DIR
    / "group_time_importance.csv"
)

GROUP_CHANNEL_BAND_PATH = (
    OUTPUT_DIR
    / "group_channel_band_importance.csv"
)


# ============================================================
# FEATURE INDEX MAP
# ============================================================

# IMPORTANT:
#
# Original feature extraction order was:
#
# channel
#   → time bin
#       → frequency
#
# therefore:
#
# shape = (14, 10, 37)

FEATURE_INDEX = np.arange(
    N_CHANNELS
    * N_TIME_BINS
    * N_FREQS
).reshape(
    N_CHANNELS,
    N_TIME_BINS,
    N_FREQS,
)


# ============================================================
# LOAD
# ============================================================

def load_subject(
    subject_id,
):

    path = (
        CACHE_DIR
        / f"subject_{subject_id}_features.npz"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing feature cache:\n{path}"
        )

    data = np.load(
        path
    )

    X = data["X"].astype(
        np.float64
    )

    y = data["y"].astype(
        int
    )

    groups = data[
        "groups"
    ].astype(
        int
    )

    return X, y, groups


# ============================================================
# REMOVE GLOBAL POWER
# ============================================================

def remove_global_power(
    X,
):
    """
    Remove each epoch's global TFR mean.

    This is exactly the same idea used in the
    previous experiment.
    """

    return (
        X
        - np.mean(
            X,
            axis=1,
            keepdims=True,
        )
    )


# ============================================================
# FEATURE GROUPS
# ============================================================

def build_feature_groups():

    band_groups = {}

    channel_groups = {}

    time_groups = {}

    channel_band_groups = {}


    # --------------------------------------------------------
    # BANDS
    # --------------------------------------------------------

    for band_name, (
        low,
        high,
    ) in BANDS.items():

        freq_mask = (
            (FREQS >= low)
            & (FREQS <= high)
        )

        indices = (
            FEATURE_INDEX[
                :,
                :,
                freq_mask,
            ].ravel()
        )

        band_groups[
            band_name
        ] = indices


    # --------------------------------------------------------
    # CHANNELS
    # --------------------------------------------------------

    for channel_idx, channel_name in enumerate(
        EEG_CHANNELS
    ):

        indices = (
            FEATURE_INDEX[
                channel_idx,
                :,
                :,
            ].ravel()
        )

        channel_groups[
            channel_name
        ] = indices


    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    for time_idx in range(
        N_TIME_BINS
    ):

        start = (
            time_idx
            * TIME_BIN_SECONDS
        )

        end = (
            start
            + TIME_BIN_SECONDS
        )

        name = (
            f"{start:.1f}-{end:.1f}s"
        )

        indices = (
            FEATURE_INDEX[
                :,
                time_idx,
                :,
            ].ravel()
        )

        time_groups[
            name
        ] = indices


    # --------------------------------------------------------
    # CHANNEL × BAND
    # --------------------------------------------------------

    for channel_idx, channel_name in enumerate(
        EEG_CHANNELS
    ):

        for band_name, (
            low,
            high,
        ) in BANDS.items():

            freq_mask = (
                (FREQS >= low)
                & (FREQS <= high)
            )

            indices = (
                FEATURE_INDEX[
                    channel_idx,
                    :,
                    freq_mask,
                ].ravel()
            )

            name = (
                channel_name,
                band_name,
            )

            channel_band_groups[
                name
            ] = indices


    return (
        band_groups,
        channel_groups,
        time_groups,
        channel_band_groups,
    )


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

def calculate_group_importance(
    model,
    X_test,
    y_test,
    feature_indices,
    baseline_accuracy,
    baseline_auc,
    rng,
):
    """
    Jointly permute all features belonging to one group.

    Example for theta:

        every theta feature is moved together
        from one test epoch to another.

    If classification gets worse, the model
    was using information from that group.
    """

    accuracy_values = []
    auc_values = []

    for _ in range(
        N_IMPORTANCE_REPEATS
    ):

        X_permuted = (
            X_test.copy()
        )

        permutation = (
            rng.permutation(
                len(X_test)
            )
        )

        X_permuted[
            :,
            feature_indices,
        ] = X_test[
            permutation
        ][
            :,
            feature_indices,
        ]

        prediction = (
            model.predict(
                X_permuted
            )
        )

        score = (
            model.decision_function(
                X_permuted
            )
        )

        accuracy_values.append(
            accuracy_score(
                y_test,
                prediction,
            )
        )

        auc_values.append(
            roc_auc_score(
                y_test,
                score,
            )
        )

    permuted_accuracy = float(
        np.mean(
            accuracy_values
        )
    )

    permuted_auc = float(
        np.mean(
            auc_values
        )
    )

    return {
        "accuracy_drop": (
            baseline_accuracy
            - permuted_accuracy
        ),

        "auc_drop": (
            baseline_auc
            - permuted_auc
        ),
    }


# ============================================================
# PROCESS SUBJECT
# ============================================================

def process_subject(
    subject_id,
    band_groups,
    channel_groups,
    time_groups,
    channel_band_groups,
):

    print()
    print("=" * 70)
    print(
        f"SUBJECT {subject_id}"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    X, y, groups = (
        load_subject(
            subject_id
        )
    )

    X = remove_global_power(
        X
    )


    # --------------------------------------------------------
    # RESULT CONTAINERS
    # --------------------------------------------------------

    band_values = {
        name: []
        for name in band_groups
    }

    channel_values = {
        name: []
        for name in channel_groups
    }

    time_values = {
        name: []
        for name in time_groups
    }

    channel_band_values = {
        name: []
        for name in channel_band_groups
    }


    # --------------------------------------------------------
    # CROSS VALIDATION
    # --------------------------------------------------------

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


    for fold_idx, (
        train_idx,
        test_idx,
    ) in enumerate(
        cv.split(
            X,
            y,
            groups,
        ),
        start=1,
    ):

        print()
        print(
            f"Fold {fold_idx}/{N_SPLITS}"
        )


        # ----------------------------------------------------
        # STANDARDIZATION
        # ----------------------------------------------------

        scaler = (
            StandardScaler()
        )

        X_train = (
            scaler.fit_transform(
                X[train_idx]
            )
        )

        X_test = (
            scaler.transform(
                X[test_idx]
            )
        )


        # ----------------------------------------------------
        # SVM
        # ----------------------------------------------------

        model = LinearSVC(
            C=SVM_C,
            dual=True,
            max_iter=20000,
            random_state=RANDOM_STATE,
        )

        model.fit(
            X_train,
            y[train_idx],
        )


        # ----------------------------------------------------
        # BASELINE TEST PERFORMANCE
        # ----------------------------------------------------

        prediction = (
            model.predict(
                X_test
            )
        )

        score = (
            model.decision_function(
                X_test
            )
        )

        baseline_accuracy = (
            accuracy_score(
                y[test_idx],
                prediction,
            )
        )

        baseline_auc = (
            roc_auc_score(
                y[test_idx],
                score,
            )
        )

        print(
            f"Baseline accuracy: "
            f"{baseline_accuracy:.4f}"
        )

        print(
            f"Baseline AUC: "
            f"{baseline_auc:.4f}"
        )


        # ----------------------------------------------------
        # BAND IMPORTANCE
        # ----------------------------------------------------

        print(
            "  Band importance..."
        )

        for group_number, (
            name,
            indices,
        ) in enumerate(
            band_groups.items()
        ):

            rng = np.random.default_rng(
                RANDOM_STATE
                + subject_id * 100000
                + fold_idx * 1000
                + group_number
            )

            result = (
                calculate_group_importance(
                    model,
                    X_test,
                    y[test_idx],
                    indices,
                    baseline_accuracy,
                    baseline_auc,
                    rng,
                )
            )

            band_values[
                name
            ].append(
                result
            )


        # ----------------------------------------------------
        # CHANNEL IMPORTANCE
        # ----------------------------------------------------

        print(
            "  Channel importance..."
        )

        for group_number, (
            name,
            indices,
        ) in enumerate(
            channel_groups.items()
        ):

            rng = np.random.default_rng(
                RANDOM_STATE
                + subject_id * 200000
                + fold_idx * 1000
                + group_number
            )

            result = (
                calculate_group_importance(
                    model,
                    X_test,
                    y[test_idx],
                    indices,
                    baseline_accuracy,
                    baseline_auc,
                    rng,
                )
            )

            channel_values[
                name
            ].append(
                result
            )


        # ----------------------------------------------------
        # TIME IMPORTANCE
        # ----------------------------------------------------

        print(
            "  Time importance..."
        )

        for group_number, (
            name,
            indices,
        ) in enumerate(
            time_groups.items()
        ):

            rng = np.random.default_rng(
                RANDOM_STATE
                + subject_id * 300000
                + fold_idx * 1000
                + group_number
            )

            result = (
                calculate_group_importance(
                    model,
                    X_test,
                    y[test_idx],
                    indices,
                    baseline_accuracy,
                    baseline_auc,
                    rng,
                )
            )

            time_values[
                name
            ].append(
                result
            )


        # ----------------------------------------------------
        # CHANNEL × BAND
        # ----------------------------------------------------

        print(
            "  Channel × band importance..."
        )

        for group_number, (
            name,
            indices,
        ) in enumerate(
            channel_band_groups.items()
        ):

            rng = np.random.default_rng(
                RANDOM_STATE
                + subject_id * 400000
                + fold_idx * 1000
                + group_number
            )

            result = (
                calculate_group_importance(
                    model,
                    X_test,
                    y[test_idx],
                    indices,
                    baseline_accuracy,
                    baseline_auc,
                    rng,
                )
            )

            channel_band_values[
                name
            ].append(
                result
            )


    # ========================================================
    # AVERAGE FOLDS
    # ========================================================

    band_rows = []

    for name, values in (
        band_values.items()
    ):

        band_rows.append(
            {
                "subject": subject_id,
                "band": name,

                "accuracy_drop": np.mean(
                    [
                        x["accuracy_drop"]
                        for x in values
                    ]
                ),

                "auc_drop": np.mean(
                    [
                        x["auc_drop"]
                        for x in values
                    ]
                ),
            }
        )


    channel_rows = []

    for name, values in (
        channel_values.items()
    ):

        channel_rows.append(
            {
                "subject": subject_id,
                "channel": name,

                "accuracy_drop": np.mean(
                    [
                        x["accuracy_drop"]
                        for x in values
                    ]
                ),

                "auc_drop": np.mean(
                    [
                        x["auc_drop"]
                        for x in values
                    ]
                ),
            }
        )


    time_rows = []

    for name, values in (
        time_values.items()
    ):

        time_rows.append(
            {
                "subject": subject_id,
                "time": name,

                "accuracy_drop": np.mean(
                    [
                        x["accuracy_drop"]
                        for x in values
                    ]
                ),

                "auc_drop": np.mean(
                    [
                        x["auc_drop"]
                        for x in values
                    ]
                ),
            }
        )


    channel_band_rows = []

    for (
        channel_name,
        band_name,
    ), values in (
        channel_band_values.items()
    ):

        channel_band_rows.append(
            {
                "subject": subject_id,

                "channel": (
                    channel_name
                ),

                "band": (
                    band_name
                ),

                "accuracy_drop": np.mean(
                    [
                        x["accuracy_drop"]
                        for x in values
                    ]
                ),

                "auc_drop": np.mean(
                    [
                        x["auc_drop"]
                        for x in values
                    ]
                ),
            }
        )


    return (
        band_rows,
        channel_rows,
        time_rows,
        channel_band_rows,
    )


# ============================================================
# GROUP SUMMARIES
# ============================================================

def build_group_summary(
    df,
    group_columns,
):

    rows = []

    if isinstance(
        group_columns,
        str,
    ):
        group_columns = [
            group_columns
        ]

    grouped = df.groupby(
        group_columns,
        sort=False,
    )

    for key, group in grouped:

        if not isinstance(
            key,
            tuple,
        ):
            key = (
                key,
            )

        row = {}

        for column, value in zip(
            group_columns,
            key,
        ):
            row[column] = value

        row[
            "mean_accuracy_drop"
        ] = float(
            group[
                "accuracy_drop"
            ].mean()
        )

        row[
            "mean_auc_drop"
        ] = float(
            group[
                "auc_drop"
            ].mean()
        )

        row[
            "subjects_positive_accuracy_drop"
        ] = int(
            np.sum(
                group[
                    "accuracy_drop"
                ] > 0
            )
        )

        row[
            "subjects_positive_auc_drop"
        ] = int(
            np.sum(
                group[
                    "auc_drop"
                ] > 0
            )
        )

        rows.append(
            row
        )

    result = pd.DataFrame(
        rows
    )

    return result.sort_values(
        "mean_accuracy_drop",
        ascending=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    (
        band_groups,
        channel_groups,
        time_groups,
        channel_band_groups,
    ) = build_feature_groups()


    all_band_rows = []
    all_channel_rows = []
    all_time_rows = []
    all_channel_band_rows = []


    for subject_id in SUBJECTS:

        (
            band_rows,
            channel_rows,
            time_rows,
            channel_band_rows,
        ) = process_subject(
            subject_id,
            band_groups,
            channel_groups,
            time_groups,
            channel_band_groups,
        )

        all_band_rows.extend(
            band_rows
        )

        all_channel_rows.extend(
            channel_rows
        )

        all_time_rows.extend(
            time_rows
        )

        all_channel_band_rows.extend(
            channel_band_rows
        )


        # Save progress after every subject.

        pd.DataFrame(
            all_band_rows
        ).to_csv(
            BAND_PATH,
            index=False,
        )

        pd.DataFrame(
            all_channel_rows
        ).to_csv(
            CHANNEL_PATH,
            index=False,
        )

        pd.DataFrame(
            all_time_rows
        ).to_csv(
            TIME_PATH,
            index=False,
        )

        pd.DataFrame(
            all_channel_band_rows
        ).to_csv(
            CHANNEL_BAND_PATH,
            index=False,
        )

        print()
        print(
            f"Saved after Subject "
            f"{subject_id}"
        )


    # ========================================================
    # FINAL DATAFRAMES
    # ========================================================

    band_df = pd.DataFrame(
        all_band_rows
    )

    channel_df = pd.DataFrame(
        all_channel_rows
    )

    time_df = pd.DataFrame(
        all_time_rows
    )

    channel_band_df = pd.DataFrame(
        all_channel_band_rows
    )


    # ========================================================
    # GROUP SUMMARIES
    # ========================================================

    group_band = (
        build_group_summary(
            band_df,
            "band",
        )
    )

    group_channel = (
        build_group_summary(
            channel_df,
            "channel",
        )
    )

    group_time = (
        build_group_summary(
            time_df,
            "time",
        )
    )

    group_channel_band = (
        build_group_summary(
            channel_band_df,
            [
                "channel",
                "band",
            ],
        )
    )


    group_band.to_csv(
        GROUP_BAND_PATH,
        index=False,
    )

    group_channel.to_csv(
        GROUP_CHANNEL_PATH,
        index=False,
    )

    group_time.to_csv(
        GROUP_TIME_PATH,
        index=False,
    )

    group_channel_band.to_csv(
        GROUP_CHANNEL_BAND_PATH,
        index=False,
    )


    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 70)
    print(
        "BAND IMPORTANCE"
    )
    print("=" * 70)

    print(
        group_band.to_string(
            index=False
        )
    )


    print()
    print("=" * 70)
    print(
        "CHANNEL IMPORTANCE"
    )
    print("=" * 70)

    print(
        group_channel.to_string(
            index=False
        )
    )


    print()
    print("=" * 70)
    print(
        "TIME IMPORTANCE"
    )
    print("=" * 70)

    print(
        group_time.to_string(
            index=False
        )
    )


    print()
    print("=" * 70)
    print(
        "TOP 20 CHANNEL × BAND"
    )
    print("=" * 70)

    print(
        group_channel_band.head(
            20
        ).to_string(
            index=False
        )
    )


    print()
    print(
        "Saved:"
    )

    print(
        GROUP_BAND_PATH
    )

    print(
        GROUP_CHANNEL_PATH
    )

    print(
        GROUP_TIME_PATH
    )

    print(
        GROUP_CHANNEL_BAND_PATH
    )


if __name__ == "__main__":
    main()