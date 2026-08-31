from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)

from sklearn.model_selection import (
    StratifiedGroupKFold,
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# ============================================================
# SETTINGS
# ============================================================

SUBJECTS = list(range(1, 9))

N_SPLITS = 5
RANDOM_STATE = 42
SVM_C = 1.0


# ============================================================
# FEATURE STRUCTURE
# ============================================================

N_CHANNELS = 14
N_TIME_BINS = 10
N_FREQS = 37

TIME_BIN_SECONDS = 0.5


# Original feature extraction order:
#
# channel
#   -> time
#       -> frequency
#
# shape:
#
# 14 × 10 × 37
#
FEATURE_INDEX = np.arange(
    N_CHANNELS * N_TIME_BINS * N_FREQS
).reshape(
    N_CHANNELS,
    N_TIME_BINS,
    N_FREQS,
)


# ============================================================
# TIME WINDOWS
# ============================================================

# ------------------------------------------------------------
# Full epoch
# ------------------------------------------------------------

TIME_WINDOWS = {
    "full_0_5": (0.0, 5.0),

    # Main onset vs sustained comparison
    "early_0_2": (0.0, 2.0),
    "late_2_5": (2.0, 5.0),

    # Equal-length 2-second windows
    "window_0_2": (0.0, 2.0),
    "window_1_3": (1.0, 3.0),
    "window_2_4": (2.0, 4.0),
    "window_3_5": (3.0, 5.0),

    # Individual 1-second windows
    "sec_0_1": (0.0, 1.0),
    "sec_1_2": (1.0, 2.0),
    "sec_2_3": (2.0, 3.0),
    "sec_3_4": (3.0, 4.0),
    "sec_4_5": (4.0, 5.0),
}


# ============================================================
# FEATURE CACHE
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
    "tfr_svm_time_windows"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SUBJECT_RESULTS_PATH = (
    OUTPUT_DIR
    / "subject_time_window_results.csv"
)

GROUP_RESULTS_PATH = (
    OUTPUT_DIR
    / "group_time_window_results.csv"
)

FOLD_RESULTS_PATH = (
    OUTPUT_DIR
    / "fold_time_window_results.csv"
)


# ============================================================
# LOAD FEATURES
# ============================================================

def load_subject_features(subject_id):

    cache_path = (
        CACHE_DIR
        / f"subject_{subject_id}_features.npz"
    )

    if not cache_path.exists():

        raise FileNotFoundError(
            f"Could not find cached features:\n"
            f"{cache_path}"
        )

    data = np.load(
        cache_path
    )

    X = data["X"].astype(
        np.float64
    )

    y = data["y"].astype(
        int
    )

    groups = data["groups"].astype(
        int
    )

    return X, y, groups


# ============================================================
# TIME WINDOW -> FEATURE INDICES
# ============================================================

def get_time_window_indices(
    start_seconds,
    end_seconds,
):
    """
    Convert a requested time range into
    feature indices.

    Each TFR temporal bin is 0.5 seconds.

    Examples:

        0-2 s
        -> bins 0,1,2,3

        2-5 s
        -> bins 4,5,6,7,8,9
    """

    start_bin = int(
        round(
            start_seconds
            / TIME_BIN_SECONDS
        )
    )

    end_bin = int(
        round(
            end_seconds
            / TIME_BIN_SECONDS
        )
    )

    if start_bin < 0:
        raise ValueError(
            "start_seconds cannot be negative."
        )

    if end_bin > N_TIME_BINS:
        raise ValueError(
            "end_seconds exceeds epoch duration."
        )

    if start_bin >= end_bin:
        raise ValueError(
            "Invalid time window."
        )

    indices = FEATURE_INDEX[
        :,
        start_bin:end_bin,
        :,
    ].ravel()

    return indices


# ============================================================
# REMOVE GLOBAL POWER WITHIN SELECTED WINDOW
# ============================================================

def center_each_epoch(
    X,
):
    """
    Remove the mean TFR level separately
    from every epoch.

    IMPORTANT:

    This is done AFTER selecting the time window.

    Therefore:

    0-2 s classification cannot simply use
    the average power of 0-2 s.

    2-5 s classification cannot simply use
    the average power of 2-5 s.

    The classifier must use relative
    channel × frequency × time structure.
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
# CLASSIFIER
# ============================================================

def create_classifier():

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "svm",
                LinearSVC(
                    C=SVM_C,
                    dual=True,
                    max_iter=20000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


# ============================================================
# CV
# ============================================================

def evaluate_window(
    X,
    y,
    groups,
):

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    all_true = []
    all_pred = []
    all_scores = []

    fold_rows = []

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

        model = create_classifier()

        model.fit(
            X[train_idx],
            y[train_idx],
        )

        prediction = model.predict(
            X[test_idx]
        )

        scores = model.decision_function(
            X[test_idx]
        )

        fold_accuracy = accuracy_score(
            y[test_idx],
            prediction,
        )

        fold_balanced_accuracy = (
            balanced_accuracy_score(
                y[test_idx],
                prediction,
            )
        )

        fold_auc = roc_auc_score(
            y[test_idx],
            scores,
        )

        fold_rows.append(
            {
                "fold": fold_idx,
                "accuracy": fold_accuracy,
                "balanced_accuracy": (
                    fold_balanced_accuracy
                ),
                "roc_auc": fold_auc,
                "n_train": len(
                    train_idx
                ),
                "n_test": len(
                    test_idx
                ),
            }
        )

        all_true.extend(
            y[test_idx]
        )

        all_pred.extend(
            prediction
        )

        all_scores.extend(
            scores
        )

    all_true = np.asarray(
        all_true
    )

    all_pred = np.asarray(
        all_pred
    )

    all_scores = np.asarray(
        all_scores
    )

    return {
        "accuracy": accuracy_score(
            all_true,
            all_pred,
        ),

        "balanced_accuracy": (
            balanced_accuracy_score(
                all_true,
                all_pred,
            )
        ),

        "roc_auc": roc_auc_score(
            all_true,
            all_scores,
        ),

        "fold_rows": fold_rows,
    }


# ============================================================
# PROCESS SUBJECT
# ============================================================

def process_subject(
    subject_id,
):

    print()
    print("=" * 75)
    print(
        f"SUBJECT {subject_id}"
    )
    print("=" * 75)

    X, y, groups = (
        load_subject_features(
            subject_id
        )
    )

    subject_rows = []
    fold_rows = []

    for window_name, (
        start_seconds,
        end_seconds,
    ) in TIME_WINDOWS.items():

        # ----------------------------------------------------
        # SELECT TIME WINDOW
        # ----------------------------------------------------

        feature_indices = (
            get_time_window_indices(
                start_seconds,
                end_seconds,
            )
        )

        X_window = X[
            :,
            feature_indices,
        ].copy()

        # ----------------------------------------------------
        # REMOVE WINDOW GLOBAL POWER
        # ----------------------------------------------------

        X_window = center_each_epoch(
            X_window
        )

        # ----------------------------------------------------
        # SANITY
        # ----------------------------------------------------

        centered_means = np.mean(
            X_window,
            axis=1,
        )

        max_abs_mean = float(
            np.max(
                np.abs(
                    centered_means
                )
            )
        )

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        result = evaluate_window(
            X_window,
            y,
            groups,
        )

        duration = (
            end_seconds
            - start_seconds
        )

        row = {
            "subject": subject_id,

            "window": window_name,

            "start_seconds": (
                start_seconds
            ),

            "end_seconds": (
                end_seconds
            ),

            "duration_seconds": (
                duration
            ),

            "n_features": int(
                len(
                    feature_indices
                )
            ),

            "accuracy": (
                result[
                    "accuracy"
                ]
            ),

            "balanced_accuracy": (
                result[
                    "balanced_accuracy"
                ]
            ),

            "roc_auc": (
                result[
                    "roc_auc"
                ]
            ),

            "max_abs_mean_after_centering": (
                max_abs_mean
            ),
        }

        subject_rows.append(
            row
        )

        for fold_row in result[
            "fold_rows"
        ]:

            fold_rows.append(
                {
                    "subject": (
                        subject_id
                    ),

                    "window": (
                        window_name
                    ),

                    "start_seconds": (
                        start_seconds
                    ),

                    "end_seconds": (
                        end_seconds
                    ),

                    **fold_row,
                }
            )

        print(
            f"{window_name:12s} | "
            f"{start_seconds:.1f}-"
            f"{end_seconds:.1f} s | "
            f"features={len(feature_indices):4d} | "
            f"ACC={result['accuracy']:.4f} | "
            f"AUC={result['roc_auc']:.4f}"
        )

    return (
        subject_rows,
        fold_rows,
    )


# ============================================================
# GROUP SUMMARY
# ============================================================

def build_group_summary(
    subject_df,
):

    rows = []

    for window_name, group in (
        subject_df.groupby(
            "window",
            sort=False,
        )
    ):

        start_seconds = float(
            group[
                "start_seconds"
            ].iloc[0]
        )

        end_seconds = float(
            group[
                "end_seconds"
            ].iloc[0]
        )

        rows.append(
            {
                "window": window_name,

                "start_seconds": (
                    start_seconds
                ),

                "end_seconds": (
                    end_seconds
                ),

                "duration_seconds": (
                    end_seconds
                    - start_seconds
                ),

                "n_features": int(
                    group[
                        "n_features"
                    ].iloc[0]
                ),

                "mean_accuracy": float(
                    group[
                        "accuracy"
                    ].mean()
                ),

                "std_accuracy": float(
                    group[
                        "accuracy"
                    ].std()
                ),

                "mean_balanced_accuracy": float(
                    group[
                        "balanced_accuracy"
                    ].mean()
                ),

                "mean_roc_auc": float(
                    group[
                        "roc_auc"
                    ].mean()
                ),

                "subjects_above_60": int(
                    np.sum(
                        group[
                            "accuracy"
                        ] > 0.60
                    )
                ),

                "subjects_above_70": int(
                    np.sum(
                        group[
                            "accuracy"
                        ] > 0.70
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PRINT SPECIAL COMPARISON
# ============================================================

def print_subject_early_late_comparison(
    subject_df,
):

    print()
    print("=" * 75)
    print(
        "EARLY 0-2 s VS LATE 2-5 s"
    )
    print("=" * 75)

    early = (
        subject_df[
            subject_df[
                "window"
            ] == "early_0_2"
        ][
            [
                "subject",
                "accuracy",
                "roc_auc",
            ]
        ]
        .rename(
            columns={
                "accuracy": (
                    "early_accuracy"
                ),
                "roc_auc": (
                    "early_auc"
                ),
            }
        )
    )

    late = (
        subject_df[
            subject_df[
                "window"
            ] == "late_2_5"
        ][
            [
                "subject",
                "accuracy",
                "roc_auc",
            ]
        ]
        .rename(
            columns={
                "accuracy": (
                    "late_accuracy"
                ),
                "roc_auc": (
                    "late_auc"
                ),
            }
        )
    )

    comparison = early.merge(
        late,
        on="subject",
    )

    comparison[
        "late_minus_early_accuracy"
    ] = (
        comparison[
            "late_accuracy"
        ]
        - comparison[
            "early_accuracy"
        ]
    )

    comparison[
        "late_minus_early_auc"
    ] = (
        comparison[
            "late_auc"
        ]
        - comparison[
            "early_auc"
        ]
    )

    print()
    print(
        comparison.to_string(
            index=False
        )
    )

    print()
    print(
        "Mean early accuracy: "
        f"{comparison['early_accuracy'].mean():.4f}"
    )

    print(
        "Mean late accuracy : "
        f"{comparison['late_accuracy'].mean():.4f}"
    )

    print(
        "Late - early       : "
        f"{comparison['late_minus_early_accuracy'].mean():+.4f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    all_subject_rows = []
    all_fold_rows = []

    for subject_id in SUBJECTS:

        (
            subject_rows,
            fold_rows,
        ) = process_subject(
            subject_id
        )

        all_subject_rows.extend(
            subject_rows
        )

        all_fold_rows.extend(
            fold_rows
        )

        # Save after every subject.

        pd.DataFrame(
            all_subject_rows
        ).to_csv(
            SUBJECT_RESULTS_PATH,
            index=False,
        )

        pd.DataFrame(
            all_fold_rows
        ).to_csv(
            FOLD_RESULTS_PATH,
            index=False,
        )

        print()
        print(
            f"Progress saved after "
            f"Subject {subject_id}"
        )

    subject_df = pd.DataFrame(
        all_subject_rows
    )

    group_df = build_group_summary(
        subject_df
    )

    group_df.to_csv(
        GROUP_RESULTS_PATH,
        index=False,
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 75)
    print(
        "GROUP TIME-WINDOW SUMMARY"
    )
    print("=" * 75)
    print()

    print(
        group_df[
            [
                "window",
                "duration_seconds",
                "n_features",
                "mean_accuracy",
                "std_accuracy",
                "mean_roc_auc",
                "subjects_above_60",
                "subjects_above_70",
            ]
        ].to_string(
            index=False
        )
    )

    print_subject_early_late_comparison(
        subject_df
    )

    print()
    print("=" * 75)
    print(
        "EQUAL-LENGTH 2 SECOND WINDOWS"
    )
    print("=" * 75)
    print()

    equal_two_second = group_df[
        group_df[
            "window"
        ].isin(
            [
                "window_0_2",
                "window_1_3",
                "window_2_4",
                "window_3_5",
            ]
        )
    ]

    print(
        equal_two_second[
            [
                "window",
                "mean_accuracy",
                "mean_roc_auc",
                "subjects_above_60",
                "subjects_above_70",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 75)
    print(
        "ONE SECOND WINDOWS"
    )
    print("=" * 75)
    print()

    one_second = group_df[
        group_df[
            "window"
        ].isin(
            [
                "sec_0_1",
                "sec_1_2",
                "sec_2_3",
                "sec_3_4",
                "sec_4_5",
            ]
        )
    ]

    print(
        one_second[
            [
                "window",
                "mean_accuracy",
                "mean_roc_auc",
                "subjects_above_60",
                "subjects_above_70",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    print(
        SUBJECT_RESULTS_PATH
    )
    print(
        GROUP_RESULTS_PATH
    )
    print(
        FOLD_RESULTS_PATH
    )


if __name__ == "__main__":
    main()