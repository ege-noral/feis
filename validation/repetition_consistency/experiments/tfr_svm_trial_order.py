from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)

from sklearn.model_selection import StratifiedGroupKFold
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
# ORIGINAL FEATURE STRUCTURE
# ============================================================

N_CHANNELS = 14
N_TIME_BINS = 10
N_FREQS = 37

TIME_BIN_SECONDS = 0.5

FEATURE_INDEX = np.arange(
    N_CHANNELS * N_TIME_BINS * N_FREQS
).reshape(
    N_CHANNELS,
    N_TIME_BINS,
    N_FREQS,
)


# ============================================================
# TRIAL-ORDER BLOCKS
# ============================================================

# Each block contains 30 paired epoch indices:
#
# 30 resting
# 30 thinking
#
# so every block has exactly 60 samples.

TRIAL_BLOCKS = {
    "block1_40_69": (40, 69),
    "block2_70_99": (70, 99),
    "block3_100_129": (100, 129),
    "block4_130_159": (130, 159),
}


# ============================================================
# WITHIN-EPOCH WINDOWS
# ============================================================

TIME_WINDOWS = {
    "full_0_5": (0.0, 5.0),
    "early_0_2": (0.0, 2.0),
    "late_2_5": (2.0, 5.0),
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
    "tfr_svm_trial_order"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUBJECT_RESULTS_PATH = (
    OUTPUT_DIR
    / "subject_trial_order_results.csv"
)

GROUP_RESULTS_PATH = (
    OUTPUT_DIR
    / "group_trial_order_results.csv"
)

FOLD_RESULTS_PATH = (
    OUTPUT_DIR
    / "fold_trial_order_results.csv"
)

SUBJECT_TREND_PATH = (
    OUTPUT_DIR
    / "subject_trial_order_trends.csv"
)


# ============================================================
# LOAD
# ============================================================

def load_subject_features(subject_id):

    path = (
        CACHE_DIR
        / f"subject_{subject_id}_features.npz"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Feature cache not found:\n{path}"
        )

    data = np.load(path)

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
# TIME WINDOW FEATURE INDICES
# ============================================================

def get_time_indices(
    start_seconds,
    end_seconds,
):

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

    indices = FEATURE_INDEX[
        :,
        start_bin:end_bin,
        :,
    ].ravel()

    return indices


# ============================================================
# REMOVE GLOBAL POWER
# ============================================================

def center_each_epoch(X):
    """
    Remove the mean TFR level separately
    from every selected epoch/window.

    This keeps the experiment consistent with
    our global-power-removed SVM analysis.
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
# CROSS VALIDATION
# ============================================================

def evaluate_cv(
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
    all_score = []

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

        score = model.decision_function(
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
            score,
        )

        fold_rows.append(
            {
                "fold": fold_idx,
                "accuracy": (
                    fold_accuracy
                ),
                "balanced_accuracy": (
                    fold_balanced_accuracy
                ),
                "roc_auc": (
                    fold_auc
                ),
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

        all_score.extend(
            score
        )

    all_true = np.asarray(
        all_true
    )

    all_pred = np.asarray(
        all_pred
    )

    all_score = np.asarray(
        all_score
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
            all_score,
        ),

        "fold_rows": fold_rows,
    }


# ============================================================
# SUBJECT
# ============================================================

def process_subject(subject_id):

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

    for block_number, (
        block_name,
        (
            first_epoch,
            last_epoch,
        ),
    ) in enumerate(
        TRIAL_BLOCKS.items(),
        start=1,
    ):

        # ----------------------------------------------------
        # SELECT RECORDING BLOCK
        # ----------------------------------------------------

        block_mask = (
            (groups >= first_epoch)
            & (groups <= last_epoch)
        )

        X_block = X[
            block_mask
        ]

        y_block = y[
            block_mask
        ]

        groups_block = groups[
            block_mask
        ]

        print()
        print(
            f"{block_name}: "
            f"{len(X_block)} samples"
        )

        print(
            "Class counts:",
            np.bincount(
                y_block
            ),
        )

        # ----------------------------------------------------
        # DIFFERENT WITHIN-EPOCH WINDOWS
        # ----------------------------------------------------

        for window_name, (
            start_seconds,
            end_seconds,
        ) in TIME_WINDOWS.items():

            feature_indices = (
                get_time_indices(
                    start_seconds,
                    end_seconds,
                )
            )

            X_window = X_block[
                :,
                feature_indices,
            ].copy()

            # Remove global mean power
            # within this particular window.
            X_window = center_each_epoch(
                X_window
            )

            result = evaluate_cv(
                X_window,
                y_block,
                groups_block,
            )

            row = {
                "subject": subject_id,

                "block_number": (
                    block_number
                ),

                "block": (
                    block_name
                ),

                "first_epoch": (
                    first_epoch
                ),

                "last_epoch": (
                    last_epoch
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

                "n_samples": int(
                    len(
                        X_window
                    )
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
            }

            subject_rows.append(
                row
            )

            for fold_row in (
                result[
                    "fold_rows"
                ]
            ):

                fold_rows.append(
                    {
                        "subject": (
                            subject_id
                        ),

                        "block_number": (
                            block_number
                        ),

                        "block": (
                            block_name
                        ),

                        "window": (
                            window_name
                        ),

                        **fold_row,
                    }
                )

            print(
                f"  "
                f"{window_name:10s} | "
                f"ACC="
                f"{result['accuracy']:.3f} | "
                f"AUC="
                f"{result['roc_auc']:.3f}"
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

    grouped = subject_df.groupby(
        [
            "block_number",
            "block",
            "window",
        ],
        sort=False,
    )

    for (
        block_number,
        block,
        window,
    ), group in grouped:

        rows.append(
            {
                "block_number": (
                    block_number
                ),

                "block": block,

                "window": window,

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

    result = pd.DataFrame(
        rows
    )

    return result.sort_values(
        [
            "window",
            "block_number",
        ]
    )


# ============================================================
# SUBJECT-SPECIFIC TREND
# ============================================================

def build_subject_trends(
    subject_df,
):
    """
    For each subject and each within-epoch window,
    fit a simple line:

        accuracy ~ block_number

    Negative slope:
        classification tends to get weaker
        as recording progresses.

    Positive slope:
        classification tends to improve.

    This is descriptive/exploratory,
    not a final inferential test.
    """

    rows = []

    grouped = subject_df.groupby(
        [
            "subject",
            "window",
        ]
    )

    for (
        subject,
        window,
    ), group in grouped:

        group = group.sort_values(
            "block_number"
        )

        x = group[
            "block_number"
        ].to_numpy(
            dtype=float
        )

        accuracy = group[
            "accuracy"
        ].to_numpy(
            dtype=float
        )

        auc = group[
            "roc_auc"
        ].to_numpy(
            dtype=float
        )

        accuracy_slope = float(
            np.polyfit(
                x,
                accuracy,
                1,
            )[0]
        )

        auc_slope = float(
            np.polyfit(
                x,
                auc,
                1,
            )[0]
        )

        first_accuracy = float(
            accuracy[0]
        )

        last_accuracy = float(
            accuracy[-1]
        )

        rows.append(
            {
                "subject": subject,

                "window": window,

                "accuracy_block1": (
                    first_accuracy
                ),

                "accuracy_block4": (
                    last_accuracy
                ),

                "block4_minus_block1": (
                    last_accuracy
                    - first_accuracy
                ),

                "accuracy_slope_per_block": (
                    accuracy_slope
                ),

                "auc_slope_per_block": (
                    auc_slope
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PRINT TABLE
# ============================================================

def print_window_summary(
    group_df,
    window_name,
):

    subset = group_df[
        group_df[
            "window"
        ] == window_name
    ].sort_values(
        "block_number"
    )

    print()
    print("=" * 75)
    print(
        window_name.upper()
    )
    print("=" * 75)

    print()

    print(
        subset[
            [
                "block_number",
                "block",
                "mean_accuracy",
                "mean_roc_auc",
                "subjects_above_60",
                "subjects_above_70",
            ]
        ].to_string(
            index=False
        )
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

        # Save after each subject.

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
            f"Saved after Subject "
            f"{subject_id}"
        )

    # --------------------------------------------------------
    # FINAL TABLES
    # --------------------------------------------------------

    subject_df = pd.DataFrame(
        all_subject_rows
    )

    group_df = build_group_summary(
        subject_df
    )

    trend_df = build_subject_trends(
        subject_df
    )

    group_df.to_csv(
        GROUP_RESULTS_PATH,
        index=False,
    )

    trend_df.to_csv(
        SUBJECT_TREND_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print_window_summary(
        group_df,
        "full_0_5",
    )

    print_window_summary(
        group_df,
        "early_0_2",
    )

    print_window_summary(
        group_df,
        "late_2_5",
    )

    print()
    print("=" * 75)
    print(
        "SUBJECT TREND SUMMARY"
    )
    print("=" * 75)
    print()

    for window_name in TIME_WINDOWS:

        subset = trend_df[
            trend_df[
                "window"
            ] == window_name
        ]

        n_negative = int(
            np.sum(
                subset[
                    "accuracy_slope_per_block"
                ] < 0
            )
        )

        mean_change = float(
            subset[
                "block4_minus_block1"
            ].mean()
        )

        mean_slope = float(
            subset[
                "accuracy_slope_per_block"
            ].mean()
        )

        print(
            f"{window_name:10s} | "
            f"negative slopes="
            f"{n_negative}/8 | "
            f"B4-B1="
            f"{mean_change:+.3f} | "
            f"mean slope="
            f"{mean_slope:+.3f}/block"
        )

    print()
    print(
        "Saved:"
    )

    print(
        SUBJECT_RESULTS_PATH
    )

    print(
        GROUP_RESULTS_PATH
    )

    print(
        FOLD_RESULTS_PATH
    )

    print(
        SUBJECT_TREND_PATH
    )


if __name__ == "__main__":
    main()