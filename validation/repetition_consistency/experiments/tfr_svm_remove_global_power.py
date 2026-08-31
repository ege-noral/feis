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

N_PERMUTATIONS = 20


# ============================================================
# EXISTING FEATURE CACHE
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
    "tfr_svm_global_power_removed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SUBJECT_SUMMARY_PATH = (
    OUTPUT_DIR
    / "subject_summary.csv"
)

FOLD_RESULTS_PATH = (
    OUTPUT_DIR
    / "fold_results.csv"
)

PERMUTATION_RESULTS_PATH = (
    OUTPUT_DIR
    / "permutation_results.csv"
)


# ============================================================
# LOAD EXISTING FEATURES
# ============================================================

def load_subject_features(subject_id):

    cache_path = (
        CACHE_DIR
        / f"subject_{subject_id}_features.npz"
    )

    if not cache_path.exists():

        raise FileNotFoundError(
            f"Could not find cached features:\n"
            f"{cache_path}\n\n"
            f"Run the original TFR-SVM script first."
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
# REMOVE GLOBAL POWER
# ============================================================

def remove_epoch_global_power(X):
    """
    Remove the global mean TFR level separately
    from every epoch.

    Input:
        X shape = (epochs, features)

    For every epoch:

        X_centered =
            X - mean(all 5180 features)

    Therefore the classifier cannot simply use:

        "this entire epoch has higher/lower power"

    It must use the relative spatial-frequency-temporal
    pattern instead.
    """

    epoch_means = np.mean(
        X,
        axis=1,
        keepdims=True,
    )

    X_centered = (
        X - epoch_means
    )

    return (
        X_centered,
        epoch_means.ravel(),
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
# CV SPLITS
# ============================================================

def make_cv():

    return StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_cv(
    X,
    y,
    groups,
):

    all_true = []
    all_pred = []
    all_scores = []

    fold_rows = []

    cv = make_cv()

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

        model = (
            create_classifier()
        )

        model.fit(
            X[train_idx],
            y[train_idx],
        )

        prediction = (
            model.predict(
                X[test_idx]
            )
        )

        scores = (
            model.decision_function(
                X[test_idx]
            )
        )

        accuracy = (
            accuracy_score(
                y[test_idx],
                prediction,
            )
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                y[test_idx],
                prediction,
            )
        )

        auc = (
            roc_auc_score(
                y[test_idx],
                scores,
            )
        )

        fold_rows.append(
            {
                "fold": fold_idx,

                "accuracy": accuracy,

                "balanced_accuracy": (
                    balanced_accuracy
                ),

                "roc_auc": auc,

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
        "accuracy": (
            accuracy_score(
                all_true,
                all_pred,
            )
        ),

        "balanced_accuracy": (
            balanced_accuracy_score(
                all_true,
                all_pred,
            )
        ),

        "roc_auc": (
            roc_auc_score(
                all_true,
                all_scores,
            )
        ),

        "fold_rows": fold_rows,
    }


# ============================================================
# PERMUTATION
# ============================================================

def permute_labels_within_pairs(
    y,
    groups,
    rng,
):

    y_permuted = (
        y.copy()
    )

    for group_id in np.unique(
        groups
    ):

        indices = np.where(
            groups == group_id
        )[0]

        if len(indices) != 2:

            raise RuntimeError(
                f"Expected two samples "
                f"for epoch group {group_id}, "
                f"got {len(indices)}."
            )

        if rng.random() < 0.5:

            y_permuted[
                indices
            ] = y_permuted[
                indices[::-1]
            ]

    return y_permuted


def run_permutation_test(
    X,
    y,
    groups,
    real_accuracy,
):

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    permutation_accuracies = []

    for permutation_idx in range(
        N_PERMUTATIONS
    ):

        print(
            f"\rPermutation "
            f"{permutation_idx + 1}/"
            f"{N_PERMUTATIONS}",
            end="",
            flush=True,
        )

        y_permuted = (
            permute_labels_within_pairs(
                y,
                groups,
                rng,
            )
        )

        result = evaluate_cv(
            X,
            y_permuted,
            groups,
        )

        permutation_accuracies.append(
            result["accuracy"]
        )

    print()

    permutation_accuracies = (
        np.asarray(
            permutation_accuracies,
            dtype=float,
        )
    )

    p_value = (
        1
        + np.sum(
            permutation_accuracies
            >= real_accuracy
        )
    ) / (
        N_PERMUTATIONS + 1
    )

    return (
        permutation_accuracies,
        float(p_value),
    )


# ============================================================
# GLOBAL POWER DESCRIPTIVE CHECK
# ============================================================

def summarize_removed_global_power(
    epoch_means,
    y,
):

    resting_values = (
        epoch_means[
            y == 0
        ]
    )

    thinking_values = (
        epoch_means[
            y == 1
        ]
    )

    resting_mean = float(
        np.mean(
            resting_values
        )
    )

    thinking_mean = float(
        np.mean(
            thinking_values
        )
    )

    return {
        "resting_global_power": (
            resting_mean
        ),

        "thinking_global_power": (
            thinking_mean
        ),

        "global_power_delta": (
            thinking_mean
            - resting_mean
        ),
    }


# ============================================================
# SUBJECT
# ============================================================

def process_subject(
    subject_id,
):

    print()
    print("=" * 70)
    print(
        f"SUBJECT {subject_id}"
    )
    print("=" * 70)

    X, y, groups = (
        load_subject_features(
            subject_id
        )
    )

    print()
    print(
        f"Original X shape: "
        f"{X.shape}"
    )

    # --------------------------------------------------------
    # ORIGINAL GLOBAL POWER
    # --------------------------------------------------------

    (
        X_centered,
        epoch_means,
    ) = remove_epoch_global_power(
        X
    )

    power_summary = (
        summarize_removed_global_power(
            epoch_means,
            y,
        )
    )

    print()
    print(
        "Global TFR power before centering:"
    )

    print(
        f"Resting : "
        f"{power_summary['resting_global_power']:.4f}"
    )

    print(
        f"Thinking: "
        f"{power_summary['thinking_global_power']:.4f}"
    )

    print(
        f"Delta   : "
        f"{power_summary['global_power_delta']:+.4f}"
    )

    # Sanity check:
    # every centered epoch should now have
    # mean approximately zero.

    centered_means = np.mean(
        X_centered,
        axis=1,
    )

    print()
    print(
        "Maximum absolute epoch mean "
        "after centering:"
    )

    print(
        f"{np.max(np.abs(centered_means)):.10f}"
    )

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    print()
    print(
        "Running BASELINE SVM..."
    )

    baseline_result = evaluate_cv(
        X,
        y,
        groups,
    )

    print(
        f"Baseline accuracy: "
        f"{baseline_result['accuracy']:.4f}"
    )

    print(
        f"Baseline ROC AUC: "
        f"{baseline_result['roc_auc']:.4f}"
    )

    # --------------------------------------------------------
    # GLOBAL POWER REMOVED
    # --------------------------------------------------------

    print()
    print(
        "Running GLOBAL-POWER-REMOVED SVM..."
    )

    centered_result = evaluate_cv(
        X_centered,
        y,
        groups,
    )

    print(
        f"Centered accuracy: "
        f"{centered_result['accuracy']:.4f}"
    )

    print(
        f"Centered balanced accuracy: "
        f"{centered_result['balanced_accuracy']:.4f}"
    )

    print(
        f"Centered ROC AUC: "
        f"{centered_result['roc_auc']:.4f}"
    )

    # --------------------------------------------------------
    # DIFFERENCE
    # --------------------------------------------------------

    accuracy_change = (
        centered_result["accuracy"]
        - baseline_result["accuracy"]
    )

    auc_change = (
        centered_result["roc_auc"]
        - baseline_result["roc_auc"]
    )

    print()
    print(
        "Effect of removing global power:"
    )

    print(
        f"Accuracy change: "
        f"{accuracy_change:+.4f}"
    )

    print(
        f"ROC AUC change: "
        f"{auc_change:+.4f}"
    )

    # --------------------------------------------------------
    # PERMUTATION
    # --------------------------------------------------------

    print()
    print(
        "Permutation test for centered features..."
    )

    (
        permutation_accuracies,
        permutation_p,
    ) = run_permutation_test(
        X_centered,
        y,
        groups,
        centered_result[
            "accuracy"
        ],
    )

    permutation_mean = float(
        np.mean(
            permutation_accuracies
        )
    )

    permutation_std = float(
        np.std(
            permutation_accuracies
        )
    )

    print()
    print(
        f"Permutation mean: "
        f"{permutation_mean:.4f}"
    )

    print(
        f"Permutation std: "
        f"{permutation_std:.4f}"
    )

    print(
        f"Permutation p: "
        f"{permutation_p:.4f}"
    )

    # --------------------------------------------------------
    # SAVE SUBJECT ROW
    # --------------------------------------------------------

    subject_row = {
        "subject": subject_id,

        "baseline_accuracy": (
            baseline_result[
                "accuracy"
            ]
        ),

        "centered_accuracy": (
            centered_result[
                "accuracy"
            ]
        ),

        "accuracy_change": (
            accuracy_change
        ),

        "baseline_balanced_accuracy": (
            baseline_result[
                "balanced_accuracy"
            ]
        ),

        "centered_balanced_accuracy": (
            centered_result[
                "balanced_accuracy"
            ]
        ),

        "baseline_roc_auc": (
            baseline_result[
                "roc_auc"
            ]
        ),

        "centered_roc_auc": (
            centered_result[
                "roc_auc"
            ]
        ),

        "roc_auc_change": (
            auc_change
        ),

        "resting_global_power": (
            power_summary[
                "resting_global_power"
            ]
        ),

        "thinking_global_power": (
            power_summary[
                "thinking_global_power"
            ]
        ),

        "global_power_delta": (
            power_summary[
                "global_power_delta"
            ]
        ),

        "permutation_mean_accuracy": (
            permutation_mean
        ),

        "permutation_std_accuracy": (
            permutation_std
        ),

        "permutation_p": (
            permutation_p
        ),
    }

    # --------------------------------------------------------
    # FOLD ROWS
    # --------------------------------------------------------

    fold_rows = []

    for row in centered_result[
        "fold_rows"
    ]:

        fold_rows.append(
            {
                "subject": subject_id,
                "feature_type": (
                    "global_power_removed"
                ),
                **row,
            }
        )

    # --------------------------------------------------------
    # PERMUTATION ROWS
    # --------------------------------------------------------

    permutation_rows = []

    for permutation_idx, accuracy in enumerate(
        permutation_accuracies,
        start=1,
    ):

        permutation_rows.append(
            {
                "subject": subject_id,
                "permutation": (
                    permutation_idx
                ),
                "accuracy": (
                    accuracy
                ),
            }
        )

    return (
        subject_row,
        fold_rows,
        permutation_rows,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    all_subject_rows = []
    all_fold_rows = []
    all_permutation_rows = []

    for subject_id in SUBJECTS:

        (
            subject_row,
            fold_rows,
            permutation_rows,
        ) = process_subject(
            subject_id
        )

        all_subject_rows.append(
            subject_row
        )

        all_fold_rows.extend(
            fold_rows
        )

        all_permutation_rows.extend(
            permutation_rows
        )

        # Save progress after every subject.

        pd.DataFrame(
            all_subject_rows
        ).to_csv(
            SUBJECT_SUMMARY_PATH,
            index=False,
        )

        pd.DataFrame(
            all_fold_rows
        ).to_csv(
            FOLD_RESULTS_PATH,
            index=False,
        )

        pd.DataFrame(
            all_permutation_rows
        ).to_csv(
            PERMUTATION_RESULTS_PATH,
            index=False,
        )

        print()
        print(
            f"Progress saved after Subject "
            f"{subject_id}"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    df = pd.DataFrame(
        all_subject_rows
    )

    print()
    print("=" * 70)
    print(
        "FINAL SUMMARY"
    )
    print("=" * 70)

    print()

    columns = [
        "subject",
        "baseline_accuracy",
        "centered_accuracy",
        "accuracy_change",
        "baseline_roc_auc",
        "centered_roc_auc",
        "global_power_delta",
    ]

    print(
        df[
            columns
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print(
        "OVERALL"
    )
    print("=" * 70)

    print()

    print(
        f"Baseline mean accuracy: "
        f"{df['baseline_accuracy'].mean():.4f}"
    )

    print(
        f"Centered mean accuracy: "
        f"{df['centered_accuracy'].mean():.4f}"
    )

    print(
        f"Mean accuracy change: "
        f"{df['accuracy_change'].mean():+.4f}"
    )

    print()

    print(
        f"Baseline mean ROC AUC: "
        f"{df['baseline_roc_auc'].mean():.4f}"
    )

    print(
        f"Centered mean ROC AUC: "
        f"{df['centered_roc_auc'].mean():.4f}"
    )

    print()

    print(
        "Mean global-power delta "
        "(Thinking - Resting): "
        f"{df['global_power_delta'].mean():+.4f}"
    )

    print()

    print(
        "Saved:"
    )

    print(
        SUBJECT_SUMMARY_PATH
    )

    print(
        FOLD_RESULTS_PATH
    )

    print(
        PERMUTATION_RESULTS_PATH
    )


if __name__ == "__main__":
    main()