from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

from sklearn.model_selection import (
    StratifiedGroupKFold,
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from constants import EEG_CHANNELS
from data_loader import get_one_epoch

from preprocessing.demean import demean

from quality_control.repetition_consistency.morlet_tfr import (
    compute_morlet_tfr,
)


# ============================================================
# SETTINGS
# ============================================================

FS = 256

SUBJECTS = list(range(1, 9))

# We deliberately use the held-out portion that we previously
# used for real resting/thinking evaluation.
#
# This keeps epochs 0-39 outside this first sanity check.
EPOCH_INDICES = list(range(40, 160))

CONDITIONS = {
    "resting": 0,
    "thinking": 1,
}


# ============================================================
# TFR FEATURE SETTINGS
# ============================================================

# Full Morlet range:
# 4, 5, ..., 40 Hz
#
# We do NOT split into theta/alpha/beta here.
# Let the SVM see the complete TFR information.

N_TIME_BINS = 10

# 5 seconds / 10 bins = 0.5 s per temporal bin.

# Each epoch will therefore have approximately:
#
# 14 channels
# × 37 frequencies
# × 10 temporal bins
#
# = 5180 features


# ============================================================
# CROSS VALIDATION
# ============================================================

N_SPLITS = 5

RANDOM_STATE = 42

SVM_C = 1.0


# ============================================================
# PERMUTATION SANITY CHECK
# ============================================================

# Start with 20 so the experiment does not become unnecessarily
# slow.
#
# Later we can increase this to 100, 500, 1000 for publication.
N_PERMUTATIONS = 20


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = Path(
    "validation/repetition_consistency/archive/"
    "tfr_svm_thinking_resting"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CACHE_DIR = OUTPUT_DIR / "feature_cache"

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUBJECT_SUMMARY_PATH = (
    OUTPUT_DIR
    / "tfr_svm_subject_summary.csv"
)

FOLD_RESULTS_PATH = (
    OUTPUT_DIR
    / "tfr_svm_fold_results.csv"
)

PERMUTATION_RESULTS_PATH = (
    OUTPUT_DIR
    / "tfr_svm_permutation_results.csv"
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def compute_tfr_features(epoch):
    """
    Convert one EEG epoch into a TFR feature vector.

    Input
    -----
    epoch:
        shape = (14, 1280)

    Steps
    -----
    1. Demean every channel.
    2. Morlet TFR: 4-40 Hz.
    3. log1p(power).
    4. Divide the 5-second epoch into 10 temporal bins.
    5. Average TFR power inside every temporal bin.

    Output
    ------
    feature_vector:
        shape approximately (5180,)
    """

    epoch = np.asarray(
        epoch,
        dtype=float,
    )

    if epoch.ndim != 2:
        raise ValueError(
            f"Expected 2-D epoch, got shape {epoch.shape}"
        )

    if epoch.shape[0] != len(EEG_CHANNELS):
        raise ValueError(
            f"Expected {len(EEG_CHANNELS)} channels, "
            f"got {epoch.shape[0]}"
        )

    # Remove DC separately from every EEG channel.
    epoch = demean(
        epoch,
        axis=-1,
    )

    all_channel_features = []

    for channel_idx in range(
        len(EEG_CHANNELS)
    ):

        signal = epoch[
            channel_idx
        ]

        freqs, power = (
            compute_morlet_tfr(
                signal,
                fs=FS,
            )
        )

        # power:
        # (37 frequencies, 1280 samples)

        log_power = np.log1p(
            power
        )

        n_samples = (
            log_power.shape[1]
        )

        # Split 5 seconds into equal temporal bins.
        bin_edges = np.linspace(
            0,
            n_samples,
            N_TIME_BINS + 1,
            dtype=int,
        )

        channel_tfr_summary = []

        for bin_idx in range(
            N_TIME_BINS
        ):

            start = (
                bin_edges[bin_idx]
            )

            end = (
                bin_edges[
                    bin_idx + 1
                ]
            )

            bin_power = (
                log_power[
                    :,
                    start:end,
                ]
            )

            # Mean power for every frequency
            # inside this 0.5-s temporal section.
            mean_frequency_power = (
                np.mean(
                    bin_power,
                    axis=1,
                )
            )

            channel_tfr_summary.append(
                mean_frequency_power
            )

        # shape:
        #
        # time bins × frequency
        #
        # (10, 37)
        channel_tfr_summary = (
            np.stack(
                channel_tfr_summary,
                axis=0,
            )
        )

        all_channel_features.append(
            channel_tfr_summary.ravel()
        )

    feature_vector = np.concatenate(
        all_channel_features
    )

    return feature_vector


# ============================================================
# SUBJECT DATASET
# ============================================================

def build_subject_dataset(
    subject_id,
):
    """
    Build:

        X = TFR feature matrix
        y = condition labels
        groups = paired epoch index

    For every epoch index we have:

        resting epoch 40
        thinking epoch 40

    and so on.

    Both receive the SAME group ID.

    This is important:
    paired epoch indices can never be split between
    training and test folds.
    """

    cache_path = (
        CACHE_DIR
        / f"subject_{subject_id}_features.npz"
    )

    if cache_path.exists():

        print()
        print(
            f"Loading cached features for "
            f"Subject {subject_id}..."
        )

        cache = np.load(
            cache_path
        )

        return (
            cache["X"],
            cache["y"],
            cache["groups"],
        )

    print()
    print(
        f"Extracting TFR features "
        f"for Subject {subject_id}..."
    )

    X = []
    y = []
    groups = []

    total_epochs = (
        len(EPOCH_INDICES)
        * len(CONDITIONS)
    )

    counter = 0

    for epoch_idx in EPOCH_INDICES:

        for condition, label in (
            CONDITIONS.items()
        ):

            counter += 1

            print(
                f"\rSubject {subject_id} | "
                f"{counter}/{total_epochs} | "
                f"{condition} epoch {epoch_idx}",
                end="",
                flush=True,
            )

            epoch = get_one_epoch(
                subject_id=subject_id,
                condition=condition,
                epoch_idx=epoch_idx,
            )

            features = (
                compute_tfr_features(
                    epoch
                )
            )

            X.append(
                features
            )

            y.append(
                label
            )

            # resting 40 and thinking 40
            # belong to the same group.
            groups.append(
                epoch_idx
            )

    print()

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    y = np.asarray(
        y,
        dtype=int,
    )

    groups = np.asarray(
        groups,
        dtype=int,
    )

    print(
        f"X shape: {X.shape}"
    )

    print(
        f"y shape: {y.shape}"
    )

    print(
        "Class counts:",
        np.bincount(y),
    )

    np.savez_compressed(
        cache_path,
        X=X,
        y=y,
        groups=groups,
    )

    print(
        f"Feature cache saved:\n"
        f"{cache_path}"
    )

    return (
        X,
        y,
        groups,
    )


# ============================================================
# SVM
# ============================================================

def create_classifier():
    """
    Standardization happens INSIDE the CV fold.

    This prevents information leakage from the test fold.
    """

    classifier = Pipeline(
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

    return classifier


# ============================================================
# CROSS VALIDATION
# ============================================================

def evaluate_cv(
    X,
    y,
    groups,
):
    """
    5-fold subject-dependent CV.

    Grouping is by epoch index.

    Resting epoch N and Thinking epoch N
    always remain together in train or test.
    """

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

        model = (
            create_classifier()
        )

        model.fit(
            X[train_idx],
            y[train_idx],
        )

        predictions = (
            model.predict(
                X[test_idx]
            )
        )

        decision_scores = (
            model.decision_function(
                X[test_idx]
            )
        )

        fold_accuracy = (
            accuracy_score(
                y[test_idx],
                predictions,
            )
        )

        fold_balanced_accuracy = (
            balanced_accuracy_score(
                y[test_idx],
                predictions,
            )
        )

        fold_auc = (
            roc_auc_score(
                y[test_idx],
                decision_scores,
            )
        )

        fold_rows.append(
            {
                "fold": fold_idx,
                "accuracy": fold_accuracy,
                "balanced_accuracy": (
                    fold_balanced_accuracy
                ),
                "roc_auc": fold_auc,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }
        )

        all_true.extend(
            y[test_idx]
        )

        all_pred.extend(
            predictions
        )

        all_scores.extend(
            decision_scores
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

    accuracy = (
        accuracy_score(
            all_true,
            all_pred,
        )
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            all_true,
            all_pred,
        )
    )

    auc = (
        roc_auc_score(
            all_true,
            all_scores,
        )
    )

    cm = confusion_matrix(
        all_true,
        all_pred,
        labels=[0, 1],
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": (
            balanced_accuracy
        ),
        "roc_auc": auc,
        "confusion_matrix": cm,
        "fold_rows": fold_rows,
    }


# ============================================================
# LABEL PERMUTATION
# ============================================================

def permute_labels_within_epoch_pairs(
    y,
    groups,
    rng,
):
    """
    Important sanity check.

    Every epoch pair originally contains:

        one resting
        one thinking

    For every pair we randomly decide whether
    to swap these two labels.

    Therefore the dataset remains perfectly balanced
    but the real EEG-condition relationship is destroyed.
    """

    y_permuted = y.copy()

    for group_id in np.unique(
        groups
    ):

        indices = np.where(
            groups == group_id
        )[0]

        if len(indices) != 2:
            raise RuntimeError(
                f"Expected 2 samples for group "
                f"{group_id}, got {len(indices)}"
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
    """
    Run a small label-shuffle sanity check.

    For publication we can later increase
    N_PERMUTATIONS substantially.
    """

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
            permute_labels_within_epoch_pairs(
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

    # Standard permutation-test p-value.
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
        build_subject_dataset(
            subject_id
        )
    )

    print()
    print(
        "Running real-label "
        "cross-validation..."
    )

    real_result = evaluate_cv(
        X,
        y,
        groups,
    )

    cm = (
        real_result[
            "confusion_matrix"
        ]
    )

    print()
    print(
        f"Accuracy: "
        f"{real_result['accuracy']:.4f}"
    )

    print(
        f"Balanced accuracy: "
        f"{real_result['balanced_accuracy']:.4f}"
    )

    print(
        f"ROC AUC: "
        f"{real_result['roc_auc']:.4f}"
    )

    print()
    print(
        "Confusion matrix"
    )

    print(
        "rows = true, columns = predicted"
    )

    print(
        "[[rest->rest, rest->think],"
    )

    print(
        " [think->rest, think->think]]"
    )

    print()
    print(cm)

    print()
    print(
        "Running label-shuffle sanity check..."
    )

    (
        permutation_accuracies,
        permutation_p,
    ) = run_permutation_test(
        X,
        y,
        groups,
        real_result[
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
        f"Permutation mean accuracy: "
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

    subject_row = {
        "subject": subject_id,

        "accuracy": (
            real_result[
                "accuracy"
            ]
        ),

        "balanced_accuracy": (
            real_result[
                "balanced_accuracy"
            ]
        ),

        "roc_auc": (
            real_result[
                "roc_auc"
            ]
        ),

        "resting_correct": int(
            cm[0, 0]
        ),

        "resting_as_thinking": int(
            cm[0, 1]
        ),

        "thinking_as_resting": int(
            cm[1, 0]
        ),

        "thinking_correct": int(
            cm[1, 1]
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

    fold_rows = []

    for row in real_result[
        "fold_rows"
    ]:

        fold_rows.append(
            {
                "subject": subject_id,
                **row,
            }
        )

    permutation_rows = []

    for index, accuracy in enumerate(
        permutation_accuracies,
        start=1,
    ):

        permutation_rows.append(
            {
                "subject": subject_id,
                "permutation": index,
                "accuracy": accuracy,
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

    subject_rows = []
    fold_rows = []
    permutation_rows = []

    for subject_id in SUBJECTS:

        (
            subject_row,
            subject_fold_rows,
            subject_permutation_rows,
        ) = process_subject(
            subject_id
        )

        subject_rows.append(
            subject_row
        )

        fold_rows.extend(
            subject_fold_rows
        )

        permutation_rows.extend(
            subject_permutation_rows
        )

        # Save after every subject.
        pd.DataFrame(
            subject_rows
        ).to_csv(
            SUBJECT_SUMMARY_PATH,
            index=False,
        )

        pd.DataFrame(
            fold_rows
        ).to_csv(
            FOLD_RESULTS_PATH,
            index=False,
        )

        pd.DataFrame(
            permutation_rows
        ).to_csv(
            PERMUTATION_RESULTS_PATH,
            index=False,
        )

        print()
        print(
            f"Progress saved after "
            f"Subject {subject_id}"
        )

    subject_df = pd.DataFrame(
        subject_rows
    )

    print()
    print("=" * 70)
    print(
        "FINAL SUBJECT SUMMARY"
    )
    print("=" * 70)

    print(
        subject_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print(
        "OVERALL"
    )
    print("=" * 70)

    print(
        f"Mean accuracy: "
        f"{subject_df['accuracy'].mean():.4f}"
    )

    print(
        f"Std accuracy: "
        f"{subject_df['accuracy'].std():.4f}"
    )

    print(
        f"Mean balanced accuracy: "
        f"{subject_df['balanced_accuracy'].mean():.4f}"
    )

    print(
        f"Mean ROC AUC: "
        f"{subject_df['roc_auc'].mean():.4f}"
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