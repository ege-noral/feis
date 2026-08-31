from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import EEG_CHANNELS
from data_loader import get_one_epoch

from preprocessing.demean import demean

from quality_control.repetition_consistency.morlet_tfr import (
    compute_morlet_tfr,
)

from quality_control.repetition_consistency.self_similarity import (
    compute_tfr_self_similarity,
)


# ============================================================
# SETTINGS
# ============================================================

FS = 256

SUBJECT_ID = 1
CHANNEL_NAME = "AF3"
BAND_NAME = "theta"

BAND_LOW = 4.0
BAND_HIGH = 7.0

CONDITION = "resting"

WINDOW_SECONDS = 0.40
STEP_SECONDS = 0.15

MIN_TIME_SEPARATION_SECONDS = 0.50

MIN_GROUP_SIZE = 4
MAX_GROUP_SIZE = 5

MIN_NEIGHBORS_PER_NODE = 2


# We will use the result produced by the band experiment.
RESULTS_CSV = Path(
    "validation/repetition_consistency/archive/"
    "real_all_channels_bands_calibrated_epoch_results.csv"
)


# ============================================================
# FIND A POSITIVE RESTING EPOCH
# ============================================================

def find_positive_epoch():
    """
    Automatically finds a resting epoch where the existing
    detector reported 4 or 5 repetitions.

    Also retrieves the already-calibrated similarity threshold.
    """

    if not RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"Could not find:\n{RESULTS_CSV}"
        )

    df = pd.read_csv(RESULTS_CSV)

    subset = df[
        (df["subject"] == SUBJECT_ID)
        & (df["channel"] == CHANNEL_NAME)
        & (df["band"] == BAND_NAME)
        & (df["condition"] == CONDITION)
        & (df["detected_repeats"] >= MIN_GROUP_SIZE)
    ].copy()

    if len(subset) == 0:
        raise RuntimeError(
            "No positive resting epoch was found "
            "for the selected subject/channel/band."
        )

    # First positive epoch
    row = subset.iloc[0]

    epoch_idx = int(row["epoch_idx"])
    threshold = float(row["similarity_threshold"])
    detected_repeats = int(row["detected_repeats"])

    return epoch_idx, threshold, detected_repeats


# ============================================================
# CREATE GRAPH FROM SIMILARITY MATRIX
# ============================================================

def build_adjacency(
    similarity_matrix,
    window_centers,
    threshold,
):
    """
    Two windows are connected only if:

    1. similarity >= threshold
    2. temporal separation >= minimum separation
    """

    n_windows = similarity_matrix.shape[0]

    adjacency = np.zeros(
        (n_windows, n_windows),
        dtype=bool,
    )

    for i in range(n_windows):

        for j in range(i + 1, n_windows):

            time_difference = abs(
                window_centers[j]
                - window_centers[i]
            )

            if (
                time_difference
                >= MIN_TIME_SEPARATION_SECONDS
                and similarity_matrix[i, j]
                >= threshold
            ):
                adjacency[i, j] = True
                adjacency[j, i] = True

    return adjacency


# ============================================================
# FIND VALID MOTIF GROUPS
# ============================================================

def group_is_valid(indices, adjacency):
    """
    Same important rule as our detector:

    every node in the group must have at least
    MIN_NEIGHBORS_PER_NODE neighbors inside that group.
    """

    indices = list(indices)

    subgraph = adjacency[
        np.ix_(indices, indices)
    ]

    degrees = np.sum(
        subgraph,
        axis=1,
    )

    return np.all(
        degrees >= MIN_NEIGHBORS_PER_NODE
    )


def group_score(
    indices,
    similarity_matrix,
    adjacency,
):
    """
    Score a candidate group.

    Higher score means stronger mutual similarity.

    Only graph edges are included.
    """

    values = []

    for i, j in combinations(indices, 2):

        if adjacency[i, j]:
            values.append(
                similarity_matrix[i, j]
            )

    if len(values) == 0:
        return -np.inf

    return float(
        np.mean(values)
    )


def find_best_group(
    similarity_matrix,
    adjacency,
):
    """
    Searches for valid 5-member groups first,
    then 4-member groups.

    Within the same size, chooses the group
    with the strongest average edge similarity.
    """

    n_windows = similarity_matrix.shape[0]

    best_group = None
    best_score = -np.inf

    # Prefer larger group first.
    for group_size in range(
        MAX_GROUP_SIZE,
        MIN_GROUP_SIZE - 1,
        -1,
    ):

        candidates_found = False

        for indices in combinations(
            range(n_windows),
            group_size,
        ):

            if not group_is_valid(
                indices,
                adjacency,
            ):
                continue

            candidates_found = True

            score = group_score(
                indices,
                similarity_matrix,
                adjacency,
            )

            if score > best_score:

                best_group = indices
                best_score = score

        if candidates_found:
            # If a valid 5-group exists,
            # do not drop to size 4.
            break

    if best_group is None:
        return None, None

    return list(best_group), best_score


# ============================================================
# PRINT GROUP DETAILS
# ============================================================

def print_group_details(
    group_indices,
    window_centers,
    similarity_matrix,
    adjacency,
):
    print()
    print("=" * 70)
    print("SELECTED MOTIF GROUP")
    print("=" * 70)

    group_times = window_centers[
        group_indices
    ]

    print()
    print("Window indices:")
    print(group_indices)

    print()
    print("Window centers (seconds):")

    for number, time_value in enumerate(
        group_times,
        start=1,
    ):
        print(
            f"  Motif {number}: "
            f"{time_value:.3f} s"
        )

    print()
    print("Time differences between consecutive motifs:")

    differences = np.diff(
        group_times
    )

    for i, difference in enumerate(
        differences,
        start=1,
    ):
        print(
            f"  {i} -> {i + 1}: "
            f"{difference:.3f} s"
        )

    print()
    print("Pairwise similarity matrix:")

    group_similarity = similarity_matrix[
        np.ix_(
            group_indices,
            group_indices,
        )
    ]

    print(
        np.round(
            group_similarity,
            3,
        )
    )

    print()
    print("Adjacency matrix:")

    group_adjacency = adjacency[
        np.ix_(
            group_indices,
            group_indices,
        )
    ]

    print(
        group_adjacency.astype(int)
    )


# ============================================================
# PLOTS
# ============================================================

def plot_theta_tfr(
    band_power,
    band_freqs,
    group_indices,
    window_centers,
    epoch_idx,
):
    """
    Shows the theta TFR and marks the selected windows.
    """

    times = np.arange(
        band_power.shape[1]
    ) / FS

    log_power = np.log1p(
        band_power
    )

    plt.figure(
        figsize=(14, 5)
    )

    plt.imshow(
        log_power,
        aspect="auto",
        origin="lower",
        extent=[
            times[0],
            times[-1],
            band_freqs[0],
            band_freqs[-1],
        ],
    )

    plt.colorbar(
        label="Log power"
    )

    half_window = (
        WINDOW_SECONDS / 2.0
    )

    for number, idx in enumerate(
        group_indices,
        start=1,
    ):

        center = window_centers[idx]

        start = (
            center
            - half_window
        )

        end = (
            center
            + half_window
        )

        plt.axvspan(
            start,
            end,
            alpha=0.20,
        )

        plt.axvline(
            center,
            linestyle="--",
            linewidth=1.5,
        )

        plt.text(
            center,
            band_freqs[-1],
            str(number),
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

    plt.title(
        f"Subject {SUBJECT_ID} | "
        f"{CHANNEL_NAME} | "
        f"{CONDITION} | "
        f"Epoch {epoch_idx}\n"
        f"Theta TFR and detected motif windows"
    )

    plt.tight_layout()
    plt.show()


def plot_similarity_matrix(
    similarity_matrix,
    group_indices,
):
    """
    Shows where the selected motif windows sit
    inside the full self-similarity matrix.
    """

    plt.figure(
        figsize=(8, 7)
    )

    plt.imshow(
        similarity_matrix,
        origin="lower",
        aspect="equal",
        vmin=-1,
        vmax=1,
    )

    plt.colorbar(
        label="TFR similarity"
    )

    for idx in group_indices:

        plt.axhline(
            idx,
            linewidth=0.6,
            alpha=0.5,
        )

        plt.axvline(
            idx,
            linewidth=0.6,
            alpha=0.5,
        )

    plt.scatter(
        group_indices,
        group_indices,
        marker="o",
        s=80,
        facecolors="none",
        edgecolors="black",
    )

    plt.xlabel("Window index")
    plt.ylabel("Window index")

    plt.title(
        "Theta TFR self-similarity matrix"
    )

    plt.tight_layout()
    plt.show()


def plot_selected_patches(
    band_power,
    band_freqs,
    group_indices,
    window_centers,
):
    """
    Plots the actual TFR patches that the graph
    considers members of the same motif group.
    """

    window_samples = int(
        round(
            WINDOW_SECONDS
            * FS
        )
    )

    n_patches = len(
        group_indices
    )

    fig, axes = plt.subplots(
        1,
        n_patches,
        figsize=(
            3 * n_patches,
            4,
        ),
        sharey=True,
    )

    if n_patches == 1:
        axes = [axes]

    # Same color scale for all patches.
    all_patch_values = []

    patches = []

    for idx in group_indices:

        center_seconds = (
            window_centers[idx]
        )

        center_sample = int(
            round(
                center_seconds
                * FS
            )
        )

        start = (
            center_sample
            - window_samples // 2
        )

        end = (
            start
            + window_samples
        )

        start = max(
            0,
            start,
        )

        end = min(
            band_power.shape[1],
            end,
        )

        patch = np.log1p(
            band_power[
                :,
                start:end,
            ]
        )

        patches.append(
            patch
        )

        all_patch_values.append(
            patch.ravel()
        )

    all_values = np.concatenate(
        all_patch_values
    )

    vmin = np.min(
        all_values
    )

    vmax = np.max(
        all_values
    )

    image = None

    for number, (
        axis,
        patch,
        idx,
    ) in enumerate(
        zip(
            axes,
            patches,
            group_indices,
        ),
        start=1,
    ):

        center = (
            window_centers[idx]
        )

        image = axis.imshow(
            patch,
            aspect="auto",
            origin="lower",
            extent=[
                -WINDOW_SECONDS / 2,
                WINDOW_SECONDS / 2,
                band_freqs[0],
                band_freqs[-1],
            ],
            vmin=vmin,
            vmax=vmax,
        )

        axis.set_title(
            f"{number}\n"
            f"t={center:.2f}s"
        )

        axis.set_xlabel(
            "Relative time (s)"
        )

    axes[0].set_ylabel(
        "Frequency (Hz)"
    )

    fig.colorbar(
        image,
        ax=axes,
        label="Log power",
        shrink=0.8,
    )

    fig.suptitle(
        "TFR patches classified as the same motif",
        fontsize=14,
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("RESTING THETA MOTIF INSPECTION")
    print("=" * 70)

    epoch_idx, threshold, original_detection = (
        find_positive_epoch()
    )

    print()
    print(
        f"Subject: {SUBJECT_ID}"
    )

    print(
        f"Channel: {CHANNEL_NAME}"
    )

    print(
        f"Band: {BAND_LOW}-{BAND_HIGH} Hz"
    )

    print(
        f"Condition: {CONDITION}"
    )

    print(
        f"Epoch: {epoch_idx}"
    )

    print(
        f"Existing detector output: "
        f"{original_detection}"
    )

    print(
        f"Similarity threshold: "
        f"{threshold:.6f}"
    )

    # --------------------------------------------------------
    # LOAD EEG
    # --------------------------------------------------------

    epoch = get_one_epoch(
        subject_id=SUBJECT_ID,
        condition=CONDITION,
        epoch_idx=epoch_idx,
    )

    epoch = demean(
        epoch,
        axis=-1,
    )

    channel_idx = list(
        EEG_CHANNELS
    ).index(
        CHANNEL_NAME
    )

    signal = epoch[
        channel_idx
    ]

    # --------------------------------------------------------
    # MORLET
    # --------------------------------------------------------

    freqs, power = (
        compute_morlet_tfr(
            signal,
            fs=FS,
        )
    )

    band_mask = (
        (freqs >= BAND_LOW)
        & (freqs <= BAND_HIGH)
    )

    band_freqs = freqs[
        band_mask
    ]

    band_power = power[
        band_mask,
        :
    ]

    # --------------------------------------------------------
    # SELF-SIMILARITY
    # --------------------------------------------------------

    similarity_matrix, window_centers = (
        compute_tfr_self_similarity(
            power=band_power,
            fs=FS,
            window_seconds=WINDOW_SECONDS,
            step_seconds=STEP_SECONDS,
        )
    )

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    adjacency = build_adjacency(
        similarity_matrix,
        window_centers,
        threshold,
    )

    group_indices, score = (
        find_best_group(
            similarity_matrix,
            adjacency,
        )
    )

    if group_indices is None:

        print()
        print(
            "WARNING:"
        )

        print(
            "The diagnostic graph reconstruction "
            "did not find a valid group."
        )

        print(
            "If the original detector still reports "
            "4/5 here, its group-selection logic differs "
            "from this diagnostic reconstruction."
        )

        return

    print()
    print(
        f"Recovered group size: "
        f"{len(group_indices)}"
    )

    print(
        f"Mean graph-edge similarity: "
        f"{score:.4f}"
    )

    if (
        len(group_indices)
        != original_detection
    ):

        print()
        print(
            "NOTE:"
        )

        print(
            "Recovered group size is not identical "
            "to the original detector output."
        )

        print(
            "The plots are still useful for inspecting "
            "the high-similarity resting structure, "
            "but we should then inspect the exact "
            "motif_group_detector implementation."
        )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print_group_details(
        group_indices,
        window_centers,
        similarity_matrix,
        adjacency,
    )

    # --------------------------------------------------------
    # ROUGH MORLET TEMPORAL SUPPORT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ROUGH MORLET TIME SCALE")
    print("=" * 70)

    print()

    for frequency in band_freqs:

        approximate_duration = (
            6.0 / frequency
        )

        print(
            f"{frequency:4.1f} Hz : "
            f"6 cycles ≈ "
            f"{approximate_duration:.3f} s"
        )

    print()
    print(
        "Current minimum motif separation = "
        f"{MIN_TIME_SEPARATION_SECONDS:.3f} s"
    )

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    plot_theta_tfr(
        band_power,
        band_freqs,
        group_indices,
        window_centers,
        epoch_idx,
    )

    plot_similarity_matrix(
        similarity_matrix,
        group_indices,
    )

    plot_selected_patches(
        band_power,
        band_freqs,
        group_indices,
        window_centers,
    )


if __name__ == "__main__":
    main()