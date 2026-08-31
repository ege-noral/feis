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

from quality_control.repetition_consistency.detectors.motif_group_detector import (
    detect_motif_groups,
)


# ============================================================
# SETTINGS
# ============================================================

FS = 256

SUBJECT_ID = 1

# Subject 1'de low-beta için bakacağımız kanal.
# İstersen daha sonra bunu kolayca değiştirebiliriz.
CHANNEL_NAME = "F4"

BAND_NAME = "low_beta"

BAND_LOW = 13.0
BAND_HIGH = 20.0

CONDITION = "resting"

WINDOW_SECONDS = 0.40
STEP_SECONDS = 0.15

MIN_TIME_SEPARATION_SECONDS = 0.50

MIN_GROUP_SIZE = 4
MAX_GROUP_SIZE = 5

MIN_GROUP_DENSITY = 0.0
MIN_NEIGHBORS_PER_NODE = 2

MERGE_TOLERANCE_SECONDS = 0.20


# ============================================================
# RESULTS FILE
# ============================================================

RESULTS_CSV = Path(
    "validation/repetition_consistency/archive/"
    "real_all_channels_bands_calibrated_epoch_results.csv"
)


# ============================================================
# FIND A POSITIVE RESTING EPOCH
# ============================================================

def find_positive_epoch():
    """
    Finds a positive resting low-beta epoch.

    Preference:
        1. detected_repeats == 5
        2. detected_repeats == 4

    Also retrieves the exact calibrated threshold
    previously used for this subject/channel/band.
    """

    if not RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"Could not find results CSV:\n{RESULTS_CSV}"
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
            "No positive resting low-beta epoch was found "
            "for the selected subject/channel."
        )

    # Prefer an epoch where the original detector found 5.
    five_rows = subset[
        subset["detected_repeats"] == 5
    ]

    if len(five_rows) > 0:
        row = five_rows.iloc[0]
    else:
        row = subset.iloc[0]

    epoch_idx = int(
        row["epoch_idx"]
    )

    threshold = float(
        row["similarity_threshold"]
    )

    detected_repeats = int(
        row["detected_repeats"]
    )

    return (
        epoch_idx,
        threshold,
        detected_repeats,
    )


# ============================================================
# GRAPH
# ============================================================

def build_adjacency(
    similarity_matrix,
    window_centers,
    threshold,
):
    """
    Create the similarity graph.

    Two windows receive an edge only if:

    1. similarity >= threshold
    2. their centers are at least 0.5 s apart
    """

    n_windows = (
        similarity_matrix.shape[0]
    )

    adjacency = np.zeros(
        (n_windows, n_windows),
        dtype=bool,
    )

    for i in range(n_windows):

        for j in range(
            i + 1,
            n_windows,
        ):

            time_difference = abs(
                window_centers[j]
                - window_centers[i]
            )

            similarity = (
                similarity_matrix[i, j]
            )

            if (
                time_difference
                >= MIN_TIME_SEPARATION_SECONDS
                and similarity >= threshold
            ):
                adjacency[i, j] = True
                adjacency[j, i] = True

    return adjacency


# ============================================================
# CANDIDATE GROUP SEARCH
# ============================================================

def group_is_valid(
    indices,
    adjacency,
):
    """
    Same min-neighbor idea as the main detector.

    Every member must have at least 2 graph neighbors
    inside the candidate group.
    """

    indices = list(indices)

    subgraph = adjacency[
        np.ix_(
            indices,
            indices,
        )
    ]

    degrees = np.sum(
        subgraph,
        axis=1,
    )

    return bool(
        np.all(
            degrees
            >= MIN_NEIGHBORS_PER_NODE
        )
    )


def group_score(
    indices,
    similarity_matrix,
    adjacency,
):
    """
    Mean similarity across valid graph edges.
    """

    values = []

    for i, j in combinations(
        indices,
        2,
    ):

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
    Diagnostic reconstruction.

    First searches for valid size-5 groups.
    If none exists, searches size-4 groups.

    This is for visualization.
    The real detector is also run separately below.
    """

    n_windows = (
        similarity_matrix.shape[0]
    )

    for group_size in range(
        MAX_GROUP_SIZE,
        MIN_GROUP_SIZE - 1,
        -1,
    ):

        best_group = None
        best_score = -np.inf

        for indices in combinations(
            range(n_windows),
            group_size,
        ):

            if not group_is_valid(
                indices,
                adjacency,
            ):
                continue

            score = group_score(
                indices,
                similarity_matrix,
                adjacency,
            )

            if score > best_score:

                best_group = list(
                    indices
                )

                best_score = score

        if best_group is not None:

            return (
                best_group,
                best_score,
            )

    return None, None


# ============================================================
# EXACT DETECTOR
# ============================================================

def run_exact_detector(
    similarity_matrix,
    window_centers,
    threshold,
):
    """
    Run the actual detector used in the experiment.

    This lets us compare the diagnostic reconstruction
    with the real algorithm.
    """

    result = detect_motif_groups(
        similarity_matrix=similarity_matrix,
        window_centers_seconds=window_centers,
        similarity_threshold=threshold,
        min_time_separation_seconds=MIN_TIME_SEPARATION_SECONDS,
        min_group_size=MIN_GROUP_SIZE,
        max_group_size=MAX_GROUP_SIZE,
        min_group_density=MIN_GROUP_DENSITY,
        min_neighbors_per_node=MIN_NEIGHBORS_PER_NODE,
        merge_tolerance_seconds=MERGE_TOLERANCE_SECONDS,
    )

    return result


# ============================================================
# PRINT DETAILS
# ============================================================

def print_group_details(
    group_indices,
    window_centers,
    similarity_matrix,
    adjacency,
):
    print()
    print("=" * 70)
    print("DIAGNOSTIC CANDIDATE GROUP")
    print("=" * 70)

    group_times = (
        window_centers[
            group_indices
        ]
    )

    print()
    print("Window indices:")
    print(group_indices)

    print()
    print("Window centers:")

    for number, time_value in enumerate(
        group_times,
        start=1,
    ):

        print(
            f"  Motif {number}: "
            f"{time_value:.3f} s"
        )

    print()
    print(
        "Consecutive time separations:"
    )

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

    group_similarity = (
        similarity_matrix[
            np.ix_(
                group_indices,
                group_indices,
            )
        ]
    )

    print()
    print(
        "Pairwise similarity matrix:"
    )

    print(
        np.round(
            group_similarity,
            3,
        )
    )

    group_adjacency = (
        adjacency[
            np.ix_(
                group_indices,
                group_indices,
            )
        ]
    )

    print()
    print(
        "Adjacency matrix:"
    )

    print(
        group_adjacency.astype(int)
    )


# ============================================================
# PLOT 1
# ============================================================

def plot_band_tfr(
    band_power,
    band_freqs,
    group_indices,
    window_centers,
    epoch_idx,
):
    """
    Full 5-second low-beta TFR.

    Detected candidate windows are marked.
    """

    times = (
        np.arange(
            band_power.shape[1]
        )
        / FS
    )

    log_power = np.log1p(
        band_power
    )

    plt.figure(
        figsize=(14, 5)
    )

    image = plt.imshow(
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
        image,
        label="Log power",
    )

    half_window = (
        WINDOW_SECONDS / 2.0
    )

    for number, idx in enumerate(
        group_indices,
        start=1,
    ):

        center = (
            window_centers[idx]
        )

        start = (
            center - half_window
        )

        end = (
            center + half_window
        )

        plt.axvspan(
            start,
            end,
            alpha=0.18,
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

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Frequency (Hz)"
    )

    plt.title(
        f"Subject {SUBJECT_ID} | "
        f"{CHANNEL_NAME} | "
        f"{CONDITION} | "
        f"Epoch {epoch_idx}\n"
        f"Low-beta TFR and candidate motif windows"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# PLOT 2
# ============================================================

def plot_similarity_matrix(
    similarity_matrix,
    group_indices,
    window_centers,
):
    """
    Full self-similarity matrix.
    """

    plt.figure(
        figsize=(8, 7)
    )

    image = plt.imshow(
        similarity_matrix,
        origin="lower",
        aspect="equal",
        vmin=-1,
        vmax=1,
    )

    plt.colorbar(
        image,
        label="TFR similarity",
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

    plt.xlabel(
        "Window index"
    )

    plt.ylabel(
        "Window index"
    )

    plt.title(
        "Low-beta TFR self-similarity matrix"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# PLOT 3
# ============================================================

def plot_selected_patches(
    band_power,
    band_freqs,
    group_indices,
    window_centers,
):
    """
    Plot selected low-beta TFR patches side-by-side
    with one common power scale.
    """

    window_samples = int(
        round(
            WINDOW_SECONDS * FS
        )
    )

    patches = []
    patch_times = []

    for idx in group_indices:

        center_seconds = (
            window_centers[idx]
        )

        center_sample = int(
            round(
                center_seconds * FS
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
            start,
            0,
        )

        end = min(
            end,
            band_power.shape[1],
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

        patch_times.append(
            center_seconds
        )

    all_values = np.concatenate(
        [
            patch.ravel()
            for patch in patches
        ]
    )

    vmin = float(
        np.min(all_values)
    )

    vmax = float(
        np.max(all_values)
    )

    n_patches = len(
        patches
    )

    fig, axes = plt.subplots(
        1,
        n_patches,
        figsize=(
            3.2 * n_patches,
            4.5,
        ),
        sharey=True,
    )

    if n_patches == 1:
        axes = [axes]

    image = None

    for number, (
        axis,
        patch,
        center_seconds,
    ) in enumerate(
        zip(
            axes,
            patches,
            patch_times,
        ),
        start=1,
    ):

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
            f"Motif {number}\n"
            f"t = {center_seconds:.2f} s"
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
        shrink=0.80,
    )

    fig.suptitle(
        "Low-beta TFR patches considered recurrent",
        fontsize=14,
    )

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("RESTING LOW-BETA MOTIF INSPECTION")
    print("=" * 70)

    (
        epoch_idx,
        threshold,
        original_detection,
    ) = find_positive_epoch()

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
        f"CSV detector output: "
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
    # MORLET TFR
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

    band_freqs = (
        freqs[
            band_mask
        ]
    )

    band_power = (
        power[
            band_mask,
            :
        ]
    )

    # --------------------------------------------------------
    # SELF SIMILARITY
    # --------------------------------------------------------

    (
        similarity_matrix,
        window_centers,
    ) = compute_tfr_self_similarity(
        power=band_power,
        fs=FS,
        window_seconds=WINDOW_SECONDS,
        step_seconds=STEP_SECONDS,
    )

    # --------------------------------------------------------
    # RUN REAL DETECTOR
    # --------------------------------------------------------

    exact_result = (
        run_exact_detector(
            similarity_matrix,
            window_centers,
            threshold,
        )
    )

    print()
    print("=" * 70)
    print("ACTUAL DETECTOR RESULT")
    print("=" * 70)

    print()
    print(exact_result)

    # --------------------------------------------------------
    # DIAGNOSTIC GRAPH
    # --------------------------------------------------------

    adjacency = build_adjacency(
        similarity_matrix,
        window_centers,
        threshold,
    )

    (
        group_indices,
        score,
    ) = find_best_group(
        similarity_matrix,
        adjacency,
    )

    if group_indices is None:

        print()
        print(
            "No diagnostic motif group "
            "could be reconstructed."
        )

        return

    print()
    print(
        f"Diagnostic candidate group size: "
        f"{len(group_indices)}"
    )

    print(
        f"Mean graph-edge similarity: "
        f"{score:.4f}"
    )

    # --------------------------------------------------------
    # DETAILS
    # --------------------------------------------------------

    print_group_details(
        group_indices,
        window_centers,
        similarity_matrix,
        adjacency,
    )

    # --------------------------------------------------------
    # MORLET TEMPORAL SCALE
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
            f"{frequency:5.1f} Hz : "
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

    plot_band_tfr(
        band_power,
        band_freqs,
        group_indices,
        window_centers,
        epoch_idx,
    )

    plot_similarity_matrix(
        similarity_matrix,
        group_indices,
        window_centers,
    )

    plot_selected_patches(
        band_power,
        band_freqs,
        group_indices,
        window_centers,
    )


if __name__ == "__main__":
    main()