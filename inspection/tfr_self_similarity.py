import numpy as np
import matplotlib.pyplot as plt

from constants import EEG_CHANNELS
from data_loader import get_one_epoch

from preprocessing.demean import demean

from quality_control.repetition_consistency.morlet_tfr import (
    compute_morlet_tfr
)

from quality_control.repetition_consistency.self_similarity import (
    compute_tfr_self_similarity
)

from quality_control.repetition_consistency.repetition_detection import (
    detect_repetition_candidates
)

SUBJECT_ID = 8
EPOCH_IDX = 44
CHANNEL_NAME = "P7"

FS = 256

WINDOW_SECONDS = 0.75
STEP_SECONDS = 0.10


# --------------------------------------------------
# 1. Thinking epoch'u yükle
# --------------------------------------------------

epoch = get_one_epoch(
    subject_id=SUBJECT_ID,
    condition="resting",
    epoch_idx=EPOCH_IDX,
)

channel_idx = EEG_CHANNELS.index(
    CHANNEL_NAME
)

signal = epoch[channel_idx]

print("Signal shape:", signal.shape)


# --------------------------------------------------
# 2. DC offset'i kaldır
# --------------------------------------------------

# signal = demean(signal)


# --------------------------------------------------
# 3. Morlet TFR
# --------------------------------------------------

freqs, power = compute_morlet_tfr(
    signal,
    fs=FS,
)

print("TFR shape:", power.shape)


# --------------------------------------------------
# 4. Sliding-window self-similarity
# --------------------------------------------------

similarity_matrix, window_centers = (
    compute_tfr_self_similarity(
        power,
        fs=FS,
        window_seconds=WINDOW_SECONDS,
        step_seconds=STEP_SECONDS,
    )
)

print(
    "Similarity matrix shape:",
    similarity_matrix.shape
)


# --------------------------------------------------
# 5. TFR görselleştir
# --------------------------------------------------

time = np.arange(
    len(signal)
) / FS

log_power = 10 * np.log10(
    power + 1e-12
)

plt.figure(
    figsize=(12, 5)
)

plt.imshow(
    log_power,
    aspect="auto",
    origin="lower",
    extent=[
        time[0],
        time[-1],
        freqs[0],
        freqs[-1],
    ],
)

plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")

plt.title(
    f"Subject {SUBJECT_ID} | "
    f"Epoch {EPOCH_IDX} | "
    f"{CHANNEL_NAME} | Morlet TFR"
)

plt.colorbar(
    label="Power (dB)"
)

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 6. Self-similarity matrix
# --------------------------------------------------

plt.figure(
    figsize=(8, 7)
)

plt.imshow(
    similarity_matrix,
    origin="lower",
    aspect="auto",
    extent=[
        window_centers[0],
        window_centers[-1],
        window_centers[0],
        window_centers[-1],
    ],
    vmin=-1,
    vmax=1,
)

plt.xlabel("Time (s)")
plt.ylabel("Time (s)")

plt.title(
    f"Subject {SUBJECT_ID} | "
    f"Epoch {EPOCH_IDX} | "
    f"{CHANNEL_NAME} | TFR Self-Similarity"
)

plt.colorbar(
    label="Similarity"
)

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 7. Overlap nedeniyle oluşan yakın-diagonal benzerlikleri
#    görmezden gel ve uzak zamanlardaki en benzer çiftleri bul
# --------------------------------------------------

time_difference = np.abs(
    window_centers[:, None]
    - window_centers[None, :]
)

nonlocal_similarity = (
    similarity_matrix.copy()
)

# 0.75 saniyeden yakın pencereler zaten birbirleriyle
# büyük ölçüde overlap ediyor.
nonlocal_similarity[
    time_difference < WINDOW_SECONDS
] = np.nan


# --------------------------------------------------
# 8. En yüksek non-local similarity çiftlerini yazdır
# --------------------------------------------------

upper_triangle = np.triu(
    np.ones_like(
        nonlocal_similarity,
        dtype=bool
    ),
    k=1,
)

valid_mask = (
    upper_triangle
    & ~np.isnan(nonlocal_similarity)
)

pairs = []

rows, cols = np.where(
    valid_mask
)

for i, j in zip(rows, cols):

    pairs.append(
        (
            nonlocal_similarity[i, j],
            window_centers[i],
            window_centers[j],
        )
    )

pairs.sort(
    reverse=True,
    key=lambda x: x[0]
)


print(
    "\n--- TOP NON-OVERLAPPING "
    "SIMILAR WINDOW PAIRS ---"
)

for similarity, time_1, time_2 in pairs[:15]:

    print(
        f"{time_1:.2f} s  <->  "
        f"{time_2:.2f} s  |  "
        f"similarity = {similarity:.3f}"
    )

plt.figure(figsize=(8, 7))

masked_similarity = np.ma.masked_invalid(
    nonlocal_similarity
)

plt.imshow(
    masked_similarity,
    origin="lower",
    aspect="auto",
    extent=[
        window_centers[0],
        window_centers[-1],
        window_centers[0],
        window_centers[-1],
    ],
)

plt.xlabel("Time (s)")
plt.ylabel("Time (s)")

plt.title(
    f"Subject {SUBJECT_ID} | "
    f"Epoch {EPOCH_IDX} | "
    f"{CHANNEL_NAME} | "
    f"Non-Local TFR Self-Similarity"
)

plt.colorbar(
    label="Similarity"
)

plt.tight_layout()
plt.show()

result = detect_repetition_candidates(
    similarity_matrix,
    window_centers,
    window_seconds=WINDOW_SECONDS,
    similarity_percentile=95.0,
    min_peak_distance_seconds=0.50,
)


print(
    "\nStrong similarity threshold:",
    round(result["threshold"], 3)
)


print(
    "\n--- CANDIDATE REPETITION TIMES ---"
)

for time, score in zip(
    result["peak_times"],
    result["peak_scores"],
):
    print(
        f"{time:.2f} s"
        f" | score = {score:.3f}"
    )

plt.figure(figsize=(12, 4))

plt.plot(
    window_centers,
    result["smoothed_scores"],
)

plt.scatter(
    result["peak_times"],
    result["peak_scores"],
)

plt.xlabel("Time (s)")
plt.ylabel("Repetition candidate score")

plt.title(
    f"Subject {SUBJECT_ID} | "
    f"Epoch {EPOCH_IDX} | "
    f"{CHANNEL_NAME} | "
    f"Repetition Candidate Score"
)

plt.grid()

plt.tight_layout()
plt.show()