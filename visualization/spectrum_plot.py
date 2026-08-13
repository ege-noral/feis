import matplotlib.pyplot as plt
from scipy.signal import welch

from constants import EEG_CHANNELS

from helper import ( 
    split_into_epochs
)

from preprocessing.artefact_removal import bandpass_filter

from data_loader import get_epochs


def plot_trial_spectrum(
    subject_id,
    trial_id,
    sfreq=256
):

    resting_trials, thinking_trials = get_epochs(
         subject_id=subject_id
    )

    resting_trial = resting_trials[trial_id]["eeg"]
    thinking_trial = thinking_trials[trial_id]["eeg"]

    thinking_trial = bandpass_filter(
        thinking_trial,
        sfreq=256,
        lowcut=1,
        highcut=40
    )

    resting_trial = bandpass_filter(
        resting_trial,
        sfreq=256,
        lowcut=1,
        highcut=40
    )

    plt.figure(figsize=(10, 6))

    for ch in range(len(EEG_CHANNELS)):

        f, psd_rest = welch(
            resting_trial[ch],
            fs=sfreq,
            nperseg=256
        )

        _, psd_think = welch(
            thinking_trial[ch],
            fs=sfreq,
            nperseg=256
        )

        plt.semilogy(
            f,
            psd_rest,
            alpha=0.4,
            color="blue"
        )

        plt.semilogy(
            f,
            psd_think,
            alpha=0.4,
            color="red"
        )

    plt.xlim(0, 60)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")

    plt.title(
        f"Subject {subject_id} | Trial {trial_id}\n"
        "Blue = Resting, Red = Thinking"
    )

    plt.grid(True)

    plt.show()