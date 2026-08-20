import matplotlib.pyplot as plt

from data_loader import get_epochs

from helper import ( 
    split_into_epochs
)

from preprocessing.artifact_removal import bandpass_filter

from constants import EEG_CHANNELS


def plot_trial(
    subject_id,
    trial_id,
    segment=None
):

    resting_trials, thinking_trials = get_epochs(
         subject_id=subject_id
    )

    thinking_trial = thinking_trials[trial_id]["eeg"]
    resting_trial = resting_trials[trial_id]["eeg"]

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

    n_channels = len(EEG_CHANNELS)

    plt.figure(figsize=(15, 10))

    for ch in range(n_channels):

        plt.subplot(
            n_channels,
            1,
            ch + 1
        )

        plt.plot(
            resting_trial[ch],
            label="Resting",
            alpha=0.8
        )

        plt.plot(
            thinking_trial[ch],
            label="Thinking",
            alpha=0.8
        )

        if segment is not None:

            plt.axvspan(
                segment[0],
                segment[1],
                alpha=0.3
            )

        plt.ylabel(
            EEG_CHANNELS[ch],
            fontsize=8
        )

        if ch == 0:
            plt.legend(fontsize=8)

    plt.suptitle(
        f"Subject {subject_id} | Trial {trial_id}"
    )

    plt.tight_layout()
    plt.show()

plot_trial(
    subject_id=3,
    trial_id=107    
)