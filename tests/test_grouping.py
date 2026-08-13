from data_loader import get_thinking_trial_epochs

from helper import (
    group_trials_by_label,
    split_into_epochs
)

def test_group_trials_by_label():

    epochs = get_thinking_trial_epochs(subject_id=1)

    groups = group_trials_by_label(
        epochs
    )

    for label, trials in groups.items():

        print(f"\nLabel: {label}")
        print(
            f"Trial count: {len(trials)}"
        )

        print(
            "Trial ids:",
            [t["trial"] for t in trials]
        )

        print(
            "EEG shape:",
            trials[0]["eeg"]["eeg"].shape
        )

       

if __name__ == "__main__":
    test_group_trials_by_label()