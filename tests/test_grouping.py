from preprocessing.data_loader import load_subject_dataframe

from preprocessing.data_loader import group_trials_by_label

def test_group_trials_by_label():

    thinking_df = load_subject_dataframe(
        subject_id=1,
        condition="resting"
    )

    groups = group_trials_by_label(
        thinking_df
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