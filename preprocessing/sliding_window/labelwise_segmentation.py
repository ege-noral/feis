from collections import defaultdict

from preprocessing.sliding_window import (
    create_sliding_windows,
    find_contiguous_selected_windows
)

from inspection import compare_window_variances

from .merge import merge_segments

from helper import ( 
    compute_window_variance,
    split_into_epochs,
    group_trials_by_label
)

from data_loader import get_epochs

from constants import EEG_CHANNELS


def extract_label_channel_segments(
    subject_id,
    label,
    ratio_threshold=8.0,
    window_size=13,
    overlap=0.5
):

    resting_trials, thinking_trials = get_epochs(
         subject_id=subject_id
    )

    resting_groups = group_trials_by_label(
        resting_trials
    )

    thinking_groups = group_trials_by_label(
        thinking_trials
    )

    resting_trials = resting_groups[label]
    thinking_trials = thinking_groups[label]

    result = {
        "label": label,
        "channels": defaultdict(list)
    }

    for resting_trial, thinking_trial in zip(
        resting_trials,
        thinking_trials
    ):

        trial_id = thinking_trial["trial"]

        resting_eeg = resting_trial["eeg"]["eeg"]
        thinking_eeg = thinking_trial["eeg"]["eeg"]

        resting_windows = create_sliding_windows(
            resting_eeg,
            window_size=window_size,
            overlap=overlap
        )

        thinking_windows = create_sliding_windows(
            thinking_eeg,
            window_size=window_size,
            overlap=overlap
        )

        resting_variance = compute_window_variance(
            resting_windows
        )

        thinking_variance = compute_window_variance(
            thinking_windows
        )

        results = compare_window_variances(
            resting_variance,
            thinking_variance,
            ratio_threshold=ratio_threshold
        )

        contiguous = find_contiguous_selected_windows(
            results
        )

        for channel in EEG_CHANNELS:

            segments = contiguous.get(
                channel,
                []
            )

            segments = merge_segments(
                segments,
                max_gap=5
            )

            segments = [
                (start, end)
                for start, end in segments
                if (end - start + 1) >= 5
            ]

            result["channels"][channel].append(
                {
                    "trial": trial_id,
                    "segments": segments
                }
            )

    return result