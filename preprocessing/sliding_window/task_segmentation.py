from collections import Counter

from preprocessing.sliding_window import (
    create_trial_windows,
    find_contiguous_selected_windows,
    create_sliding_windows
)

from helper import ( 
    compute_window_variance,
    split_into_epochs,
    merge_segments
)

from inspection import compare_window_variances

from data_loader import get_epochs

def extract_task_segment(
    subject_id,
    trial_id,
    ratio_threshold=12.0,
    top_k_channels=5,
    coverage_threshold=1,
    window_size=13,
    overlap=0.5
):

    resting_trials, thinking_trials = get_epochs(
         subject_id=subject_id
    )

    thinking_trial = thinking_trials[trial_id]["eeg"]
    resting_trial = resting_trials[trial_id]["eeg"]

    resting_windows = create_sliding_windows(
        resting_trial,
        window_size=window_size,
        overlap=overlap
    )

    thinking_windows = create_sliding_windows(
        thinking_trial,
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


    channel_counts = Counter(
        r["channel"]
        for r in results
    )

    top_channels = [
        ch
        for ch, _
        in channel_counts.most_common(top_k_channels)
    ]

    results = [
        r for r in results
        if r["channel"] in top_channels
    ]

    contiguous_windows = (
        find_contiguous_selected_windows(
            results
        )
    )

    if not contiguous_windows:
        return None


    coverage_counter = Counter()

    for channel, ranges in contiguous_windows.items():

        for start, end in ranges:

            for w in range(start, end + 1):

                coverage_counter[w] += 1

    task_windows = [
        w
        for w, count in coverage_counter.items()
        if count >= coverage_threshold
    ]

    if not task_windows:
        return None

    task_windows.sort()

    segments = []

    start = task_windows[0]
    end = task_windows[0]

    for w in task_windows[1:]:

        if w == end + 1:
            end = w
        else:
            segments.append((start, end))
            start = w
            end = w

    segments.append((start, end))

    segments = merge_segments(
    segments,
    max_gap=5
    )

    longest_segment = max(
        segments,
        key=lambda x: x[1] - x[0]
    )

    return {
        "subject": subject_id,
        "trial": trial_id,
        "segment": longest_segment,
        "top_channels": top_channels,
        # "segments": segments
    }