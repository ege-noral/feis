from collections import Counter


def compute_channel_trial_coverage(result):
    """
    For each channel, calculate how consistently
    significant blocks occur across trials.

    Assumes that minimum block length filtering
    has already been performed during segmentation.
    """

    coverage = {}

    for channel, trials in result["channels"].items():

        trial_count = len(trials)

        trials_with_segments = sum(
            1
            for trial in trials
            if trial["segments"]
        )

        coverage_ratio = (
            trials_with_segments / trial_count
            if trial_count > 0
            else 0
        )

        block_lengths = []

        for trial in trials:

            for start, end in trial["segments"]:

                length = end - start + 1
                block_lengths.append(length)

        coverage[channel] = {
            "trials_with_segments": trials_with_segments,
            "total_trials": trial_count,
            "coverage": round(coverage_ratio, 3),
            "block_lengths": block_lengths
        }

    return coverage


def evaluate_label_across_subjects(
    all_results,
    label
):
    """
    Evaluate channel trial coverage for a given label
    across multiple subjects.
    """

    channel_results = {}

    for result in all_results:

        if result["label"] != label:
            continue

        subject_id = result["subject"]

        coverage = compute_channel_trial_coverage(
            result
        )

        for channel, stats in coverage.items():

            if channel not in channel_results:
                channel_results[channel] = []

            channel_results[channel].append({
                "subject": subject_id,
                "coverage": stats["coverage"],
                "trials_with_segments": stats[
                    "trials_with_segments"
                ],
                "total_trials": stats[
                    "total_trials"
                ],
                "block_lengths": stats[
                    "block_lengths"
                ]
            })

    return channel_results