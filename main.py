from preprocessing.data_loader import load_subject_dataframe
from preprocessing.sliding_window import compare_window_variances
from preprocessing.sliding_window import compute_window_variance
from preprocessing.sliding_window import select_task_related_windows
from preprocessing.data_loader import split_into_epochs
from preprocessing.sliding_window import create_trial_windows
from preprocessing.sliding_window import find_contiguous_selected_windows
from preprocessing.sliding_window import find_longest_block_per_channel
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():

    for subject_id in range(2, 3):

        resting_df = load_subject_dataframe(
        subject_id=subject_id,
        condition="resting"
        )

        thinking_df = load_subject_dataframe(
            subject_id=subject_id,
            condition="thinking"
        )

        resting_trials = split_into_epochs(
            resting_df
        )

        thinking_trials = split_into_epochs(
            thinking_df
        )


        resting_windows = create_trial_windows(
            resting_trials,
            window_size=13,
            overlap=0.5
        )

        thinking_windows = create_trial_windows(
            thinking_trials,
            window_size=13,
            overlap=0.5
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
            ratio_threshold=12.0
        )

        print("Channel selections:", len(results))

        unique_windows = {
            (r["trial"], r["window"])
            for r in results
        }

        print("Unique windows:", len(unique_windows))

        channel_counts = Counter(
        r["channel"]
        for r in results
        )

        contiguous_windows = find_contiguous_selected_windows(results)

        longest = find_longest_block_per_channel(contiguous_windows)

        for channel, info in sorted(
            longest.items(),
            key=lambda x: x[1]["length"],
            reverse=True
        )[:5]:

            print(
                channel,
                "length =", info["length"],
                "trial =", info["trial"]
            )

        coverage_counter = Counter()

        for trial, channels in contiguous_windows.items():

            for channel, ranges in channels.items():

                for start, end in ranges:

                    for w in range(start, end + 1):

                        coverage_counter[w] += 1
        

        top_windows = sorted(
            coverage_counter.items(),
            key=lambda x: x[1],
            reverse=True
        )

        print(top_windows[:20])

        windows = sorted(coverage_counter.keys())
        counts = [coverage_counter[w] for w in windows]

        plt.figure(figsize=(10, 4))
        plt.bar(windows, counts)
        plt.xlabel("Window Index")
        plt.ylabel("Coverage Count")
        plt.title(f"Subject {subject_id} - Selected Window Coverage")
        plt.show()
    

if __name__ == "__main__":
    main()