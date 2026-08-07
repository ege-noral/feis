from preprocessing.data_loader import load_subject_dataframe
from preprocessing.sliding_window import compare_window_variances
from preprocessing.sliding_window import compute_window_variance
from preprocessing.sliding_window import select_task_related_windows
from preprocessing.data_loader import split_into_epochs
from preprocessing.sliding_window import create_trial_windows
from preprocessing.sliding_window import find_contiguous_selected_windows
from preprocessing.sliding_window import find_longest_block_per_channel
from preprocessing.sliding_window import extract_task_segment
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():


    all_results = []

    for subject_id in range(1, 22): 
        for trial_id in range(0, 160):  

            try:
                result = extract_task_segment(
                    subject_id=subject_id,
                    trial_id=trial_id
                )

                if result is not None:
                    print(result)
                    all_results.append(result)

            except Exception as e:
                print(
                    f"Subject {subject_id}, "
                    f"Trial {trial_id} failed: {e}"
                )

    print(f"Total results: {len(all_results)}")

#     subject_id = 8
#     trial_id = 128

#     # Load data
#     resting_df = load_subject_dataframe(
#         subject_id=subject_id,
#         condition="resting"
#     )

#     thinking_df = load_subject_dataframe(
#         subject_id=subject_id,
#         condition="thinking"
#     )

#     # Split into trials
#     resting_trials = split_into_epochs(resting_df)
#     thinking_trials = split_into_epochs(thinking_df)

#     # Window all trials
#     resting_windows = create_trial_windows(
#         resting_trials,
#         window_size=13,
#         overlap=0.5
#     )

#     thinking_windows = create_trial_windows(
#         thinking_trials,
#         window_size=13,
#         overlap=0.5
#     )

#     # Compute variances
#     resting_variance = compute_window_variance(
#         resting_windows
#     )

#     thinking_variance = compute_window_variance(
#         thinking_windows
#     )

#     # Compare variances
#     results = compare_window_variances(
#         resting_variance,
#         thinking_variance,
#         ratio_threshold=12.0
#     )

#     # Keep only selected trial
#     trial_results = [
#         r for r in results
#         if r["trial"] == trial_id
#     ]

#     print(
#         f"Selected channel-windows in trial {trial_id}:",
#         len(trial_results)
#     )

#     channel_counts = Counter(
#     r["channel"]
#     for r in trial_results
# )

#     print("\nChannel counts:")
#     print(channel_counts)

#     top_channels = [
#         ch
#         for ch, count in channel_counts.most_common(5)
#     ]

#     print("\nTop channels:")
#     print(top_channels)

#     trial_results = [
#         r for r in trial_results
#         if r["channel"] in top_channels
#     ]

#     # Find contiguous blocks
#     contiguous_windows = find_contiguous_selected_windows(
#         trial_results
#     )

#     # Coverage of channels over windows
#     coverage_counter = Counter()

#     channels = contiguous_windows[trial_id]

#     for channel, ranges in channels.items():

#         for start, end in ranges:

#             for w in range(start, end + 1):

#                 coverage_counter[w] += 1

#     # Print strongest windows
#     top_windows = sorted(
#         coverage_counter.items(),
#         key=lambda x: x[1],
#         reverse=True
#     )

#     print("\nTop windows:")
#     print(top_windows[:20])

#     # Plot
#     windows = sorted(coverage_counter.keys())
#     counts = [coverage_counter[w] for w in windows]

#     plt.figure(figsize=(10, 4))
#     plt.plot(windows, counts)
#     plt.xlabel("Window Index")
#     plt.ylabel("Selected Channels")
#     plt.title(
#         f"Subject {subject_id} - Trial {trial_id}"
#     )
#     plt.grid(True)
#     plt.show()


if __name__ == "__main__":
    main()