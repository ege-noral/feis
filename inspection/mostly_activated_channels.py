from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


def inspect_task_segments(all_results):

    if not all_results:
        print("No results found.")
        return

    # ----------------------------------
    # Channel frequency
    # ----------------------------------

    channel_counter = Counter()

    for result in all_results:

        for channel in result["top_channels"]:

            channel_counter[channel] += 1

    print("\nMost selected channels:")

    for channel, count in channel_counter.most_common():
        print(channel, count)

    plt.figure(figsize=(10, 4))

    plt.bar(
        channel_counter.keys(),
        channel_counter.values()
    )

    plt.title(
        "Top Channel Frequency"
    )

    plt.xlabel("Channel")
    plt.ylabel("Count")

    plt.show()

    # ----------------------------------
    # Segment lengths
    # ----------------------------------

    lengths = []

    for result in all_results:

        start, end = result["segment"]

        lengths.append(
            end - start + 1
        )

    print("\nSegment length statistics")

    print(
        "Mean:",
        np.mean(lengths)
    )

    print(
        "Median:",
        np.median(lengths)
    )

    print(
        "Min:",
        np.min(lengths)
    )

    print(
        "Max:",
        np.max(lengths)
    )

    plt.figure(figsize=(8, 4))

    plt.hist(
        lengths,
        bins=30
    )

    plt.title(
        "Segment Length Distribution"
    )

    plt.xlabel(
        "Length (windows)"
    )

    plt.ylabel(
        "Count"
    )

    plt.show()

    # ----------------------------------
    # Segment starts
    # ----------------------------------

    starts = [
        result["segment"][0]
        for result in all_results
    ]

    plt.figure(figsize=(8, 4))

    plt.hist(
        starts,
        bins=30
    )

    plt.title(
        "Segment Start Distribution"
    )

    plt.xlabel(
        "Start Window"
    )

    plt.ylabel(
        "Count"
    )

    plt.show()

    # ----------------------------------
    # Segment centers
    # ----------------------------------

    centers = []

    for result in all_results:

        start, end = result["segment"]

        centers.append(
            (start + end) / 2
        )

    plt.figure(figsize=(8, 4))

    plt.hist(
        centers,
        bins=30
    )

    plt.title(
        "Segment Center Distribution"
    )

    plt.xlabel(
        "Center Window"
    )

    plt.ylabel(
        "Count"
    )

    plt.show()

    print(
        "\nTotal segments:",
        len(all_results)
    )