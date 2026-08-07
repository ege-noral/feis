from collections import defaultdict


def find_contiguous_selected_windows(results):
    """
    Find contiguous selected windows for each channel.

    Parameters
    ----------
    results : list of dict

    Example
    -------
    {
        "window": 3,
        "channel": "AF3",
        "ratio": 20
    }

    Returns
    -------
    dict

    Example
    -------
    {
        "AF3": [(0, 3), (7, 8)],
        "T7": [(4, 10)]
    }
    """

    grouped_windows = defaultdict(list)

    for item in results:

        grouped_windows[
            item["channel"]
        ].append(
            item["window"]
        )

    contiguous = {}

    for channel, windows in grouped_windows.items():

        windows = sorted(windows)

        ranges = []

        start = windows[0]
        previous = windows[0]

        for current in windows[1:]:

            if current == previous + 1:
                previous = current

            else:
                ranges.append(
                    (start, previous)
                )

                start = current
                previous = current

        ranges.append(
            (start, previous)
        )

        contiguous[channel] = ranges

    return contiguous