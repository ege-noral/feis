from collections import defaultdict


def find_contiguous_selected_windows(results):
    """
    Find contiguous selected windows for each trial and channel.

    Parameters
    ----------
    results : list of dict
        Output from variance ratio selection.
        Example:
        {
            'trial': 0,
            'window': 3,
            'channel': 'AF3',
            'ratio': 20
        }

    Returns
    -------
    dict
        Nested dictionary containing contiguous window ranges.

        Example:
        {
            0: {
                'AF3': [(0,3), (7,8)]
            }
        }
    """

    # trial-channel bazında window listeleri
    grouped_windows = defaultdict(list)

    for item in results:
        key = (
            item["trial"],
            item["channel"]
        )

        grouped_windows[key].append(
            item["window"]
        )


    contiguous = defaultdict(dict)


    for (trial, channel), windows in grouped_windows.items():

        windows = sorted(windows)

        ranges = []

        start = windows[0]
        previous = windows[0]


        for current in windows[1:]:

            # ardışık devam ediyor
            if current == previous + 1:
                previous = current

            else:
                ranges.append(
                    (start, previous)
                )

                start = current
                previous = current


        # son grubu ekle
        ranges.append(
            (start, previous)
        )


        contiguous[trial][channel] = ranges


    return dict(contiguous)