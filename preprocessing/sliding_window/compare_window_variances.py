import numpy as np

EEG_CHANNELS = [
    "F3", "FC5", "AF3", "F7",
    "T7", "P7", "O1", "O2",
    "P8", "T8", "F8", "AF4",
    "FC6", "F4"
]


def compare_window_variances(
    resting_variance,
    thinking_variance,
    ratio_threshold=1.2
):

    ratio = (
        thinking_variance
        / (resting_variance + 1e-10)
    )

    selected = ratio > ratio_threshold

    window_ids, channel_ids = np.where(
        selected
    )

    results = []

    for w, c in zip(
        window_ids,
        channel_ids
    ):

        results.append({
            "window": int(w),
            "channel": EEG_CHANNELS[c],
            "resting_var":
                float(resting_variance[w, c]),
            "thinking_var":
                float(thinking_variance[w, c]),
            "ratio":
                float(ratio[w, c])
        })
        
    return results