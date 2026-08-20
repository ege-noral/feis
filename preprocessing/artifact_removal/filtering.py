from scipy.signal import butter, filtfilt


def bandpass_filter(
    eeg,
    sfreq,
    lowcut=1.0,
    highcut=40.0,
    order=4
):
    nyquist = sfreq / 2

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(
        order,
        [low, high],
        btype="band"
    )

    filtered = filtfilt(
        b,
        a,
        eeg,
        axis=-1
    )

    return filtered