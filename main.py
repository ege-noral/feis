from preprocessing.data_loader import load_subject
from preprocessing.sliding_window import create_sliding_windows


def main():
    subject_id = 1
    condition = "thinking"

    eeg_data = load_subject(
        subject_id=subject_id,
        condition=condition
    )

    print("EEG shape:", eeg_data.shape)

    windows = create_sliding_windows(
        eeg_data,
        window_size=50,
        overlap=0.5
    )

    sfreq = 256
    window_ms = 50

    window_size = int(sfreq * window_ms / 1000)

    windows = create_sliding_windows(
        eeg_data,
        window_size=window_size,
        overlap=0.5
    )

    print(windows.shape)


if __name__ == "__main__":
    main()