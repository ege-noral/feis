from preprocessing.sliding_window.labelwise_segmentation import (
    extract_label_channel_segments
)

from inspection import compute_channel_trial_coverage

from constants import (LABELS, SUBJECTS)


def main():

    for label in LABELS:

        print("\n" + "=" * 60)
        print(f"LABEL: {label}")
        print("=" * 60)

        for subject_id in SUBJECTS:

            try:

                result = extract_label_channel_segments(
                    subject_id=subject_id,
                    label=label
                )

                if result is None:
                    print(
                        f"\nSubject {subject_id}: "
                        f"No result"
                    )
                    continue

                coverage = compute_channel_trial_coverage(
                    result
                )

                print(
                    f"\nSubject {subject_id}"
                )

                for channel, stats in coverage.items():

                    print(
                        f"  {channel:4s} | "
                        f"{stats['trials_with_segments']}/"
                        f"{stats['total_trials']} "
                        f"({stats['coverage']:.1%})"
                    )

            except Exception as e:

                print(
                    f"\nSubject {subject_id} failed: {e}"
                )


if __name__ == "__main__":
    main()