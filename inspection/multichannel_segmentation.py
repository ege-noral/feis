from preprocessing.sliding_window import create_trial_windows
from preprocessing.sliding_window import find_contiguous_selected_windows
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

if __name__ == "__main__":
    main()