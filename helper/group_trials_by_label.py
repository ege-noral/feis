from collections import defaultdict

from helper import split_into_epochs


def group_trials_by_label(epochs):

    grouped = defaultdict(list)

    for epoch_id, epoch in epochs.items():

        label = epoch["label"]

        grouped[label].append({
            "trial": int(epoch_id),
            "eeg": epoch
        })

    return dict(grouped)