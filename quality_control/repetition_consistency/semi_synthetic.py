import numpy as np
from scipy.signal import resample


def _create_base_motif(
    fs,
    duration_seconds,
    rng,
):
    """
    Create one time-localized oscillatory motif.

    The motif is not intended to model imagined speech
    physiology exactly.

    Its purpose is to create a known repeated
    time-frequency pattern for validating the
    repetition-detection pipeline.
    """

    n_samples = int(
        round(duration_seconds * fs)
    )

    time = np.arange(
        n_samples
    ) / fs

    # Different frequency components.
    #
    # We deliberately do not make the motif a single
    # pure sinusoid because the real TFR structure will
    # also contain multiple frequencies.
    frequencies = np.array([
        8.0,
        13.0,
        22.0,
    ])

    amplitudes = np.array([
        1.0,
        0.7,
        0.4,
    ])

    phases = rng.uniform(
        0,
        2 * np.pi,
        size=len(frequencies),
    )

    motif = np.zeros(
        n_samples,
        dtype=float,
    )

    for frequency, amplitude, phase in zip(
        frequencies,
        amplitudes,
        phases,
    ):

        motif += (
            amplitude
            * np.sin(
                2
                * np.pi
                * frequency
                * time
                + phase
            )
        )

    # Smooth onset and offset.
    envelope = np.hanning(
        n_samples
    )

    motif *= envelope

    motif -= np.mean(motif)

    std = np.std(motif)

    if std > 0:
        motif /= std

    return motif


def _generate_event_layout(
    n_repeats,
    durations,
    epoch_seconds,
    min_gap_seconds,
    edge_margin_seconds,
    rng,
):
    """
    Generate irregular repetition times while making sure
    motifs fit inside the epoch without overlapping.

    Importantly, repetitions are NOT forced to occur
    every 1 second.
    """

    if n_repeats == 0:
        return np.array([])

    total_motif_duration = np.sum(
        durations
    )

    mandatory_gap_duration = (
        min_gap_seconds
        * max(
            0,
            n_repeats - 1,
        )
    )

    available_duration = (
        epoch_seconds
        - 2 * edge_margin_seconds
    )

    required_duration = (
        total_motif_duration
        + mandatory_gap_duration
    )

    slack = (
        available_duration
        - required_duration
    )

    if slack < 0:
        raise ValueError(
            "The requested motifs do not fit "
            "inside the epoch."
        )

    # Randomly distribute the remaining free time
    # before, between, and after the events.
    #
    # This is what prevents us from forcing the
    # repetitions to be exactly 1 second apart.
    random_gaps = rng.dirichlet(
        np.ones(
            n_repeats + 1
        )
    ) * slack

    starts = []

    current_time = (
        edge_margin_seconds
        + random_gaps[0]
    )

    for repeat_idx in range(
        n_repeats
    ):

        starts.append(
            current_time
        )

        current_time += (
            durations[repeat_idx]
        )

        if repeat_idx < (
            n_repeats - 1
        ):
            current_time += (
                min_gap_seconds
                + random_gaps[
                    repeat_idx + 1
                ]
            )

    return np.asarray(
        starts,
        dtype=float,
    )


def create_semi_synthetic_epoch(
    background_signal,
    fs=256,
    n_repeats=5,
    duration_range=(0.40, 0.80),
    motif_std_ratio=0.50,
    amplitude_jitter=(0.80, 1.20),
    min_gap_seconds=0.10,
    edge_margin_seconds=0.10,
    seed=None,
):
    """
    Inject repeated but slightly variable motifs into a
    real EEG background signal.

    Parameters
    ----------
    background_signal : np.ndarray
        Real EEG background, shape (n_samples,).

    fs : int or float
        Sampling frequency.

    n_repeats : int
        Number of motifs to inject.

    duration_range : tuple
        Minimum and maximum motif duration in seconds.

    motif_std_ratio : float
        Motif amplitude relative to the standard deviation
        of the real EEG background.

        Example:
        0.5 means motif std is approximately half the
        background EEG std.

    amplitude_jitter : tuple
        Random amplitude multiplier applied independently
        to each repetition.

    min_gap_seconds : float
        Minimum gap between motifs.

    edge_margin_seconds : float
        Minimum distance from epoch boundaries.

    seed : int or None
        Random seed.

    Returns
    -------
    synthetic_signal : np.ndarray
        EEG background + injected repetitions.

    event_info : list of dict
        Ground-truth repetition information.
    """

    background_signal = np.asarray(
        background_signal,
        dtype=float,
    )

    if background_signal.ndim != 1:
        raise ValueError(
            "background_signal must be 1D."
        )

    if n_repeats < 0:
        raise ValueError(
            "n_repeats cannot be negative."
        )

    rng = np.random.default_rng(
        seed
    )

    synthetic_signal = (
        background_signal.copy()
    )

    n_samples = len(
        synthetic_signal
    )

    epoch_seconds = (
        n_samples / fs
    )

    if n_repeats == 0:
        return (
            synthetic_signal,
            [],
        )

    # --------------------------------------------------
    # Every repetition may have a slightly different
    # duration.
    # --------------------------------------------------

    durations = rng.uniform(
        duration_range[0],
        duration_range[1],
        size=n_repeats,
    )

    starts = _generate_event_layout(
        n_repeats=n_repeats,
        durations=durations,
        epoch_seconds=epoch_seconds,
        min_gap_seconds=min_gap_seconds,
        edge_margin_seconds=edge_margin_seconds,
        rng=rng,
    )

    # --------------------------------------------------
    # Create ONE underlying motif family.
    #
    # Repetitions will be stretched/compressed versions
    # of this same motif rather than exact copies.
    # --------------------------------------------------

    nominal_duration = np.mean(
        duration_range
    )

    base_motif = _create_base_motif(
        fs=fs,
        duration_seconds=nominal_duration,
        rng=rng,
    )

    background_std = np.std(
        background_signal
    )

    if background_std == 0:
        background_std = 1.0

    event_info = []

    for repeat_idx in range(
        n_repeats
    ):

        duration = durations[
            repeat_idx
        ]

        motif_samples = int(
            round(
                duration * fs
            )
        )

        # Slight temporal stretching/compression.
        motif = resample(
            base_motif,
            motif_samples,
        )

        # Slight amplitude variation between repetitions.
        amplitude_factor = rng.uniform(
            amplitude_jitter[0],
            amplitude_jitter[1],
        )

        motif = (
            motif
            * background_std
            * motif_std_ratio
            * amplitude_factor
        )

        start_sample = int(
            round(
                starts[repeat_idx]
                * fs
            )
        )

        end_sample = (
            start_sample
            + motif_samples
        )

        synthetic_signal[
            start_sample:end_sample
        ] += motif

        onset_seconds = (
            start_sample / fs
        )

        offset_seconds = (
            end_sample / fs
        )

        center_seconds = (
            (
                start_sample
                + end_sample
            )
            / 2
            / fs
        )

        event_info.append({
            "repeat_index": repeat_idx,
            "onset_seconds": onset_seconds,
            "offset_seconds": offset_seconds,
            "center_seconds": center_seconds,
            "duration_seconds": (
                motif_samples / fs
            ),
            "amplitude_factor": (
                amplitude_factor
            ),
        })

    return (
        synthetic_signal,
        event_info,
    )