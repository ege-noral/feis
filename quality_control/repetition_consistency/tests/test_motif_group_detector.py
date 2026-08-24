import numpy as np

from quality_control.repetition_consistency.detectors.motif_group_detector import (
    detect_motif_groups,
)


def test_detects_three_way_motif_group():

    times = np.array([
        0.5,
        1.5,
        2.5,
        3.5,
    ])

    similarity = np.eye(4)

    # İlk üç pencere birbirine güçlü şekilde benziyor.
    similarity[0, 1] = similarity[1, 0] = 0.80
    similarity[0, 2] = similarity[2, 0] = 0.82
    similarity[1, 2] = similarity[2, 1] = 0.78

    # Dördüncü pencere benzemiyor.
    similarity[0, 3] = similarity[3, 0] = 0.20
    similarity[1, 3] = similarity[3, 1] = 0.25
    similarity[2, 3] = similarity[3, 2] = 0.30

    result = detect_motif_groups(
        similarity_matrix=similarity,
        window_centers_seconds=times,
        similarity_threshold=0.70,
        min_time_separation_seconds=0.50,
        min_group_size=3,
    )

    assert result["detected_repeats"] == 3

    np.testing.assert_allclose(
        result["peak_times"],
        [0.5, 1.5, 2.5],
    )


def test_single_pair_is_not_a_motif_group():

    times = np.array([
        0.5,
        1.5,
        2.5,
        3.5,
    ])

    similarity = np.eye(4)

    # Sadece tek güçlü pair var.
    similarity[0, 2] = similarity[2, 0] = 0.90

    result = detect_motif_groups(
        similarity_matrix=similarity,
        window_centers_seconds=times,
        similarity_threshold=0.70,
        min_time_separation_seconds=0.50,
        min_group_size=3,
    )

    assert result["detected_repeats"] == 0
    assert result["best_group"] is None


def test_detects_five_way_motif_group():

    times = np.array([
        0.4,
        1.2,
        2.0,
        3.0,
        4.1,
    ])

    similarity = np.full((5, 5), 0.80)

    np.fill_diagonal(
        similarity,
        1.0,
    )

    result = detect_motif_groups(
        similarity_matrix=similarity,
        window_centers_seconds=times,
        similarity_threshold=0.70,
        min_time_separation_seconds=0.50,
        min_group_size=3,
    )

    assert result["detected_repeats"] == 5

    np.testing.assert_allclose(
        result["peak_times"],
        times,
    )

def test_dense_group_allows_one_missing_connection():

    times = np.array([
        0.5,
        1.5,
        2.5,
        3.5,
        4.5,
    ])

    similarity = np.full(
        (5, 5),
        0.80,
    )

    np.fill_diagonal(
        similarity,
        1.0,
    )

    # Bir bağlantıyı bilerek bozuyoruz.
    similarity[0, 4] = 0.30
    similarity[4, 0] = 0.30

    # 5 node = 10 pair.
    # 9 tanesi güçlü, 1 tanesi zayıf.
    #
    # density = 9/10 = 0.90

    result = detect_motif_groups(
        similarity_matrix=similarity,
        window_centers_seconds=times,
        similarity_threshold=0.70,
        min_time_separation_seconds=0.50,
        min_group_size=3,
        max_group_size=5,
        min_group_density=0.80,
    )

    assert result["detected_repeats"] == 5

    assert (
        result["best_group"]["density"]
        == 0.90
    )

def test_minimum_neighbor_support_detects_consensus_group():

    times = np.array([
        0.5,
        1.5,
        2.5,
        3.5,
        4.5,
    ])

    similarity = np.eye(5)

    # A herkese benziyor.
    similarity[0, 1] = similarity[1, 0] = 0.80
    similarity[0, 2] = similarity[2, 0] = 0.80
    similarity[0, 3] = similarity[3, 0] = 0.80
    similarity[0, 4] = similarity[4, 0] = 0.80

    # Diğer üyeler de birbirlerinden destek alıyor.
    similarity[1, 2] = similarity[2, 1] = 0.80
    similarity[2, 3] = similarity[3, 2] = 0.80
    similarity[3, 4] = similarity[4, 3] = 0.80

    result = detect_motif_groups(
        similarity_matrix=similarity,
        window_centers_seconds=times,
        similarity_threshold=0.70,
        min_time_separation_seconds=0.50,

        min_group_size=3,
        max_group_size=5,

        # Clique istemiyoruz.
        min_group_density=0.0,

        # Ama HER node'un en az 2 desteği olmalı.
        min_neighbors_per_node=2,
    )

    assert result["detected_repeats"] == 5


def test_star_structure_is_rejected_by_neighbor_support():

    times = np.array([
        0.5,
        1.5,
        2.5,
        3.5,
        4.5,
    ])

    similarity = np.eye(5)

    # A herkese benziyor.
    similarity[0, 1] = similarity[1, 0] = 0.80
    similarity[0, 2] = similarity[2, 0] = 0.80
    similarity[0, 3] = similarity[3, 0] = 0.80
    similarity[0, 4] = similarity[4, 0] = 0.80

    # AMA B, C, D, E birbirlerine benzemiyor.

    result = detect_motif_groups(
        similarity_matrix=similarity,
        window_centers_seconds=times,
        similarity_threshold=0.70,
        min_time_separation_seconds=0.50,

        min_group_size=3,
        max_group_size=5,

        min_group_density=0.0,
        min_neighbors_per_node=2,
    )

    assert result["detected_repeats"] == 0