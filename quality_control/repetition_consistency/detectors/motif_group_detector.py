from itertools import combinations

import numpy as np


def build_similarity_graph(
    similarity_matrix,
    window_centers_seconds,
    similarity_threshold,
    min_time_separation_seconds,
):
    """
    Self-similarity matrisinden threshold edilmiş graph oluşturur.

    Her pencere bir node'dur.

    İki pencere arasında edge olması için:
    1. Zaman olarak yeterince uzak olmaları gerekir.
    2. Similarity threshold'u geçmelidir.
    """

    similarity_matrix = np.asarray(
        similarity_matrix,
        dtype=float,
    )

    window_centers_seconds = np.asarray(
        window_centers_seconds,
        dtype=float,
    )

    if similarity_matrix.ndim != 2:
        raise ValueError(
            "similarity_matrix must be 2D."
        )

    if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError(
            "similarity_matrix must be square."
        )

    n_windows = similarity_matrix.shape[0]

    if len(window_centers_seconds) != n_windows:
        raise ValueError(
            "window_centers_seconds length must match similarity matrix."
        )

    adjacency = [
        set()
        for _ in range(n_windows)
    ]

    for i in range(n_windows):
        for j in range(i + 1, n_windows):

            time_difference = abs(
                window_centers_seconds[j]
                - window_centers_seconds[i]
            )

            if (
                time_difference
                < min_time_separation_seconds
            ):
                continue

            similarity = similarity_matrix[i, j]

            if similarity >= similarity_threshold:
                adjacency[i].add(j)
                adjacency[j].add(i)

    return adjacency


def _get_connected_components(adjacency):
    """
    Graph'i connected component'lara ayırır.

    Dense-group aramasını tüm pencereler üzerinde yapmak yerine
    yalnızca birbirleriyle en azından bir şekilde bağlantılı
    bölgelerde yaparak hesabı azaltır.
    """

    visited = set()
    components = []

    for start_node in range(len(adjacency)):

        if start_node in visited:
            continue

        stack = [start_node]
        component = set()

        while stack:

            node = stack.pop()

            if node in visited:
                continue

            visited.add(node)
            component.add(node)

            stack.extend(
                adjacency[node] - visited
            )

        components.append(
            sorted(component)
        )

    return components


def _is_temporally_valid_group(
    group_indices,
    window_centers_seconds,
    min_time_separation_seconds,
):
    """
    Bir grup içindeki HER iki pencerenin yeterince uzak
    olduğunu doğrular.

    Dense-group'ta bazı similarity bağlantılarının eksik
    olmasına izin vereceğimiz için bu kontrol ayrıca gereklidir.

    Böylece aynı gerçek event'in iki yakın sliding window'unu
    iki ayrı repetition olarak saymayız.
    """

    for i, j in combinations(
        group_indices,
        2,
    ):

        time_difference = abs(
            window_centers_seconds[j]
            - window_centers_seconds[i]
        )

        if (
            time_difference
            < min_time_separation_seconds
        ):
            return False

    return True


def _compute_group_density(
    group_indices,
    adjacency,
):
    """
    Gruptaki mevcut güçlü bağlantı sayısının
    mümkün olan toplam bağlantı sayısına oranı.

    Örnek:
        5 node -> 10 olası pair

        8 tanesi threshold'u geçiyorsa:

        density = 8 / 10 = 0.80
    """

    group_indices = list(
        group_indices
    )

    n_nodes = len(
        group_indices
    )

    possible_edges = (
        n_nodes
        * (n_nodes - 1)
        // 2
    )

    if possible_edges == 0:
        return 0.0, 0, 0

    edge_count = 0

    for i, j in combinations(
        group_indices,
        2,
    ):

        if j in adjacency[i]:
            edge_count += 1

    density = (
        edge_count
        / possible_edges
    )

    return (
        density,
        edge_count,
        possible_edges,
    )


def find_dense_groups(
    adjacency,
    window_centers_seconds,
    min_time_separation_seconds,
    min_group_size=3,
    max_group_size=5,
    min_group_density=1.0,
    min_neighbors_per_node=None,
):
    """
    Graph içinde motif gruplarını bulur.

    İki farklı kriter desteklenir:

    1. Group density
       Örneğin:
           min_group_density = 1.0
           -> tam clique

    2. Minimum neighbor support
       Örneğin:
           min_neighbors_per_node = 2
           -> gruptaki HER node en az 2 başka node ile
              güçlü similarity bağlantısına sahip olmalı.

    İki kriter de verilirse ikisinin de sağlanması gerekir.
    """

    if min_group_size < 2:
        raise ValueError(
            "min_group_size must be >= 2."
        )

    if max_group_size < min_group_size:
        raise ValueError(
            "max_group_size must be >= min_group_size."
        )

    if not 0.0 <= min_group_density <= 1.0:
        raise ValueError(
            "min_group_density must be between 0 and 1."
        )

    if (
        min_neighbors_per_node is not None
        and min_neighbors_per_node < 1
    ):
        raise ValueError(
            "min_neighbors_per_node must be >= 1 or None."
        )

    components = _get_connected_components(
        adjacency
    )

    dense_groups = []

    for component in components:

        if len(component) < min_group_size:
            continue

        largest_size = min(
            max_group_size,
            len(component),
        )

        for group_size in range(
            min_group_size,
            largest_size + 1,
        ):

            # Örneğin 3 üyeli bir grupta bir node'un
            # en fazla 2 komşusu olabilir.
            if (
                min_neighbors_per_node is not None
                and group_size - 1 < min_neighbors_per_node
            ):
                continue

            for group_indices in combinations(
                component,
                group_size,
            ):

                # Aynı gerçek olayın birbirine çok yakın
                # sliding-window temsillerini ayrı repetition sayma.
                if not _is_temporally_valid_group(
                    group_indices=group_indices,
                    window_centers_seconds=window_centers_seconds,
                    min_time_separation_seconds=(
                        min_time_separation_seconds
                    ),
                ):
                    continue

                group_indices = list(
                    group_indices
                )

                possible_edges = (
                    group_size
                    * (group_size - 1)
                    // 2
                )

                edge_count = 0

                node_degrees = {
                    node: 0
                    for node in group_indices
                }

                for i, j in combinations(
                    group_indices,
                    2,
                ):

                    if j in adjacency[i]:

                        edge_count += 1

                        node_degrees[i] += 1
                        node_degrees[j] += 1

                density = (
                    edge_count
                    / possible_edges
                )

                # --------------------------------------------
                # DENSITY CONTROL
                # --------------------------------------------

                if density < min_group_density:
                    continue

                # --------------------------------------------
                # MINIMUM NEIGHBOR SUPPORT CONTROL
                # --------------------------------------------

                min_degree = min(
                    node_degrees.values()
                )

                if (
                    min_neighbors_per_node is not None
                    and min_degree < min_neighbors_per_node
                ):
                    continue

                dense_groups.append(
                    {
                        "indices": group_indices,
                        "density": float(
                            density
                        ),
                        "edge_count": int(
                            edge_count
                        ),
                        "possible_edges": int(
                            possible_edges
                        ),
                        "min_degree": int(
                            min_degree
                        ),
                        "node_degrees": (
                            node_degrees
                        ),
                    }
                )

    return dense_groups
    """
    Graph içinde yeterince yoğun motif gruplarını bulur.

    Tam clique şartı yoktur.

    Örneğin 5 üyeli bir grup için 10 olası pair vardır.

        min_group_density = 1.00
            -> 10/10 gerekli

        min_group_density = 0.80
            -> en az 8/10 gerekli

        min_group_density = 0.70
            -> en az 7/10 gerekli
    """

    if min_group_size < 2:
        raise ValueError(
            "min_group_size must be >= 2."
        )

    if max_group_size < min_group_size:
        raise ValueError(
            "max_group_size must be >= min_group_size."
        )

    if not 0.0 <= min_group_density <= 1.0:
        raise ValueError(
            "min_group_density must be between 0 and 1."
        )

    components = _get_connected_components(
        adjacency
    )

    dense_groups = []

    for component in components:

        if len(component) < min_group_size:
            continue

        largest_size = min(
            max_group_size,
            len(component),
        )

        for group_size in range(
            min_group_size,
            largest_size + 1,
        ):

            for group_indices in combinations(
                component,
                group_size,
            ):

                # Çok yakın iki window aynı gerçek repetition
                # olabilir. Böyle grupları reddet.
                if not _is_temporally_valid_group(
                    group_indices=group_indices,
                    window_centers_seconds=window_centers_seconds,
                    min_time_separation_seconds=(
                        min_time_separation_seconds
                    ),
                ):
                    continue

                (
                    density,
                    edge_count,
                    possible_edges,
                ) = _compute_group_density(
                    group_indices=group_indices,
                    adjacency=adjacency,
                )

                if (
                    density
                    >= min_group_density
                ):
                    dense_groups.append(
                        {
                            "indices": list(
                                group_indices
                            ),
                            "density": float(
                                density
                            ),
                            "edge_count": int(
                                edge_count
                            ),
                            "possible_edges": int(
                                possible_edges
                            ),
                        }
                    )

    return dense_groups


def summarize_group(
    group,
    similarity_matrix,
    window_centers_seconds,
):
    """
    Dense motif grubunun ayrıntılarını hesaplar.
    """

    group_indices = group[
        "indices"
    ]

    pairwise_similarities = []

    strong_pairwise_similarities = []

    for i, j in combinations(
        group_indices,
        2,
    ):
        similarity = float(
            similarity_matrix[i, j]
        )

        pairwise_similarities.append(
            similarity
        )

    pairwise_similarities = np.asarray(
        pairwise_similarities,
        dtype=float,
    )

    times = np.asarray(
        [
            window_centers_seconds[index]
            for index in group_indices
        ],
        dtype=float,
    )

    result = {
        "indices": list(
            group_indices
        ),
        "times": times,
        "size": len(
            group_indices
        ),
        "density": float(
            group["density"]
        ),
        "edge_count": int(
            group["edge_count"]
        ),
        "possible_edges": int(
            group["possible_edges"]
        ),
        "mean_similarity": float(
            np.mean(
                pairwise_similarities
            )
        ),
        "min_similarity": float(
            np.min(
                pairwise_similarities
            )
        ),
        "max_similarity": float(
            np.max(
                pairwise_similarities
            )
        ),
    }

    return result


def _groups_temporally_overlap(
    group_a,
    group_b,
    tolerance_seconds,
):
    """
    Aynı gerçek repetition grubunun biraz kaydırılmış
    sliding-window versiyonlarını tespit eder.
    """

    times_a = np.asarray(
        group_a["times"],
        dtype=float,
    )

    times_b = np.asarray(
        group_b["times"],
        dtype=float,
    )

    used_b = set()
    matches = 0

    for time_a in times_a:

        distances = np.abs(
            times_b - time_a
        )

        candidate_indices = np.argsort(
            distances
        )

        for index_b in candidate_indices:

            if index_b in used_b:
                continue

            if (
                distances[index_b]
                <= tolerance_seconds
            ):
                matches += 1

                used_b.add(
                    index_b
                )

                break

    denominator = min(
        len(times_a),
        len(times_b),
    )

    if denominator == 0:
        return False

    match_ratio = (
        matches
        / denominator
    )

    return match_ratio >= 0.6


def remove_redundant_groups(
    groups,
    merge_tolerance_seconds,
):
    """
    Aynı gerçek motif grubunun kaymış window versiyonlarından
    oluşan çok sayıdaki dense group'u tek sonuca indirger.

    Öncelik:
        1. daha büyük grup
        2. daha yüksek density
        3. daha yüksek mean similarity
    """

    sorted_groups = sorted(
        groups,
        key=lambda group: (
            group["size"],
            group["density"],
            group["mean_similarity"],
        ),
        reverse=True,
    )

    selected_groups = []

    for candidate in sorted_groups:

        redundant = False

        for selected in selected_groups:

            if _groups_temporally_overlap(
                candidate,
                selected,
                tolerance_seconds=(
                    merge_tolerance_seconds
                ),
            ):
                redundant = True
                break

        if not redundant:
            selected_groups.append(
                candidate
            )

    return selected_groups


def detect_motif_groups(
    similarity_matrix,
    window_centers_seconds,
    similarity_threshold,
    min_time_separation_seconds,
    min_group_size=3,
    max_group_size=5,
    min_group_density=1.0,
    min_neighbors_per_node=None,
    merge_tolerance_seconds=0.20,
):
    """
    Self-similarity matrisinde karşılıklı olarak birbirini
    destekleyen motif gruplarını bulur.

    min_group_density:
        1.00 -> tam clique
        0.90 -> %90 bağlantı
        0.80 -> %80 bağlantı
        0.70 -> %70 bağlantı
    """

    similarity_matrix = np.asarray(
        similarity_matrix,
        dtype=float,
    )

    window_centers_seconds = np.asarray(
        window_centers_seconds,
        dtype=float,
    )

    adjacency = build_similarity_graph(
        similarity_matrix=similarity_matrix,
        window_centers_seconds=window_centers_seconds,
        similarity_threshold=similarity_threshold,
        min_time_separation_seconds=(
            min_time_separation_seconds
        ),
    )

    dense_groups = find_dense_groups(
        adjacency=adjacency,
        window_centers_seconds=window_centers_seconds,
        min_time_separation_seconds=(
            min_time_separation_seconds
        ),
        min_group_size=min_group_size,
        max_group_size=max_group_size,
        min_group_density=min_group_density,
        min_neighbors_per_node=min_neighbors_per_node,
    )

    groups = [
        summarize_group(
            group=group,
            similarity_matrix=similarity_matrix,
            window_centers_seconds=window_centers_seconds,
        )
        for group in dense_groups
    ]

    groups = remove_redundant_groups(
        groups=groups,
        merge_tolerance_seconds=(
            merge_tolerance_seconds
        ),
    )

    groups = sorted(
        groups,
        key=lambda group: (
            group["size"],
            group["density"],
            group["mean_similarity"],
        ),
        reverse=True,
    )

    if not groups:
        return {
            "groups": [],
            "best_group": None,
            "detected_repeats": 0,
            "peak_times": np.array(
                [],
                dtype=float,
            ),
        }

    best_group = groups[0]

    return {
        "groups": groups,
        "best_group": best_group,
        "detected_repeats": (
            best_group["size"]
        ),
        "peak_times": (
            best_group["times"]
        ),
    }