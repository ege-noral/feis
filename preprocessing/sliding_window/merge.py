def merge_segments(segments, max_gap=5):
    """
    Yakın segmentleri birleştirir.

    Parameters
    ----------
    segments : list[tuple[int, int]]
        [(start, end), ...]
    max_gap : int
        İki segment arasındaki izin verilen maksimum boşluk
        (pencere cinsinden).

    Returns
    -------
    list[tuple[int, int]]
    """

    if not segments:
        return []

    merged = [segments[0]]

    for current_start, current_end in segments[1:]:
        last_start, last_end = merged[-1]

        gap = current_start - last_end - 1

        if gap <= max_gap:
            merged[-1] = (
                last_start,
                max(last_end, current_end)
            )
        else:
            merged.append(
                (current_start, current_end)
            )

    return merged