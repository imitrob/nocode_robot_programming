
from typing import Iterable, List, Tuple, Dict
import math

def cluster(task_index, window_size: int = 10):
    non_zero_offsets = [item for item in task_index['offsets'] if item != 0]
    windows = cluster_branching_windows(non_zero_offsets, window_size=window_size)['windows']
    
    relevant_names = relevant_skillparts(task_index, windows)

    decision_states = []
    for w, rel in zip(windows, relevant_names):
        decision_states.append({'start': w[0], 'end': w[1], "relevant_parts": rel})

    return decision_states

def relevant_skillparts(task_index, windows):

    def search_for_parent(offset: int):
        name = None
        for name_, offset_ in zip(task_index['names'], task_index['offsets']):
            if 'trial' in name_:
                continue
            if offset == offset_:
                if name is not None: raise Exception("Multiple skill parts with similar offset")
                name = name_
        assert name is not None, f"Skill part with offset: {offset} is not found,\n{list(zip(task_index['names'], task_index['offsets']))}"
        return name

    skillparts_for_each_windows = []
    for window in windows:
        relevant_parts = []
        for i, skillpart_name in enumerate(task_index['names']):
            
            offset = task_index['offsets'][i]
            parent_offset = task_index['parent_offsets'][i]

            parent_name = search_for_parent(parent_offset)

            if "trial" not in skillpart_name:
                if (window[0] <= offset <= window[1]):
                    if parent_name not in relevant_parts:
                        relevant_parts.append(parent_name)
                    relevant_parts.append(skillpart_name)
        skillparts_for_each_windows.append(relevant_parts)

    return skillparts_for_each_windows

def cluster_branching_windows(
    starts: Iterable[int],
    window_size: int,
    max_multiple: int = 3,
    merge_touching: bool = True,
) -> Dict[str, object]:
    """ Cluster branch-start times into decision (branching) windows.

    Inputs
    ------
    starts : Iterable[int]
        Timesteps where alternative branches start (the original branch at t=0 is NOT included).
    window_size : int
        Default window size e. Each start t defines a base interval [t, t+e].
    max_multiple : int, default=3
        Enforce that any single decision window's length <= max_multiple * e.
    merge_touching : bool, default=True
        If True, treat intervals that just touch (next.start == current.end) as overlapping.

    Output
    ------
    dict with:
      - "windows": List[Tuple[int, int]] of merged decision windows (start, end)
      - "count": int number of windows
      - "method": str name of the algorithm used

    Notes
    -----
    The algorithm performs a classic union-of-intervals (“merge overlapping intervals”) via a
    sweep-line procedure (O(n log n) from sorting). After merging, any window longer than
    max_multiple * window_size is split into consecutive chunks of at most that length.

    Example
    -------
    >>> cluster_branching_windows([8, 12, 20, 24], window_size=3)
    {'windows': [(8, 15), (20, 27)], 'count': 2, 'method': 'Sweep-line union of intervals'}
    """
    e = int(window_size)
    if e <= 0:
        raise ValueError("window_size must be positive.")
    max_len = max_multiple * e

    # Build base intervals from starts
    intervals: List[Tuple[int, int]] = []
    for t in starts:
        t = int(t)
        intervals.append((t, t + e))

    if not intervals:
        return {"windows": [], "count": 0, "method": "Sweep-line union of intervals"}

    # Sort by start time
    intervals.sort(key=lambda x: x[0])

    # Merge (union of intervals) using a sweep through sorted intervals
    merged: List[Tuple[int, int]] = []
    cur_s, cur_e = intervals[0]
    for s, e2 in intervals[1:]:
        # Overlap / touching check
        if merge_touching:
            overlaps = s <= cur_e
        else:
            overlaps = s < cur_e
        if overlaps:
            cur_e = max(cur_e, e2)  # extend current window
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e2
    merged.append((cur_s, cur_e))

    # Enforce max window length (<= max_multiple * e) by splitting if needed
    bounded: List[Tuple[int, int]] = []
    for s, t in merged:
        L = t - s
        if L <= max_len:
            bounded.append((s, t))
        else:
            # split into consecutive chunks of size at most max_len
            k = math.ceil(L / max_len)
            for i in range(k):
                ss = s + i * max_len
                ee = min(s + (i + 1) * max_len, t)
                bounded.append((ss, ee))

    return {"windows": bounded, "count": len(bounded), "method": "Sweep-line union of intervals"}
