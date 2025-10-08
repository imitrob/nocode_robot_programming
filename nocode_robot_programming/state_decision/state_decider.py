
import torch

class StateDeciderBase():
    def __init__(self):
        self.model = None
        self.y_cls = None

    def train(self, X: torch.Tensor, y: torch.Tensor, y_cls): 
        '''
            X: shape (samples, w, h) 
            y: shape (samples, )
        '''
        target_label = y[0]
        self.model = target_label
        self.y_cls = y_cls

    def predict(self, image: torch.Tensor, timestep: float) -> tuple[bool, str]: # returns a branch (y) or -1 as anomaly
        return False, "test"
    
class StateDeciderMultiModel():
    def __init__(self, modelfactory):
        self.modelfactory = modelfactory
        self.models = []
        self.models_range = []

        self.window_size = 30

    def train(self, X: torch.Tensor, Xt: torch.Tensor, y: torch.Tensor):
        
        unique_branches = list(set(y))
        ret = cluster_branching_windows(unique_branches, self.window_size)
        chunks = ret['windows']
        nchunks = ret['count']
        
        print("==============================")
        print(f"Total {nchunks} models created")
        print(f"- model windows: {chunks}")
        print("==============================")

        for ichunk in range(nchunks):
            start, end = chunks[ichunk]

            Xch = torch.tensor([])
            tch = torch.tensor([])
            ych = torch.tensor([])
            for x,t,y_ in zip(X, Xt, y):
                if start <= t <= end:
                    Xch.contatenate([Xch, x])
                    tch.contatenate([tch, t])
                    ych.contatenate([ych, y_])

            model = self.modelfactory()
            
            model.train(Xch, ych)

            self.models.append(model)
            self.models_range.append((start, end))

    def predict(self, image: torch.Tensor, timestep: float) -> tuple[bool, int]: # returns a branch (y) or -1 as anomaly
        selected_model = None
        
        for n, model_range in enumerate(self.models_range):
            if model_range[0] <= timestep <= model_range[1]:
                print(f"Selected model {n} because: {model_range[0]} <= {timestep} <= {model_range[1]}")
                selected_model = n
        
        return self.models[selected_model].predict(image, timestep)


from typing import Iterable, List, Tuple, Dict
import math

def cluster_branching_windows(
    starts: Iterable[int],
    window_size: int,
    max_multiple: int = 3,
    merge_touching: bool = True,
) -> Dict[str, object]:
    """
    Cluster branch-start times into decision (branching) windows.

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
