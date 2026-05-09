from __future__ import annotations
import os, glob, numpy as np, torch, time
from typing import Dict, List, Set, Tuple
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, Counter
from pathlib import Path
import cv2 as cv

import trajectory_data
from nocode_robot_programming.task_graph.task_graph import TaskGraph
from nocode_robot_programming.state_decision.utils import Filename, saved_img_processing
from nocode_robot_programming.state_decision_dataset_prepare.trajectory_criteria import (
    filter_trajectory_files,
    sync_trajectory_criteria,
)
from nocode_robot_programming.jupyter_plot import show_gray_video_cuda, show_gray_video_cuda_captions, show_gray_video_cuda_captions_aligned
from IPython.display import display, HTML
import re

# Might be moved later to utils
class cc:
    H = '\033[95m'
    OK = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    W = '\033[93m'
    F = '\033[91m'
    E = '\033[0m'
    B = '\033[1m'
    U = '\033[4m'

_TRIAL_RE = re.compile(r"^(?P<base>.+)_trial_(?P<n>\d+)$")

# Branch bases (NOT trials):
_BRANCH_AT_RE   = re.compile(r"^(?P<orig>.+)_branch_at_(?P<at>\d+)$")
_BRANCH_FROM_RE = re.compile(r"^(?P<orig>.+)_branch_from_(?P<frm>\d+)_at_(?P<at>\d+)$")

class TimestepView(dict):
    """One rollout @ a single timestep with a convenience .image (H,W) uint8."""
    @property
    def image(self):
        img = self.get('img', None)  # (1,H,W) float32 in [0,1]
        if img is None: return None
        arr = img.squeeze(0).detach().cpu().numpy()
        return (arr * 255).astype(np.uint8)

class BatchTimestepView(dict):
    """Batch across multiple rollouts @ same timestep with quick image helpers."""
    def images_uint8(self):
        """Return numpy stack (B,H,W) uint8, or None if no images."""
        imgs = self.get('img', None)  # (B,1,H,W) float32 in [0,1]
        if imgs is None: return None
        arr = imgs.squeeze(1).detach().cpu().numpy()
        return (arr * 255).astype(np.uint8)
    def strip_uint8(self):
        """Horizontal strip (H, sum W) uint8 for quick side-by-side viewing."""
        arr = self.images_uint8()
        if arr is None: return None
        return np.concatenate([a for a in arr], axis=1)

class TrajectoryDataset(TaskGraph, Dataset):
    """
    Each item is one rollout (skill) loaded from a .npz file.
    Expected keys: ['traj','ori','grip','img','img_feedback_flag',
                    'spiral_flag','risk_flag','safe_flag',
                    'novel_risk_flag','novel_safe_flag','tag']
    """
    default_keys = ['traj','ori','grip','img',
                    'img_feedback_flag','spiral_flag','risk_flag',
                    'safe_flag','novel_risk_flag','novel_safe_flag','tag']

    def __init__(
        self,
        package_path: str | None = None,
        keys=None,
        print_index: bool = False,
        # Temporary CSV criteria gate for manually excluding bad/corrupted .npz files.
        criteria_csv: str | os.PathLike | None = None,
        sync_criteria_csv: bool = False,
        require_criteria_rows: bool = False,
        print_criteria_report: bool = False,
        use_criteria: frozenset[str] | set[str] | None = None,
    ):
        """ package_path (str) = custom trajectory package path.

        The criteria_csv* arguments are a temporary/manual dataset cleanup gate:
        sync a CSV, mark use=0 for corrupted trajectories, and load only use=1 rows.
        """
        if package_path is None:
            self.dir = os.path.join(trajectory_data.package_path, "trajectories")
        else:
            self.dir = package_path

        self.files = sorted(glob.glob(os.path.join(self.dir, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {self.dir}")

        # === TEMPORARY CSV TRAJECTORY CRITERIA GATE: BEGIN ===
        # Remove this block plus the criteria_csv* __init__ args/imports to return
        # TrajectoryDataset to plain "load every .npz in self.dir" behavior.
        self.criteria_csv = None
        self.criteria_report = None
        if criteria_csv is not None:
            criteria_path = Path(criteria_csv)
            if not criteria_path.is_absolute():
                criteria_path = Path(self.dir) / criteria_path
            self.criteria_csv = criteria_path
            if sync_criteria_csv:
                sync_trajectory_criteria(criteria_path, self.files)
            elif not criteria_path.exists():
                raise FileNotFoundError(
                    f"Trajectory criteria file {criteria_path} does not exist. "
                    "Pass sync_criteria_csv=True once to create it."
                )
            self.files, self.criteria_report = filter_trajectory_files(
                self.files,
                criteria_path,
                require_rows=require_criteria_rows,
                use_criteria=use_criteria,
            )
            if not self.files:
                raise ValueError(f"Trajectory criteria file {criteria_path} excluded every .npz file")
            if print_criteria_report:
                print(self.criteria_report)
        # === TEMPORARY CSV TRAJECTORY CRITERIA GATE: END ===

        self.keys = keys or self.default_keys

        self._task_index = self._build_task_index()
        if print_index:
            print("Found tasks:\n" + self.__str__())

    def __str__(self) -> str:
        self.warn_incomplete_trials_and_branches()
        return self.table_by_user_modality()

    def __repr__(self):
        return self.__str__()

    def get_all_names(self, name_skill: str):
        p = Path(f'{trajectory_data.package_path}/trajectories/')
        return [file.name[:-4] for file in p.iterdir() if file.is_file() and file.name.startswith(name_skill)]

    def _build_task_index(self) -> Dict[str, Dict[str, List]]:
        """
        Returns:
          {
            task: {
              'names':   [str, ...],
              'offsets': [int, ...],
              'trials':  [int, ...],
              'files':   [str, ...],
            },
            ...
          }
        """
        index = defaultdict(lambda: {"names": [], "offsets": [], "trials": [], "files": [], "tags": [], "lengths": [], "parent_offsets": []})
        for f in self.names: 
            f_ = Filename(f)
            entry = index[f_.task]
            entry["names"].append(f_.name)
            entry["offsets"].append(int(f_.offset))
            entry["parent_offsets"].append(f_.parent_offset)
            l, tag = self.get_length_and_tag(f_.name)
            entry["lengths"].append(l)
            entry["trials"].append(int(f_.trial))
            entry["files"].append(f)
            entry["tags"].append(tag)
        return dict(index)

    def get_length(self, idx: str):
        if isinstance(idx, str) and idx in self.names:
            idx = self.names.index(idx) # idx string -> idx id (int)
        
        if isinstance(idx, slice):
            return self.__getitems__(idx)
        
        data = np.load(self.files[idx], allow_pickle=False, mmap_mode='r')
        return len(data['grip'][0])

    def get_length_and_tag(self, idx: str):
        if isinstance(idx, str) and idx in self.names:
            idx = self.names.index(idx) # idx string -> idx id (int)
        
        if isinstance(idx, slice):
            return self.__getitems__(idx)
        
        data = np.load(self.files[idx], allow_pickle=False, mmap_mode='r')

        if 'tag' in data.keys():
            return len(data['grip'][0]), data['tag']
        else:
            return len(data['grip'][0]), ""


    # video_train_names = get_all_names("user_0_kine_peg_pick")

    @property
    def names(self):
        ''' .../trajectories/new_skill.npz -> new_skill.npz '''
        return [f.split("/")[-1].split(".")[0] for f in self.files]

    @property
    def tasks(self) -> Dict[str, Dict[str, List]]:
        """Structured view of available tasks (names, offsets, trials, files)."""
        return self._task_index
    
    def summary(self) -> Dict[str, dict]:
        """
        Returns per-task stats useful for logs/UI:
          { task: { 'files': N, 'unique_names': K, 'trial_range': (min,max), 'offset_range': (min,max) } }
        """
        info = {}
        for task, rec in self._task_index.items():
            trials = rec["trials"]
            offs = rec["offsets"]
            info[task] = {
                "files": len(rec["files"]),
                "unique_names": len(set(rec["names"])),
                "trial_range": (min(trials), max(trials)) if trials else None,
                "offset_range": (min(offs), max(offs)) if offs else None,
            }
        return info
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: str | slice | int):
        if isinstance(idx, str) and idx in self.names:
            idx = self.names.index(idx) # idx string -> idx id (int)
        
        if isinstance(idx, slice):
            return self.__getitems__(idx)
        # lazy, low-RAM reads
        data = np.load(self.files[idx], allow_pickle=False, mmap_mode='r')
        out = {}
        for k in self.keys:
            if k not in data: 
                continue
            v = data[k]
            if k == 'tag':
                v = str(v)
            elif k == 'img':            # (T,H,W) -> torch.float32 (T,1,H,W) in [0,1]
                v = torch.from_numpy(v).float()
                if v.ndim == 3: v = v[:, None, :, :]
                v = v / 255.0
            else:
                v = torch.from_numpy(np.asarray(v).T)
            out[k] = v
        out['name'] = os.path.splitext(os.path.basename(self.files[idx]))[0]
        out['length'] = out['traj'].shape[0] if 'traj' in out else None
        
        # arr = out['img'].squeeze(1).detach().cpu().numpy()
        # out['img_uint8'] = (arr * 255).astype(np.uint8)
        return out

    def __getitems__(self, fromto):
        ret = []
        for idx in range(fromto.start, fromto.stop):
            ret.append(self[idx])
        return ret

    # ---- quick access helpers ----
    def timestep(self, idx, t):
        """
        Return TimestepView with tensors for a single timestep t from rollout idx.
        Adds .image property -> uint8 (H,W).
        """
        data = self[idx]
        # infer sequence length T from the time-major tensors
        time_tensors = [v for k,v in data.items() if k not in ('name','length') and torch.is_tensor(v) and v.ndim>0]
        T = data['length'] or min(v.shape[0] for v in time_tensors)
        t = int(max(0, min(t, T-1)))

        one = {}
        for k, v in data.items():
            if k in ('name','length'): continue
            one[k] = v[t] if (torch.is_tensor(v) and v.dim()>0 and v.size(0)==T) else v
        one['name'] = data['name']; one['t'] = t
        return TimestepView(one)

    def batch_timestep(self, indices, t):
        """
        Stack a batch at timestep t across multiple rollouts.
        Returns BatchTimestepView with tensors (e.g., img -> (B,1,H,W)) and:
          - .images_uint8() -> (B,H,W) uint8
          - .strip_uint8()  -> (H, sum W) uint8
        """
        buckets = {}
        names = []
        for i in indices:
            ts = self.timestep(i, t)
            names.append(ts['name'])
            for k, v in ts.items():
                if k in ('name','t'): continue
                buckets.setdefault(k, []).append(v if v.ndim else v.unsqueeze(0))
        stacked = {k: torch.stack([x if x.ndim>0 else x.unsqueeze(0) for x in lst], dim=0)
                   for k, lst in buckets.items()}
        stacked['names'] = names
        stacked['t'] = int(t)
        return BatchTimestepView(stacked)

    def side_by_side_images(self, indices, t):
        """Torch tensor (1,H,W_total) float32 in [0,1], concatenated horizontally."""
        ims = [self.timestep(i, t).get('img') for i in indices]
        ims = [im for im in ims if im is not None]
        if not ims: return None
        return torch.cat(ims, dim=-1)
    
    def play_video(self, idx: int, fps: int = 30):
        delay = max(1, int(1000 / fps))  # milliseconds
        for i in range(len(self[idx]['img'])):
            f = self.timestep(idx=idx, t=i).image
            cv.imshow('video', f)
            if cv.waitKey(delay) & 0xFF == 27:  # ESC to quit
                break
        cv.destroyAllWindows()

    def get_image_dataset(self, file_names: list) -> ImageDatasetView:
        assert len(file_names) > 0
        X = torch.tensor([])
        Xt = torch.tensor([])
        y_int = torch.tensor([], dtype=torch.int)
        y_names = []

        for i, file in enumerate(file_names):
            f = Filename(file)
            branch_timestep = f.offset

            idx = self.files.index(f"{self.dir}/{file}.npz")
            nsamples = len(self[idx]['img'])
            
            # X.shape = (samples, width, height)
            X = torch.hstack([X, saved_img_processing(self[idx]['img'].squeeze())])
            
            # Xt.shape 
            xt_list = torch.tensor(list(range(branch_timestep,branch_timestep+nsamples)))
            Xt = torch.concatenate([Xt, xt_list])
            
            y_int = torch.concatenate([y_int, torch.tensor([i] * nsamples)])
            y_names.extend([file] * nsamples)
        return ImageDatasetView(X = X.squeeze(), Xt = Xt, y_int = y_int, y_names = y_names, y_cls = file_names)

    @property
    def all_tasks(self) -> List[str]:
        ''' All task names '''
        return list(self.tasks.keys())

    def _split_user_modality_task(self, task_key: str) -> Tuple[str, str, str]:
        """
        task_key format: <user>_<modality>_<task...>
        Returns (user, modality, task_root)
        """
        parts = task_key.split("_")
        if len(parts) < 3:
            return ("unknown", "unknown", task_key)
        user = parts[0]
        modality = parts[1]
        task_root = "_".join(parts[2:])
        return user, modality, task_root

    def group_by_user_modality(self) -> Dict[str, Dict[str, Dict[str, List[dict]]]]:
        """
        Build a nested index:
        { user: {
            modality: {
            task_root: [
                { 'name': <original_or_branch_name>, 'offset': <int>, 'trials': <int> },
                ...
            ]
            }
        }
        }
        """
        umt = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for task_key, rec in self._task_index.items():
            user, modality, task_root = self._split_user_modality_task(task_key)

            # Count trials per offset once
            counts_by_offset = Counter(
                off for off, tr in zip(rec["offsets"], rec["trials"]) if tr is not None and tr >= 0
            )

            # Rows = only originals/branches (trial == -1)
            for name, off, tr in zip(rec["names"], rec["offsets"], rec["trials"]):
                if tr == -1:
                    n_trials_int  = int(counts_by_offset.get(off, 0))
                    if n_trials_int > 3:
                        n_trials_str = f"{n_trials_int}" #f"{cc.OKGREEN}{n_trials_int}{cc.E}"
                    else:
                        n_trials_str = f"{n_trials_int}" # f"{cc.W}{n_trials_int}{cc.E}"

                    umt[user][modality][task_root].append(
                        {"name": name, "offset": int(off), "trials": n_trials_str}
                    )

        # sort deterministically (by offset within each task; root(0) first)
        for user in umt:
            for modality in umt[user]:
                for task_root in umt[user][modality]:
                    umt[user][modality][task_root].sort(key=lambda x: (x["offset"], x["name"]))
        return umt

    def _format_table(self, rows: List[Tuple[str, str, str]]) -> str:
        """
        rows: list of (Task, Name/Branch, #Trials) for one (user, modality) section.
        """
        if not rows:
            return "(no items)"
        headers = ("Task", "Name/Branch", "# Trials")
        # compute widths
        col_widths = [
            max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(3)
        ]
        def fmt(cols): return " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cols))
        sep = "-+-".join("-" * w for w in col_widths)
        out = [fmt(headers), sep]
        out += [fmt(r) for r in rows]
        return "\n".join(out)

    def table_by_user_modality(self) -> str:
        """
        Returns a printable string clustered by user -> modality.
        Each line is an original or a branch; trials are aggregated as a count.
        """
        umt = self.group_by_user_modality()
        if not umt:
            return "(no tasks)"

        total_trials = ""

        lines = []
        for user in sorted(umt.keys()):
            total_trials += f"{user:15s} | "

            lines.append(f"User: {user}")
            for modality in sorted(umt[user].keys()):
                lines.append(f"  Modality: {modality}")
                # build rows for this (user, modality)
                section_rows = []
                for task_root in sorted(umt[user][modality].keys()):
                    variants = umt[user][modality][task_root]
                    for v in variants:
                        total_trials += str(v["trials"]) + "+"
                        if int(v["trials"]) <= 3:
                            v["trials"] = f'{cc.W}{v["trials"]}{cc.E}'
                        section_rows.append(
                            (task_root, v["name"], str(v["trials"]))
                        )
                    if total_trials[-1] == '+': total_trials = total_trials[:-1]
                    total_trials += " "

                # indent the table
                table = self._format_table(section_rows)
                indented = "    " + table.replace("\n", "\n    ")
                lines.append(indented)
                total_trials += "| "
            total_trials += "\n"
            lines.append("")  # blank line between users
        
        print(total_trials)
        return "\n".join(lines).rstrip()

    def warn_incomplete_trials_and_branches(self, *, print_warnings: bool = True) -> List[str]:
        """
        Warns about:
        A) Trials:
            1) trial missing its parent base: <base>_trial_k exists but <base> missing
            2) gaps in trial numbering: trial_1 exists but trial_0 missing, etc.

        B) Branch bases:
            3) any branch base requires original: <orig>_branch_* exists but <orig> missing
            4) chain requirement for branch_from:
                <orig>_branch_from_<frm>_at_<at> requires a branch at <frm>
                specifically: <orig>_branch_from_<any>_at_<frm> must exist
                (for frm==0, the required parent is the original <orig>)

        Notes:
        - This checks only within each task_key entry of self._task_index.
        - It uses rec["names"] strings (whatever you store there).
        """
        warnings: List[str] = []

        for task_key, rec in self._task_index.items():
            names: Set[str] = set(rec["names"])

            # ---------- A) Trials ----------
            trials_by_base: Dict[str, Set[int]] = defaultdict(set)

            for nm in names:
                m = _TRIAL_RE.match(nm)
                if not m:
                    continue
                base = m.group("base")
                n = int(m.group("n"))
                trials_by_base[base].add(n)

            for base, nums in sorted(trials_by_base.items()):
                # (A1) missing trial parent base
                if base not in names:
                    warnings.append(
                        f"[missing trial parent] task='{task_key}': trial(s) exist for base '{base}' "
                        f"but parent '{base}' is missing. Present trials: {sorted(nums)}"
                    )

                # (A2) missing previous trials (gaps)
                if nums:
                    max_n = max(nums)
                    missing = [k for k in range(0, max_n + 1) if k not in nums]
                    if missing:
                        warnings.append(
                            f"[missing previous trial] task='{task_key}': base '{base}' has trial gap(s). "
                            f"Present: {sorted(nums)}; Missing: {missing}"
                        )

            # ---------- B) Branch bases ----------
            # Collect branch bases per original, keyed by the "at" offset.
            # We track:
            #   branch_at_offsets[orig]    = {at, ...} for names like orig_branch_at_at
            #   branch_from_at_offsets[orig] = {at, ...} for names like orig_branch_from_*_at_at
            branch_at_offsets: Dict[str, Set[int]] = defaultdict(set)
            branch_from_at_offsets: Dict[str, Set[int]] = defaultdict(set)

            # Also record all branch bases we saw so we can validate them
            branch_bases: List[Tuple[str, str]] = []  # (kind, name)

            for nm in names:
                # Skip trial names; branch parentage is about branch bases.
                if _TRIAL_RE.match(nm):
                    continue

                m_at = _BRANCH_AT_RE.match(nm)
                if m_at:
                    orig = m_at.group("orig")
                    at = int(m_at.group("at"))
                    branch_at_offsets[orig].add(at)
                    branch_bases.append(("branch_at", nm))
                    continue

                m_from = _BRANCH_FROM_RE.match(nm)
                if m_from:
                    orig = m_from.group("orig")
                    frm = int(m_from.group("frm"))
                    at = int(m_from.group("at"))
                    branch_from_at_offsets[orig].add(at)
                    branch_bases.append(("branch_from", nm))
                    continue

            # (B3) any branch base requires the original
            for kind, nm in branch_bases:
                if kind == "branch_at":
                    m = _BRANCH_AT_RE.match(nm)
                    assert m
                    orig = m.group("orig")
                    if orig not in names:
                        warnings.append(
                            f"[missing original for branch] task='{task_key}': branch '{nm}' exists but "
                            f"original '{orig}' is missing."
                        )

                elif kind == "branch_from":
                    m = _BRANCH_FROM_RE.match(nm)
                    assert m
                    orig = m.group("orig")
                    frm = int(m.group("frm"))

                    # Original must exist regardless (your requirement)
                    if orig not in names:
                        warnings.append(
                            f"[missing original for branch] task='{task_key}': branch '{nm}' exists but "
                            f"original '{orig}' is missing."
                        )

                    # (B4) chain: branch_from_<frm>_at_* requires a branch at <frm>
                    # - if frm==0: parent is the original (already checked above)
                    # - else: require existence of *some* branch_from_<any>_at_<frm>
                    if frm != 0:
                        has_parent_branch_at_frm = (
                            frm in branch_from_at_offsets.get(orig, set())
                        )
                        if not has_parent_branch_at_frm:
                            warnings.append(
                                f"[missing branch parent] task='{task_key}': branch '{nm}' requires a parent "
                                f"branch ending at {frm}, i.e. '{orig}_branch_from_<any>_at_{frm}', but none found."
                            )

        if print_warnings and warnings:
            print("Consistency warnings:")
            for w in warnings:
                print(" - " + w)
        elif print_warnings:
            print("No trial/branch consistency issues found.")

        return warnings

class ImageDatasetView(Dataset):
    def __init__(self, X, Xt, y_int, y_names, y_cls):
        super(ImageDatasetView, self).__init__()
        assert X.ndim == 3 and Xt.ndim == 1 and y_int.ndim == 1
        self.X = X.cuda() # X.shape = (samples, width, height)
        self.Xt = Xt.cuda() # Xt.shape = (samples, )
        self.y_int = y_int.cuda() # y_int.shape = (samples, )
        self.y_names = y_names # len(y_names) = samples
        self.y_cls = y_cls # len(y_cls) = "number of skill variants - files"
    
    def timestep_range(self) -> dict[str, int | float]:
        """ gets timestep range of used dataset view images """
        mu =    self.Xt.float().mean()
        std = self.Xt.float().std()
        min = self.Xt.float().min()
        max = self.Xt.float().max()
        return {"mean": float(mu), "std": float(std), "min": int(min), "max": int(max)}

    def y_decode(self, y_int):
        return self.y_cls[y_int]
    
    def image(self, i: int, scale=10.0):
        img = self.X[i]  # (1,H,W) float32 in [0,1]
        if img is None: return None
        arr = img.squeeze(0).detach().cpu().numpy()
        
        label = f"y={self.y_int[i]}"

        new_size = (int(self.w * scale), int(self.h * scale))

        cv.putText(arr, label, (0, 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, 2)

        arr = cv.resize(arr, new_size, interpolation=cv.INTER_NEAREST)

        
        return (arr * 255).astype(np.uint8)
    
    @property
    def n(self):
        return len(self.X)
    @property
    def w(self):
        return self.X[0].shape[0]
    @property
    def h(self):
        return self.X[0].shape[1]
    
    def show_image(self, i: int, scale: float = 10.0, timeout=3000):
        cv.imshow('video', self.image(i, scale))
        t0 = time.time()
        while True:
            k = cv.waitKey(20) & 0xFF
            if k == 27: # or cv.getWindowProperty('video', 0) < 1: # or (time.time()-t0)*1000 > timeout:
                break
        cv.destroyAllWindows()
        cv.waitKey(1)

    def play_video(self, fps: int=60, scale: float = 10.0):
        delay = max(1, int(1000 / fps))  # milliseconds
        for i in range(self.n):
            cv.imshow('video', self.image(i, scale))
            if cv.waitKey(delay) & 0xFF == 27:  # ESC to quit
                break
        cv.destroyAllWindows()
    
    def showcase(self, fps: int = 20, scale: int = 5):
        display(show_gray_video_cuda(self.X, fps=fps, scale=scale))

    def showcase_captions(self, fps: int = 20, scale: int = 5):
        captions = []
        for i,name in zip(self.y_int, self.y_names):
            captions.append(f"y={i},{name}")
        display(show_gray_video_cuda_captions(self.X, fps=fps, scale=scale, captions=captions, caption_fontsize=10))

    def showcase_aligned(self, fps: int = 20, scale: int = 5):
        captions = []
        for i,name in zip(self.y_int, self.y_names):
            captions.append(f"y={i},{name}")
        display(show_gray_video_cuda_captions_aligned(
            self.X,
            fps=fps,
            scale=scale,
            captions=captions,
            Xt=self.Xt, # shape [T], ints
            max_rows=10
        ))
