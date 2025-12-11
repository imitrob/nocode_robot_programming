from __future__ import annotations
import os, glob, numpy as np, torch, time
from typing import Dict, List, Any
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from pathlib import Path
import cv2 as cv

import trajectory_data
from nocode_robot_programming.task_graph.task_graph import TaskGraph
from nocode_robot_programming.state_decision.utils import Filename, _ellipsize, _minmax, To01FromDtype, saved_img_processing, saved_img_processing_old
from nocode_robot_programming.jupyter_plot import show_gray_video_cuda, show_gray_video_cuda_captions, show_gray_video_cuda_captions_aligned
from IPython.display import display, HTML

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

    def __init__(self, package_path, keys=None, print_index: bool = False):
        self.dir = os.path.join(package_path, "trajectories")
        self.files = sorted(glob.glob(os.path.join(self.dir, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {self.dir}")
        self.keys = keys or self.default_keys

        self._task_index = self._build_task_index()
        if print_index:
            print("Found tasks:\n" + self.__str__())

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

    def __str__(self):
        return str(self.tasks)

    def get_all_names(self, name_skill: str):
        p = Path(f'{trajectory_data.package_path}/trajectories/')
        return [file.name[:-4] for file in p.iterdir() if file.is_file() and file.name.startswith(name_skill)]

    # video_train_names = get_all_names("user_0_kine_peg_pick")

    @property
    def names(self):
        ''' .../trajectories/new_skill.npz -> new_skill.npz '''
        return [f.split("/")[-1].split(".")[0] for f in self.files]


    @property
    def tasks(self) -> Dict[str, Dict[str, List]]:
        """Structured view of available tasks (names, offsets, trials, files)."""
        return self._task_index
    
    def __str__(self) -> str:
        """Pretty, compact table of the available tasks."""
        if not self._task_index:
            return "(no tasks)"
        # Compute column values
        rows = []
        for task, rec in sorted(self._task_index.items()):
            count = len(rec["files"])
            names = _ellipsize(rec["names"], max_chars=50)
            trials = _minmax(rec["trials"])
            offsets = _minmax(rec["offsets"])
            rows.append((task, str(count), names, trials, offsets))

        # Column headers
        headers = ("Task", "# Files", "Names (unique)", "Trials", "Offsets")
        # Determine widths
        col_widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

        def fmt_row(cols):
            return " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cols))

        sep = "-+-".join("-" * w for w in col_widths)
        out = [fmt_row(headers), sep]
        out += [fmt_row(r) for r in rows]
        return "\n".join(out)

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
        display(show_gray_video_cuda_captions(self.X, fps=fps, scale=scale, captions=captions))

    def showcase_aligned(self, fps: int = 20, scale: int = 5):
        captions = []
        for i,name in zip(self.y_int, self.y_names):
            captions.append(f"y={i},{name}")
        display(show_gray_video_cuda_captions_aligned(
            self.X,
            fps=fps,
            scale=scale,
            captions=captions,
            Xt=self.Xt,          # shape [T], ints
            max_rows=8           # optional, default 8
        ))
