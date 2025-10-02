from __future__ import annotations
import os, glob, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
import cv2 as cv

from nocode_robot_programming.state_decision.task_graph import TaskGraph
from pathlib import Path
import trajectory_data

import torchvision
def saved_img_processing(img):
    min_dim_size = min(img.shape[0], img.shape[1])
    resize_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.CenterCrop((min_dim_size, min_dim_size)),
            torchvision.transforms.Resize(
                (64, 64), torchvision.transforms.InterpolationMode.BILINEAR
            ),
        ]
    )

    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    return resize_transform(img_tensor) / 255.0


# ---- tiny dict-like wrappers ----
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

# ---- dataset ----
class TrajectoryDataset(TaskGraph, Dataset):
    """
    Each item is one rollout (skill) loaded from a .npz file.
    Expected keys: ['traj','ori','grip','img','img_feedback_flag',
                    'spiral_flag','risk_flag','safe_flag',
                    'novel_risk_flag','novel_safe_flag']
    """
    default_keys = ['traj','ori','grip','img',
                    'img_feedback_flag','spiral_flag','risk_flag',
                    'safe_flag','novel_risk_flag','novel_safe_flag']

    def __init__(self, package_path, keys=None):
        self.dir = os.path.join(package_path, "trajectories")
        self.files = sorted(glob.glob(os.path.join(self.dir, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {self.dir}")
        self.keys = keys or self.default_keys

    def get_all_names(self, name_skill: str):
        p = Path(f'{trajectory_data.package_path}/trajectories/')
        return [file.name[:-4] for file in p.iterdir() if file.is_file() and file.name.startswith(name_skill)]

    # video_train_names = get_all_names("user_0_kine_peg_pick")

    @property
    def names(self):
        ''' .../trajectories/new_skill.npz -> new_skill.npz '''
        return [f.split("/")[-1].split(".")[0] for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
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
            if k == 'img':            # (T,H,W) -> torch.float32 (T,1,H,W) in [0,1]
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
            f = self.timestep(idx=0, t=i).image
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
            try:
                branch_timestep = int(file.split("_")[-1])
            except ValueError:
                branch_timestep = 0
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
    
    def y_decode(self, y_int):
        return self.y_cls[y_int]
