from typing import List
import os, signal
import psutil  # pip install psutil
import numpy as np
import trajectory_data
import torch, torchvision

def list_other_ipykernels():
    me = os.getpid()

    # collect our ancestor PIDs to avoid self-kill
    ancestors = set()
    try:
        p = psutil.Process(me)
        while True:
            ancestors.add(p.pid)
            if p.ppid() == 0 or p.ppid() == p.pid:
                break
            p = psutil.Process(p.ppid())
    except psutil.Error:
        pass

    victims = []
    for proc in psutil.process_iter(['pid','cmdline']):
        try:
            cmd = proc.info['cmdline'] or []
            if any('ipykernel_launcher' in part for part in cmd):
                if proc.pid not in ancestors:
                    victims.append((proc.pid, ' '.join(cmd)))
        except psutil.Error:
            continue
    return victims

def kill_other_ipykernels(force=False):
    victims = list_other_ipykernels()
    for pid, cmd in victims:
        try:
            os.kill(pid, signal.SIGKILL if force else signal.SIGTERM)
            print(f"Killed {pid}: {cmd}")
        except ProcessLookupError:
            pass
    if not victims:
        print("No other ipykernel_launcher processes found.")

# Preview what would be killed:
for pid, cmd in list_other_ipykernels():
    print(pid, cmd)

## TODO: FIX ERROR rclpy._rclpy_pybind11.RCLError with the following code
# Run this in one cell to start ROS in a notebook
'''
import threading, atexit
import rclpy
from rclpy.context import Context
from rclpy.signals import SignalHandlerOptions
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

_ctx = Context()
rclpy.init(context=_ctx, signal_handler_options=SignalHandlerOptions.NO)  # <-- no SIGINT/SIGTERM hooks in notebooks

node = Node("nb_node", context=_ctx)
executor = SingleThreadedExecutor(context=_ctx)
executor.add_node(node)

_spin_thread = threading.Thread(target=executor.spin, daemon=True)
_spin_thread.start()

def _ros_cleanup():
    # Safe to call multiple times
    try:
        executor.remove_node(node)
    except Exception:
        pass
    try:
        node.destroy_node()
    except Exception:
        pass
    try:
        executor.shutdown()
    except Exception:
        pass
    try:
        rclpy.shutdown(context=_ctx, uninstall_handlers=False)
    except Exception:
        pass
    try:
        _ctx.destroy()
    except Exception:
        pass

atexit.register(_ros_cleanup)  # fires on kernel shutdown, too
'''

def add_tag(filename: str, tag: str):
    # Load existing .npz
    data = np.load(f"{trajectory_data.package_path}/trajectories/{filename}.npz")

    # Copy everything into a new dict
    arrays = {key: data[key] for key in data.files}

    # Add your tag
    arrays["tag"] = tag

    # Overwrite (or save to a new file if you want safety)
    np.savez(f"{trajectory_data.package_path}/trajectories/{filename}.npz", **arrays)
    data.close()


class Filename():
    """ With filename: 'peg_pick_branch_at_123_trial_4', it extracts:
            - task name: 'peg_pick'
            - offset timestep: 123
            - trial number: 4
    """
    branch_suffix = "branch_at"
    trial_suffix = "trial"
    def __init__(self, filename: str):
        """ filename can be either with or without ".npz" extension
        """
        if filename[-4:] == ".npz":
            self.filename = filename
            self.name: str = self.filename[:-4] # without npz
        else:
            self.filename = filename + ".npz"
            self.name = filename

        trial_split = self.name.split(f"_{self.trial_suffix}_")
        before_trial_suffix = trial_split[0]
        if len(trial_split) > 1:
            self.trial: int = int(self.name.split(f"_{self.trial_suffix}_")[1])
        else:
            self.trial: int = -1 # initial (nominal) demonstration

        branch_split = before_trial_suffix.split(f"_{self.branch_suffix}_")
        if len(branch_split) > 1:
            self.offset: int = int(before_trial_suffix.split(f"_{self.branch_suffix}_")[1])
        else:
            self.offset: int = 0

        self.task: str = branch_split[0]


def _ellipsize(items: List[str], max_chars: int = 60, sep: str = ", ") -> str:
    """Join unique items and ellipsize to keep rows compact."""
    uniq = list(dict.fromkeys(items))  # stable unique
    s = sep.join(uniq)
    if len(s) <= max_chars:
        return s
    # keep adding until limit, then ellipsis + count
    out, total = [], 0
    for it in uniq:
        if out:
            candidate = sep.join(out + [it])
        else:
            candidate = it
        if len(candidate) > max_chars:
            break
        out.append(it)
        total = len(out)
    hidden = max(0, len(uniq) - total)
    return (sep.join(out) + (f"{sep}… (+{hidden} more)" if hidden else ""))

def _minmax(nums: List[int]) -> str:
    if not nums:
        return "-"
    mn, mx = min(nums), max(nums)
    return f"{mn}" if mn == mx else f"{mn}-{mx}"

class To01FromDtype(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (C,H,W) or (H,W). Map to [0,1] based on dtype/range.
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.dtype == torch.uint16:
            x = x.float() / 65535.0
        else:
            x = x.float()  # assume already float; DO NOT rescale again
        return x.clamp_(0, 1)


def saved_img_processing(img):
    min_dim_size = min(img.shape[-2], img.shape[-1])
    resize_transform = torchvision.transforms.Compose([
        To01FromDtype(),  # <-- do this BEFORE resize if x is float to avoid weird interpolation with huge values
        torchvision.transforms.Lambda(lambda x: x if x.ndim == 3 else x.unsqueeze(0)),  # HxW -> 1xHxW
        torchvision.transforms.CenterCrop(min_dim_size),
        torchvision.transforms.Resize((224, 224), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
    ])
    return resize_transform(img).unsqueeze(0) # ?

def saved_img_processing_old(img):
    min_dim_size = min(img.shape[0], img.shape[1])
    # min_dim_size = 90
    resize_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(), # uint8/uint16 -> float, channel-first
            torchvision.transforms.CenterCrop((min_dim_size, min_dim_size)),
            torchvision.transforms.Resize(
                (64, 64), torchvision.transforms.InterpolationMode.BILINEAR
            ),
            torchvision.transforms.ConvertImageDtype(torch.float32), # ensures dtype=float32, [0,1] for integer inputs
        ]
    )
    return resize_transform(img).unsqueeze(0)
    # img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    # return resize_transform(img_tensor) / 255.0


#####################
### GENERIC PLOTS ###
#####################
from typing import List, Sequence, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import os
import re
from nocode_robot_programming.jupyter_plot import jupyter_plot as ipt

def _ensure_arrays(train_2d: Sequence[Sequence[float]],
                   test_2d: Sequence[Sequence[float]],
                   model_names: Sequence[str],
                   task_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    train = np.array(train_2d, dtype=float)
    test = np.array(test_2d, dtype=float)
    models = list(model_names)
    tasks = list(task_names)

    if train.shape != test.shape:
        raise ValueError(f"Train/test shapes differ: {train.shape} vs {test.shape}")
    if train.ndim != 2:
        raise ValueError(f"Expected 2D lists. Got array of ndim={train.ndim}")

    m, n = train.shape
    # Try to align axes with provided names
    if len(models) == m and len(tasks) == n:
        return train, test, models, tasks
    if len(models) == n and len(tasks) == m:
        return train.T, test.T, models, tasks
    if len(models) == m:
        if len(tasks) != n:
            raise ValueError("task_names length must match number of columns in the arrays.")
        return train, test, models, tasks
    if len(tasks) == n:
        if len(models) != m:
            raise ValueError("model_names length must match number of rows in the arrays.")
        return train, test, models, tasks

    raise ValueError(
        f"Could not align names with shapes. "
        f"Array shape is {train.shape}, model_names={len(models)}, task_names={len(tasks)}."
    )


def _safe_dir(dirname: str) -> Path:
    p = Path("auto_fig_generator") / dirname
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def plot_heatmap(matrix: np.ndarray, x_labels: List[str], y_labels: List[str], title: str, save_path: Path):
    fig = plt.figure(figsize=(max(6, len(x_labels)*0.8), max(4.5, len(y_labels)*0.6)))
    ax = fig.add_subplot(111)
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Tasks")
    ax.set_ylabel("Models")
    ax.set_title(title)
    # annotate
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    # fig.savefig(save_path, dpi=160, bbox_inches="tight")
    # plt.show()


def plot_grouped_bars_per_task(train: np.ndarray, test: np.ndarray, models: List[str],
                               task: str, task_idx: int, save_path: Path):
    tr = train[:, task_idx]
    te = test[:, task_idx]
    x = np.arange(len(models))
    width = 0.35

    fig = plt.figure(figsize=(max(6, len(models)*0.8), 4.8))
    ax = fig.add_subplot(111)
    ax.bar(x - width/2, tr, width, label="Train")
    ax.bar(x + width/2, te, width, label="Test")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{task}: Train vs Test by Model")
    ax.legend()
    fig.tight_layout()
    # fig.savefig(save_path, dpi=160, bbox_inches="tight")
    # plt.show()


def plot_bars_per_model(values: np.ndarray, x_labels: List[str], model_name: str, title: str, save_path: Path):
    x = np.arange(len(x_labels))
    fig = plt.figure(figsize=(max(6, len(x_labels)*0.8), 4.8))
    ax = fig.add_subplot(111)
    
    x = list(x)
    x.append(x[-1]+1)
    mean = values.mean()
    values = list(values)
    values.append(mean)
    x_labels = list(x_labels)
    x_labels.append("mean")

    colors = ["blue"] * len(x_labels)
    colors[-1] = "orange"

    ax.bar(x, values, color=colors)
    fig.text(0.93, 0.9, f'{mean:.0f}%', ha='center', va='bottom')


    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{title} — {model_name}")
    fig.tight_layout()
    # fig.savefig(save_path, dpi=160, bbox_inches="tight")
    # plt.show()


def visualize_accuracies(train_2d: Sequence[Sequence[float]],
                         test_2d: Sequence[Sequence[float]],
                         model_names: Sequence[str],
                         task_names: Sequence[str],
                         out_dir: str = "accuracy_viz"):
    """Create visualizations and a CSV summary.

    Returns
    -------
    dict with keys:
      - 'out_dir': directory path containing PNGs
      - 'zip_path': path to ZIP with all PNGs
      - 'csv_path': path to CSV summary
    """
    train, test, models, tasks = _ensure_arrays(train_2d, test_2d, model_names, task_names)
    out_path = _safe_dir(out_dir)

    # Summary DataFrame
    cols = pd.MultiIndex.from_product([["Train", "Test"], tasks], names=["Split", "Task"])
    df = pd.DataFrame(index=models, columns=cols, dtype=float)
    for i, m in enumerate(models):
        for j, t in enumerate(tasks):
            df.loc[m, ("Train", t)] = float(train[i, j])
            df.loc[m, ("Test", t)] = float(test[i, j])
    csv_path = out_path / "summary_train_test.csv"
    df.to_csv(csv_path)

    # Heatmaps
    plot_heatmap(train, tasks, models, "Train Accuracy (%)", out_path / "heatmap_train.png"); ipt.save()
    plot_heatmap(test, tasks, models, "Test Accuracy (%)", out_path / "heatmap_test.png"); ipt.save()
    # plot_heatmap(test - train, tasks, models, "Generalization Gap (Test - Train, pp)", out_path / "heatmap_gap.png"); ipt.save()

    ipt.show()
    # Per-task grouped bars
    for j, t in enumerate(tasks):
        plot_grouped_bars_per_task(train, test, models, t, j, out_path / f"grouped_{j:02d}_{_safe_name(t)}.png"); ipt.save()
        if j%4==3:
            ipt.show()

    ipt.show()
    # Per-model bars (test across tasks)
    for i, m in enumerate(models):
        plot_bars_per_model(test[i, :], tasks, m, "Test Accuracy by Task", out_path / f"per_model_test_{i:02d}_{_safe_name(m)}.png"); ipt.save()
        if i%4==3:
            ipt.show()


    ipt.show()
    # ZIP
    # zip_path = out_path.with_suffix(".zip")
    # with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    #     for png in sorted(out_path.glob("*.png")):
    #         zf.write(png, arcname=png.name)
    # return {"out_dir": str(out_path), "zip_path": str(zip_path), "csv_path": str(csv_path)}


