from typing import List
import os, signal
import psutil  # pip install psutil
import numpy as np
import trajectory_data
import torch, torchvision
from matplotlib.colors import ListedColormap, BoundaryNorm

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


def plot_heatmap(matrix: np.ndarray, x_labels: List[str], y_labels: List[str], title: str, save_path: Path, jupyter_plot: bool):
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

    if jupyter_plot:
        ipt.save()
    else:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.show()


def plot_grouped_bars_per_task(train: np.ndarray, test: np.ndarray, models: List[str],
                               task: str, task_idx: int, save_path: Path, jupyter_plot: bool):
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
    
    if jupyter_plot:
        ipt.save()
    else:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.show()


def plot_bars_per_model(values: np.ndarray, x_labels: List[str], model_name: str, title: str, save_path: Path, jupyter_plot: bool):
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

    if jupyter_plot:
        ipt.save()
    else:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.show()


def visualize_accuracies(train_2d: Sequence[Sequence[float]],
                         test_2d: Sequence[Sequence[float]],
                         model_names: Sequence[str],
                         task_names: Sequence[str],
                         out_dir: str = "accuracy_viz",
                         jupyter_plot: bool = True):
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

    difficulty_group = []
    modality_group = []
    task_group = []
    diff_mod_group = []
    for task_name in task_names:
        difficulty_group.append( task_name.split(" ")[-1] )
        modality_group.append( task_name.split(" ")[-2] )
        task_group.append( " ".join(task_name.split(" ")[:-2]) )
        diff_mod_group.append( " ".join(task_name.split(" ")[-2:]) )

    difficulty_means = []
    modality_means = []
    task_means = []
    diff_mod_means = []

    a_idxs = None
    b_idxs = None
    c_idxs = None
    d_idxs = None
    for model_values in test_2d:
        a_ = pd.Series(np.array(model_values)).groupby(difficulty_group).mean()
        if a_idxs is not None:
            assert a_.index.to_list() == a_idxs, f"{a_.index.to_list()} != {difficulty_group}"
        a_idxs = a_.index.to_list()

        b_ = pd.Series(np.array(model_values)).groupby(modality_group).mean()
        if b_idxs is not None:    
            assert b_.index.to_list() == b_idxs, f"{b_.index.to_list()} != {modality_group}"
        b_idxs = b_.index.to_list()

        c_ = pd.Series(np.array(model_values)).groupby(task_group).mean()
        if c_idxs is not None:
            assert c_.index.to_list() == c_idxs, f"{c_.index.to_list()} != {task_group}"
        c_idxs = c_.index.to_list()

        d_ = pd.Series(np.array(model_values)).groupby(diff_mod_group).mean()
        if d_idxs is not None:
            assert d_.index.to_list() == d_idxs, f"{d_.index.to_list()} != {diff_mod_group}"
        d_idxs = d_.index.to_list()

        difficulty_means.append(a_.to_list())
        modality_means.append(b_.to_list())
        task_means.append(c_.to_list())
        diff_mod_means.append(d_.to_list())
    
    print("   difficulty_means  ", difficulty_means)
    
    plot_heatmap(np.array(difficulty_means), a_idxs, models, "Test Accuracy (%)", out_path / "heatmap_test_dfcly_group.png", jupyter_plot)
    plot_heatmap(np.array(modality_means), b_idxs, models, "Test Accuracy (%)", out_path / "heatmap_test_mdlt_group.png", jupyter_plot)
    if jupyter_plot:
        ipt.show()
    plot_heatmap(np.array(task_means), c_idxs, models, "Test Accuracy (%)", out_path / "heatmap_test_task_group.png", jupyter_plot)
    plot_heatmap(np.array(diff_mod_means), d_idxs, models, "Test Accuracy (%)", out_path / "heatmap_test_dfclt_task_group.png", jupyter_plot)
    if jupyter_plot:
        ipt.show()

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
    plot_heatmap(train, tasks, models, "Train Accuracy (%)", out_path / "heatmap_train.png", jupyter_plot)
    plot_heatmap(test, tasks, models, "Test Accuracy (%)", out_path / "heatmap_test.png", jupyter_plot)
    # plot_heatmap(test - train, tasks, models, "Generalization Gap (Test - Train, pp)", out_path / "heatmap_gap.png", jupyter_plot)
    if jupyter_plot:
        ipt.show()
    # Per-task grouped bars
    for j, t in enumerate(tasks):
        plot_grouped_bars_per_task(train, test, models, t, j, out_path / f"grouped_{j:02d}_{_safe_name(t)}.png", jupyter_plot)
        if jupyter_plot and j%4==3:
            ipt.show()

    if jupyter_plot:
        ipt.show()
    # Per-model bars (test across tasks)
    for i, m in enumerate(models):
        plot_bars_per_model(test[i, :], tasks, m, "Test Accuracy by Task", out_path / f"per_model_test_{i:02d}_{_safe_name(m)}.png", jupyter_plot)
        if jupyter_plot and i%4==3:
            ipt.show()

    if jupyter_plot:
        ipt.show()
    else:
        # ZIP
        zip_path = out_path.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for png in sorted(out_path.glob("*.png")):
                zf.write(png, arcname=png.name)
        return {"out_dir": str(out_path), "zip_path": str(zip_path), "csv_path": str(csv_path)}





import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import ListedColormap
import matplotlib as mpl

def likelihood_sparklines_withtruelabels(
    likelihoods,             # shape: (c, n)
    class_names,             # list/seq length c
    true_labels=None,        # len n; ints in [0..c-1] or class-name strings
    truth_style="barcode",   # 'barcode' (default), 'strip', 'overlay', 'both'
    cols=6,
    sharey=True,
    figsize=None,
    dpi=200,
    fig_bg="#f6f7f9",
    tile_bg="#ffffff",
    rounding=10,
    fill_alpha=0.25,
    line_lw=1.6,
    truth_band_height=0.08,     # height for inline barcode (axes fraction)
    truth_bar_alpha=0.85,       # opacity for inline barcode
    truth_overlay_alpha=0.10,   # opacity for vertical overlay spans
):
    """
    Adds ground-truth encoding without clutter:
      - 'barcode': thin colored band at the bottom of each tile where that class is true
      - 'strip': one consolidated strip across the figure showing the true class timeline
      - 'overlay': faint vertical spans in tiles where that class is true
      - 'both': barcode + strip

    Returns (fig, axes, truth_ax_or_None)
    """
    L = np.asarray(likelihoods, dtype=float)
    if L.ndim != 2:
        raise ValueError("likelihoods must be a 2D array of shape (c, n)")
    C, N = L.shape
    if len(class_names) != C:
        raise ValueError("class_names length must match number of classes (c)")

    # Map true_labels to integer indices (0..C-1)
    true_idx = None
    if true_labels is not None:
        if len(true_labels) != N:
            raise ValueError("true_labels length must match number of steps (n)")
        name_to_idx = {str(nm): i for i, nm in enumerate(class_names)}
        if isinstance(true_labels[0], str):
            true_idx = np.array([name_to_idx[str(t)] for t in true_labels], dtype=int)
        else:
            true_idx = np.asarray(true_labels, dtype=int)
        if np.any((true_idx < 0) | (true_idx >= C)):
            raise ValueError("true_labels indices out of range")

    # Layout: add a short row for the figure-level truth strip if needed
    want_strip = truth_style in ("strip", "both")
    rows = int(np.ceil(C / cols))
    if figsize is None:
        figsize = (max(2, cols) * 1.15, (rows + (0.35 if want_strip else 0)) * 1.15)

    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.patch.set_facecolor(fig_bg)

    if want_strip:
        gs = fig.add_gridspec(rows + 1, cols, wspace=0.1, hspace=0.2,
                              height_ratios=[1]*rows + [0.25])
        strip_row = rows
    else:
        gs = fig.add_gridspec(rows, cols, wspace=0.1, hspace=0.2)

    # Shared y-limits
    if sharey:
        y_min = np.nanmin(L)
        y_max = np.nanmax(L)
        if not np.isfinite(y_min): y_min = 0.0
        if not np.isfinite(y_max): y_max = 1.0
        pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
        shared_ylim = (y_min - pad, y_max + pad)
    else:
        shared_ylim = None

    xs = np.linspace(0, 1, N)

    # Stable class colors (same color everywhere for a given class)
    cmap = plt.get_cmap("tab20")
    class_colors = [cmap((i*2) % 20) for i in range(C)]  # 0,2,4,... -> dark set



    axes = []
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        if want_strip and r == strip_row:
            # This row is reserved for the truth strip
            continue
        ax = fig.add_subplot(gs[r, c])
        axes.append(ax)

        if i >= C:
            ax.axis("off")
            continue

        # Rounded tile background
        ax.set_facecolor(tile_bg)
        tile = FancyBboxPatch(
            (-0.03, -0.03), 1.06, 1.06,
            boxstyle=f"round,pad=0.012,rounding_size={rounding}",
            transform=ax.transAxes, linewidth=0, facecolor=tile_bg, zorder=0
        )
        ax.add_artist(tile)

        y = L[i]
        color = class_colors[i]

        # Optional overlay: faint vertical spans where this class is the ground truth
        if true_idx is not None and truth_style in ("overlay", "both"):
            mask = (true_idx == i)
            # Build contiguous spans in data space
            def spans_from_mask(mask, xs):
                spans = []
                in_run = False
                for k, m in enumerate(mask):
                    if m and not in_run:
                        start = xs[k-1] if k > 0 else xs[0]
                        in_run = True
                    elif not m and in_run:
                        end = xs[k]
                        spans.append((start, end))
                        in_run = False
                if in_run:
                    spans.append((start, xs[-1]))
                return spans
            for s, e in spans_from_mask(mask, xs):
                ax.axvspan(s, e, color=color, alpha=truth_overlay_alpha, zorder=0.5)

        # Likelihood curve and fill
        ax.plot(xs, y, lw=line_lw, color=color, zorder=2)
        ax.fill_between(xs, 0, y, alpha=fill_alpha, color=color, zorder=1)
        ax.scatter([xs[-1]], [y[-1]], s=10, color=color, zorder=3)

        # Inline barcode band (per-tile)
        if true_idx is not None and truth_style in ("barcode", "both"):
            band_ax = ax
            band = (true_idx == i).astype(int)[None, :]  # shape (1, N)
            band_cmap = ListedColormap([(0, 0, 0, 0), color])  # 0 transparent, 1 colored
            # Draw in x-data coords and y as axes fraction (0=bottom, 1=top)
            band_ax.imshow(
                band,
                extent=(xs[0], xs[-1], 0.0, truth_band_height),
                cmap=band_cmap,
                interpolation="nearest",
                aspect="auto",
                origin="lower",
                transform=band_ax.get_xaxis_transform(),
                alpha=truth_bar_alpha,
                zorder=2.5,
            )

        # Clean micro-axes
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        # Y-limits
        if shared_ylim:
            ax.set_ylim(shared_ylim)
        else:
            yymin, yymax = np.nanmin(y), np.nanmax(y)
            pad = 0.05 * (yymax - yymin if yymax > yymin else 1.0)
            ax.set_ylim(yymin - pad, yymax + pad)

        def _safe_label(s, max_len=28):
            t = str(s)
            if mpl.rcParams.get('text.usetex', False):
                t = t.replace('_', r'\_')   # escape for TeX
            if len(t) > max_len:
                t = t[:max_len-1] + '…'     # keep tiles tidy
            return t

        ax.text(0.02, 0.96, _safe_label(class_names[i]),
            transform=ax.transAxes, ha="left", va="top",
            fontsize=6, weight="bold", clip_on=True)


    truth_ax = None
    if want_strip:
        # One consolidated strip across the bottom row
        truth_ax = fig.add_subplot(gs[strip_row, :])
        truth_ax.set_facecolor(tile_bg)
        if true_idx is None:
            # Empty but still visually coherent
            truth_ax.axis("off")
        else:
            strip = true_idx[None, :]  # shape (1, N)
            strip_cmap = ListedColormap(class_colors)
            truth_ax.imshow(
                strip,
                extent=(xs[0], xs[-1], 0, 1),
                cmap=strip_cmap,
                interpolation="nearest",
                aspect="auto",
                origin="lower",
            )
            # Minimal label to signify ground truth
            truth_ax.text(-0.01, 0.5, "Truth",
                          transform=truth_ax.transAxes,
                          ha="right", va="center",
                          fontsize=9, weight="bold")
            truth_ax.set_xticks([]); truth_ax.set_yticks([])
            for sp in truth_ax.spines.values():
                sp.set_visible(False)

    return fig, axes, truth_ax


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def likelihood_sparklines(
    likelihoods,             # shape: (c, n)
    class_names,             # list/seq length c
    cols=6,                  # how many tiles per row
    sharey=True,             # share y-scale across classes
    figsize=None,            # (w,h) in inches; auto if None
    dpi=200,                 # crisp small plots
    fig_bg="#f6f7f9",        # figure background color
    tile_bg="#ffffff",       # tile background
    rounding=10,             # corner radius for tiles
    fill_alpha=0.25,         # area fill opacity
    line_lw=1.6             # line thickness
):
    """
    Draws a grid of tiny likelihood plots (one per class) with minimal chrome.
    likelihoods: array-like (c, n)
    class_names: sequence of length c
    Returns (fig, axes) for further customization/saving.
    """
    L = np.asarray(likelihoods, dtype=float)
    if L.ndim != 2:
        raise ValueError("likelihoods must be a 2D array of shape (c, n)")
    C, N = L.shape
    if len(class_names) != C:
        raise ValueError("class_names length must match number of classes (c)")

    rows = int(np.ceil(C / cols))
    if figsize is None:
        # Small default: ~1.15" per tile in width/height
        figsize = (max(2, cols) * 1.15, max(1, rows) * 1.15)

    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.patch.set_facecolor(fig_bg)
    gs = fig.add_gridspec(rows, cols, wspace=0.1, hspace=0.2)

    # Shared y-limits for comparability if requested
    if sharey:
        y_min = np.nanmin(L)
        y_max = np.nanmax(L)
        if not np.isfinite(y_min): y_min = 0.0
        if not np.isfinite(y_max): y_max = 1.0
        pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
        shared_ylim = (y_min - pad, y_max + pad)
    else:
        shared_ylim = None

    xs = np.linspace(0, 1, N)
    cmap = plt.get_cmap("tab10")

    axes = []
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(gs[r, c])
        axes.append(ax)

        if i >= C:
            ax.axis("off")
            continue

        # Rounded tile background
        ax.set_facecolor(tile_bg)
        tile = FancyBboxPatch(
            (-0.03, -0.03), 1.06, 1.06,
            boxstyle=f"round,pad=0.012,rounding_size={rounding}",
            transform=ax.transAxes, linewidth=0, facecolor=tile_bg, zorder=0
        )
        ax.add_artist(tile)

        y = L[i]
        color = cmap(i % 10)

        # Plot and fill
        line, = ax.plot(xs, y, lw=line_lw, color=color, zorder=2)
        ax.fill_between(xs, 0, y, alpha=fill_alpha, color=color, zorder=1)

        # Optional final-point marker (subtle)
        ax.scatter([xs[-1]], [y[-1]], s=10, color=color, zorder=3)

        # Remove all ticks/labels/spines for a clean micro-plot
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Y limits
        if shared_ylim:
            ax.set_ylim(shared_ylim)
        else:
            yymin, yymax = np.nanmin(y), np.nanmax(y)
            pad = 0.05 * (yymax - yymin if yymax > yymin else 1.0)
            ax.set_ylim(yymin - pad, yymax + pad)

        # Class label inside the tile
        ax.text(0.02, 0.96, str(class_names[i]),
                transform=ax.transAxes, ha="left", va="top",
                fontsize=9, weight="bold")

    return fig, axes


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl

def _safe_label(s, max_len=40):
    t = str(s)
    if mpl.rcParams.get('text.usetex', False):
        t = t.replace('_', r'\_')
    return t if len(t) <= max_len else t[:max_len-1] + '…'

def likelihood_single_axis(
    likelihoods,             # shape: (c, n)
    class_names,             # list length c
    true_labels=None,        # len n; class indices [0..c-1] or names
    mode="barcode",          # 'barcode' (adds truth strip), or 'none'
    figsize=(7, 2.0),
    dpi=220,
    fig_bg="#f6f7f9",
    bg_colors=("#ffffff", "#f1f3f6"),   # alternating background band colors
    show_separators=True,    # thin lines along ranked boundaries
    separator_color="#e3e7ee",
    separator_lw=0.6,
    line_lw=1.6,
    winner_fill=True,        # fill only when class is argmax to avoid mixing
    winner_fill_alpha=0.28,
    event_x=None,                 # timestep to mark; int index (default) or float in x
    event_units="index",          # 'index' -> treat event_x as step index, 'x' -> already in [0..1] data coords
    event_line_kwargs=None,       # style overrides for the vertical line
    event_star_kwargs=None,       # style overrides for the star marker
    ):
    """
    Single-axes plot:
      • Background is painted by the zones between ranked likelihoods at each timestep.
      • Each class curve is plotted on top with stable colors.
      • Optional fill only where the class is top-1 (no color mixing).
      • Optional bottom barcode indicating ground-truth over time.

    Returns (fig, ax).
    """
    L = np.asarray(likelihoods, dtype=float)
    if L.ndim != 2:
        raise ValueError("likelihoods must be (c, n)")
    C, N = L.shape
    if len(class_names) != C:
        raise ValueError("class_names length must match c")
    xs = np.linspace(0, 1, N)

    def _event_positions(event_x, units, N):
        if event_x is None:
            return []
        # allow scalar or iterable
        xs_raw = np.atleast_1d(event_x)
        if units == "index":
            # map integer step k -> normalized x in [0,1]
            if N <= 1:
                return [0.0] * len(xs_raw)
            return [float(int(k)) / (N - 1) for k in xs_raw]
        elif units == "x":
            return [float(x) for x in xs_raw]
        else:
            raise ValueError("event_units must be 'index' or 'x'")

    # Stable class colors (dark half of tab20)
    tab20 = plt.get_cmap("tab20")
    class_colors = [tab20.colors[(i*2) % 20] for i in range(C)]

    # Map true_labels to indices (if provided)
    true_idx = None
    if true_labels is not None:
        if len(true_labels) != N:
            raise ValueError("true_labels length must match n")
        name_to_idx = {str(nm): i for i, nm in enumerate(class_names)}
        if isinstance(true_labels[0], str):
            true_idx = np.array([name_to_idx[str(t)] for t in true_labels], dtype=int)
        else:
            true_idx = np.asarray(true_labels, dtype=int)
        if np.any((true_idx < 0) | (true_idx >= C)):
            raise ValueError("true_labels indices out of range")

    # --- Background: zones between ranked likelihoods ---
    # Sort likelihoods per timestep ascending; shape: (C, N)
    order = np.argsort(L, axis=0)
    sorted_vals = np.take_along_axis(L, order, axis=0)

    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.patch.set_facecolor(fig_bg)
    ax = fig.add_subplot(111)
    ax.set_xlim(xs[0], xs[-1])

    # Y-limits padded a touch
    ymin, ymax = float(np.nanmin(L)), float(np.nanmax(L))
    pad = 0.04 * (ymax - ymin if ymax > ymin else 1.0)
    ax.set_ylim(ymin - pad, ymax + pad)

    # Paint alternating bands between ranks: [sorted[k], sorted[k+1]]
    for k in range(C - 1):
        y0 = sorted_vals[k]
        y1 = sorted_vals[k + 1]
        ax.fill_between(
            xs, y0, y1,
            facecolor=bg_colors[k % 2],
            edgecolor='none',
            zorder=0
        )
        if show_separators:
            ax.plot(xs, y0, color=separator_color, lw=separator_lw, zorder=0.5)
    if show_separators and C > 0:
        # top envelope separator for completeness
        ax.plot(xs, sorted_vals[-1], color=separator_color, lw=separator_lw, zorder=0.5)

    # --- Foreground: class curves (and optional fills only when top-1) ---
    argmax_idx = np.argmax(L, axis=0)
    for i in range(C):
        y = L[i]
        col = class_colors[i]

        if winner_fill:
            # Fill only where this class is the maximum to avoid color overlap
            mask = (argmax_idx == i)
            # draw contiguous spans for the mask
            start = None
            for k in range(N):
                if mask[k] and start is None:
                    start = k
                if (start is not None) and (k == N-1 or not mask[k+1]):
                    end = k
                    seg_x = xs[start:end+1]
                    seg_y = y[start:end+1]
                    ax.fill_between(seg_x, [ax.get_ylim()[0]]*len(seg_x), seg_y,
                                    color=col, alpha=winner_fill_alpha, zorder=1.5)
                    start = None

        # Line on top
        ax.plot(xs, y, color=col, lw=line_lw, zorder=2.0, solid_capstyle="round")

    # Minimal: remove ticks/box
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Class labels as a compact legend-like text block (left, inside)
    from matplotlib.lines import Line2D

    # Create a custom legend using Line2D handles for accurate colors
    handles = [
        Line2D([0], [0], color=class_colors[i], lw=2, label=_safe_label(class_names[i]))
        for i in range(C)
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=min(C, 6),   # wrap long legends into rows
        fontsize=8.5,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.4,
    )
    # Optional: bottom barcode for ground truth
    if true_idx is not None and mode == "barcode":
        strip_h = 0.08  # fraction of axes height
        strip_cmap = ListedColormap(class_colors)
        # Discrete mapping: -0.5..0.5 -> color 0, 0.5..1.5 -> color 1, ..., C-1
        norm = BoundaryNorm(np.arange(-0.5, C + 0.5, 1), C)

        y0, y1 = ax.get_ylim()
        ax.imshow(
            true_idx[None, :],
            extent=(xs[0], xs[-1], y0, y0 + strip_h * (y1 - y0)),
            cmap=strip_cmap,
            norm=norm,                 # <-- key line
            interpolation="nearest",
            aspect="auto",
            origin="lower",
            zorder=2.2
        )
        ax.text(-0.01, 0.02, "Truth", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, weight="bold", color="#2b2f36")

    # --- Optional event marker(s): vertical dashed line + orange star at bottom ---
    evt_xs = _event_positions(event_x, event_units, N)
    if evt_xs:
        # defaults
        _line_kw = dict(color="black", lw=1.2, linestyle=(0, (4, 4)), zorder=3.0)
        _star_kw = dict(marker="*", markersize=9, color="orange", zorder=3.2, clip_on=False)
        if event_line_kwargs:
            _line_kw.update(event_line_kwargs)
        if event_star_kwargs:
            _star_kw.update(event_star_kwargs)

        y0, y1 = ax.get_ylim()
        # place star a bit above the very bottom (and above the truth strip if present)
        star_y_axes = 0.015  # axes-fraction from bottom
        if (true_idx is not None) and (mode == "barcode"):
            star_y_axes = 0.03  # sit just above the barcode strip

        for x_ev in evt_xs:
            ax.axvline(x_ev, **_line_kw)
            # star anchored in x-data, y in axes coords
            ax.plot([x_ev], [star_y_axes], transform=ax.get_xaxis_transform(), **_star_kw)