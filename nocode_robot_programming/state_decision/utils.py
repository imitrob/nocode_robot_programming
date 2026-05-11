from typing import List
import os, signal
import psutil  # pip install psutil
import numpy as np
import trajectory_data
import torch, torchvision
from matplotlib.colors import ListedColormap, BoundaryNorm
import cv2
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import random

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

# Preview what would be killed:
# for pid, cmd in list_other_ipykernels():
#     print(pid, cmd)

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

def set_session(name):
    ''' Save to subdirectory '''
    global session
    session = name
    # print(f"[{__name__}] session is set ", name)

def get_session():
    global session
    try:
        session
    except NameError:
        session = ""
    # print(f"[{__name__}] session is read ", session)
    return session


def number_of_saved(video: str, cat: str):
    n = 0
    while os.path.isfile(f'{trajectory_data.package_path}/trajectories/{get_session()}/{video}_{cat}_{n}.npz'):
        n += 1
    return n

def exists(file: str):
    return os.path.isfile(f'{trajectory_data.package_path}/trajectories/{get_session()}/{file}.npz')

class Filename:
    """ Filename parser.
    Supported patterns (with or without '.npz'):
    - 'p0_peg_pick'
        task='p0_peg_pick', offset=0, parent_offset=None, trial=-1
    - 'p0_peg_pick_trial_0'
        task='p0_peg_pick', offset=0, parent_offset=None, trial=0
    - 'p0_peg_pick_branch_at_39'
        task='p0_peg_pick', offset=39, parent_offset=0, trial=-1  # branch from root demo
    - 'p0_peg_pick_branch_from_29_at_158'
        task='p0_peg_pick', offset=158, parent_offset=29, trial=-1
    - 'p0_peg_pick_branch_from_0_at_158'
        task='p0_peg_pick', offset=158, parent_offset=0, trial=-1
    """
    branch_suffix = "branch_at"        # old format
    branch_from_suffix = "branch_from" # new format
    trial_suffix = "trial"
    def __init__(self, filename: str, offset: int | None = None, parent_offset: int = 0, trial: int = -1, init_exec_trial: bool = False):
        if offset is None:
            self.from_filename(filename)
        else:
            self.from_params(filename, offset, parent_offset, trial)
        
        if init_exec_trial:
            self.add_execution_trial()

    def from_params(self, task: str, offset: int, parent_offset: int, trial: int):
        self.task = task
        self.offset = offset
        self.parent_offset = parent_offset
        self.trial = trial

    def from_filename(self, filename: str):
        # Normalize extension
        if filename.endswith(".npz"):
            self.filename = filename
            self.name: str = filename[:-4]  # without '.npz'
        else:
            self.filename = filename + ".npz"
            self.name = filename

        # Parse trial suffix
        trial_split = self.name.split(f"_{self.trial_suffix}_")
        self.part_name = trial_split[0]

        if len(trial_split) > 1:
            # everything after "_trial_" is the integer trial id
            self.trial: int = int(trial_split[1])
        else:
            self.trial = -1  # -1 = demonstration (no explicit trial)

        # default values (will be overwritten if we detect branches)
        self.offset: int = 0
        self.parent_offset = 0

        # New format: {task}_branch_from_{parent}_at_{offset}
        parent_split = self.part_name.split(f"_{self.branch_from_suffix}_")
        if len(parent_split) > 1:
            # parent_split[0] = task, parent_split[1] = '{parent}_at_{offset}'
            parent_and_at = parent_split[1].split("_at_")
            if len(parent_and_at) == 2:
                self.task: str = parent_split[0]
                self.parent_offset = int(parent_and_at[0])
                self.offset = int(parent_and_at[1])
                return  # we're done

        # Old format: {task}_branch_at_{offset}
        branch_split = self.part_name.split(f"_{self.branch_suffix}_")
        if len(branch_split) > 1:
            # branch from root demo (offset 0)
            self.task = branch_split[0]
            self.offset = int(branch_split[1])
            self.parent_offset = 0
        else:
            # No branch info at all: root demonstration
            self.task = self.part_name

    def add_execution_trial(self):
        assert self.is_demo, "Adding execution trial, but the filename is execution trial already!"
        self.trial = number_of_saved(self.to_str(), "trial") # trials 0, ..., n-1 exists

    def find_unique(self):
        """ Iterate offset until file not exists """
        while self.exists():
            self.offset += 1

    def exists(self):
        return exists(self.to_str())

    @property
    def is_demo(self) -> bool:
        """True if this is a demonstration (no trial)."""
        return self.trial == -1

    def to_str(self):
        """ Constructs filename without .npz extension
        """
        if self.offset == 0:
            branchfromat = f""
        else:
            branchfromat = f"_branch_from_{self.parent_offset}_at_{self.offset}"

        if self.trial == -1:
            return f"{self.task}{branchfromat}"
        elif self.trial >= 0:
            return f"{self.task}{branchfromat}_trial_{self.trial}"
        else: raise Exception("self.trial not match")

    @property
    def person(self) -> str:
        return self.task.split("_")[0]
    
    @property
    def modality(self) -> str:
        if len(self.task.split("_")) < 2:
            return "none"
        else:
            return self.task.split("_")[1]
    
    @property
    def task_userstudy(self) -> str:
        if len(self.task.split("_")) < 3:
            return self.task
        else:
            return " ".join(self.task.split("_")[2:])
    
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

def visualize_video_frame_with_text(image, text: str = "", color: tuple[int, int, int] = (0,0,255), press_for_next_frame: bool = False, resize=(64,64)):
    image = image.squeeze().astype(np.uint8)
    image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)
    
    cv2.putText(image, text, (0, 12), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, color, 1, 2)
    
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Image",3700,0)
    cv2.resizeWindow("Image", 640, 640)
    zoomed_image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Image", zoomed_image)

    if press_for_next_frame:
        cv2.waitKey(0)  # Wait for a key press to close the window
    if cv2.waitKey(25) & 0xFF == 27:  # Press 'Esc' to exit
        return True
    return False


def user_study_tasks_only(dataset_builder, tasktemplates_to_evaluate=['peg_pick', 'probe', 'wrap']):
    tasks_to_evaluate = []
    for t in dataset_builder.all_tasks:
        for tt in tasktemplates_to_evaluate:
            if (tt in t):
                tasks_to_evaluate.append(t) 
    return tasks_to_evaluate

def user_study_additional_task_only(dataset_builder):
    # I made some filter, but I don't like it like this:
    tasktemplates_to_evaluate = ['additional']
    tasks_to_evaluate = []
    for t in dataset_builder.all_tasks:
        for tt in tasktemplates_to_evaluate:
            if (tt in t):
                tasks_to_evaluate.append(t) 
    return tasks_to_evaluate


def user_study_nice_model_names(model_names):
    TO_NICE_NAMES = { # complicated name -> nice name
    'dinov2_vits14,224,mean': 'dinov2 small mean',
    'facebook/dinov3-vits16-pretrain-lvd1689m,224,mean': 'dinov3 small mean',
    'facebook/dinov3-vitl16-pretrain-lvd1689m,224,mean': 'dinov3 large mean',
    'dinov2_vits14,224,concat': 'dinov2 small concat',
    'dinov2_vits14,224,attn,hard,mean,0.4': 'dinov2 small attn',
    'dinov2_vits14,224,MIL,H=128,e=1000': 'dinov2 small MIL',
    'SIFT': "SIFT",
    'AEGP,bin=False': 'AEGP Multiclass',
    }

    for i in range(len(model_names)):
        if model_names[i] in TO_NICE_NAMES:
            model_names[i] = TO_NICE_NAMES[model_names[i]]

    return model_names

def y_cls_to_nice_name(y_cls):
    f = Filename(y_cls[0])
    name = str(f.task)
    name += "_0"

    for n in y_cls:
        f = Filename(n) 
        if f.offset != 0:
            name += f"|{f.offset}"
    return name



def user_study_plot_hist(stats, 
                         brackets = [[0.3, 0.7, "Camera doesn't see\ndiscriminated location.", 0.3]],
                         savename: str = "sns_hist_2.pdf",
                         print_examples: int = 0,
                         print_howmany_over_90: bool = True,
                         bins: int = 21,
                         folder: str = "plot", 
                        ):

    keys = list(stats.keys())
    vals = np.array(list(stats.values()), dtype=float)
    if vals.max() > 1.5:  # if you stored 90 instead of 0.90
        vals = vals / 100.0

    bins = np.linspace(0, 1, bins)          # 10 bins from 0%..100%
    
    fig, ax = plt.subplots(figsize=(4, 2))
    counts, edges, patches = ax.hist(vals, bins=bins, edgecolor="black")

    # group keys by bin
    idx = np.digitize(vals, edges) - 1
    idx[idx == len(edges) - 1] = len(edges) - 2  # right-edge fix
    groups = [[] for _ in range(len(edges) - 1)]
    for k, i in zip(keys, idx):
        groups[i].append(k)

    # annotate each bar with count + example key(s)
    for i, (c, p) in enumerate(zip(counts, patches)):
        if c == 0:
            continue
        x = p.get_x() + p.get_width() / 2
        y = p.get_height() - 3

        txt = f"{int(c)}\n"
        # print(f"At {i=}: {groups[i]}")
        if print_examples > 0:
            examples = groups[i][:print_examples]
            
            txt += ", ".join(examples)
        ax.text(x, y, txt, ha="center", va="bottom", fontsize=6, rotation=0)

    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Counts")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f"{t:.0%}" for t in np.linspace(0, 1, 6)])

    def bracket(ax, x1, x2, text, y_ax):
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

        # horizontal line
        ax.plot([x1, x2], [y_ax, y_ax], transform=trans, clip_on=False)
        # vertical ticks
        ax.plot([x1, x1], [y_ax-0.03, y_ax], transform=trans, clip_on=False)
        ax.plot([x2, x2], [y_ax-0.03, y_ax], transform=trans, clip_on=False)

        # text
        ax.text((x1+x2)/2, y_ax+0.02, text,
                ha="center", va="bottom", transform=trans)

    for b in brackets:
        bracket(ax, b[0], b[1], b[2], b[3])

    if print_howmany_over_90:
        n_over_90 = (vals > 0.899).sum()
        n_under_80 = (vals < 0.799).sum()
        ax.text(0.03, 0.95,
                f"> 90%: {n_over_90}/{len(vals)}\n< 80%: {n_under_80}/{len(vals)}",
                transform=ax.transAxes,
                ha="left", va="top")

    plt.tight_layout()
    from pathlib import Path

    p = Path(f"auto_fig_generator/{folder}/")
    p.mkdir(parents=True, exist_ok=True)

    plt.savefig(Path(f"auto_fig_generator/{folder}/") / savename)
    plt.show()


def user_study_plot_hist_grouped(
    grouped_stats,
    brackets=[[0.3, 0.7, "Camera doesn't see\ndiscriminated location.", 0.3]],
    savename: str = "sns_hist_grouped.pdf",
    print_howmany_over_90: bool = True,
    bins: int = 21,
    folder: str = "plot",
    colors: list | None = None,
):
    """Stacked histogram comparing multiple dataset groups with different bar colors.

    grouped_stats: {group_name: {task_name: accuracy}}
    """
    print("grouped_stats keys: ", grouped_stats.keys())
    _default_colors = ["#00e676", "#e74c3c", "#3498db", "#e67e22", "#9b59b6", "#1abc9c"]
    if colors is None:
        colors = _default_colors[:len(grouped_stats)]

    bins_arr = np.linspace(0, 1, bins)
    vals_list, labels = [], []
    for group_name, stats_group in grouped_stats.items():
        vals = np.array(list(stats_group.values()), dtype=float)
        if len(vals) and vals.max() > 1.5:
            vals /= 100.0
        vals_list.append(vals)
        labels.append(f"{group_name} (n={len(vals)})")

    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.hist(vals_list, bins=bins_arr, stacked=True, label=labels,
            color=colors, edgecolor="black", linewidth=0.4)
    ax.legend(fontsize=6, loc="lower left")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Counts")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f"{t:.0%}" for t in np.linspace(0, 1, 6)])

    def bracket(ax, x1, x2, text, y_ax):
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.plot([x1, x2], [y_ax, y_ax], transform=trans, clip_on=False)
        ax.plot([x1, x1], [y_ax - 0.03, y_ax], transform=trans, clip_on=False)
        ax.plot([x2, x2], [y_ax - 0.03, y_ax], transform=trans, clip_on=False)
        ax.text((x1 + x2) / 2, y_ax + 0.02, text, ha="center", va="bottom", transform=trans)

    for b in brackets:
        bracket(ax, b[0], b[1], b[2], b[3])

    if print_howmany_over_90:
        lines = []
        for group_name, vals in zip(grouped_stats.keys(), vals_list):
            if len(vals):
                lines.append(f"{group_name}: {(vals > 0.899).sum()}/{len(vals)} >90%")
        ax.text(0.03, 0.95, "\n".join(lines), transform=ax.transAxes,
                ha="left", va="top", fontsize=5)

    plt.tight_layout()
    from pathlib import Path
    Path(f"auto_fig_generator/{folder}/").mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(f"auto_fig_generator/{folder}/") / savename)
    plt.show()


def set_seed(seed: int = 48):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)