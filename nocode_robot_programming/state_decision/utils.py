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
    min_dim_size = 90
    resize_transform = torchvision.transforms.Compose([
        To01FromDtype(),  # <-- do this BEFORE resize if x is float to avoid weird interpolation with huge values
        torchvision.transforms.Lambda(lambda x: x if x.ndim == 3 else x.unsqueeze(0)),  # HxW -> 1xHxW
        torchvision.transforms.CenterCrop(min_dim_size),
        torchvision.transforms.Resize((64, 64), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
    ])
    return resize_transform(img).unsqueeze(0) # ?

def saved_img_processing_old(img):
    min_dim_size = min(img.shape[0], img.shape[1])
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
