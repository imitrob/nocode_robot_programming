import os, signal
import psutil  # pip install psutil
import numpy as np
import trajectory_data

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

