from nocode_robot_programming.state_decision.utils import Filename
from nocode_robot_programming.state_decision.dataloader import ImageDatasetView, saved_img_processing
import torch
from copy import deepcopy
import numpy as np
from pathlib import Path
import os 
from typing import List
import shutil

import trajectory_data
from nocode_robot_programming.artificial_dataset.dataset_generator import generate_run, RunArgs

from nocode_robot_programming.state_decision.dataset_D2 import get_dataset_view_artificial_dataset

def d1_spawned_box(dummy: None):
    datasets = []
    # clear folder 
    dsfolder = Path(trajectory_data.package_path) / 'trajectories' / 'artificial_dataset'
    assert dsfolder.is_dir()
    shutil.rmtree(dsfolder)

    generate_run(RunArgs(folder="d1_train", cls="peg",    offset=(0,0,0), spawn_box=False))
    generate_run(RunArgs(folder="d1_test",  cls="peg",    offset=(0,0,0), spawn_box=False))
    generate_run(RunArgs(folder="d1_test",  cls="anomaly",offset=(0,0,0), spawn_box=True))
    # generate_run(RunArgs(folder="d1_train", cls="nopeg",  offset=(0,0,0), spawn_box=False))
    # generate_run(RunArgs(folder="d1_test",  cls="nopeg" , offset=(0,0,0), spawn_box=True))

    d_train = get_dataset_view_artificial_dataset(folder="d1_train")
    d_test = get_dataset_view_artificial_dataset(folder="d1_test", y_cls=d_train.y_cls)

    datasets.append([d_train, d_test, "Anomaly Box"])
    return datasets

def d1_spawned_box_2cls(dummy: None):
    datasets = []
    # clear folder 
    dsfolder = Path(trajectory_data.package_path) / 'trajectories' / 'artificial_dataset'
    assert dsfolder.is_dir()
    shutil.rmtree(dsfolder)


    generate_run(RunArgs(folder="d1_train", cls="peg",    offset=(0,0,0), spawn_box=False))
    generate_run(RunArgs(folder="d1_train", cls="anomaly",    offset=(0,0,0), spawn_box=False, mesh="taskboard_nopeg.stl"))
    generate_run(RunArgs(folder="d1_test",  cls="peg",    offset=(0,0,0), spawn_box=False))
    generate_run(RunArgs(folder="d1_test",  cls="anomaly",offset=(0,0,0), spawn_box=True))
    # generate_run(RunArgs(folder="d1_train", cls="nopeg",  offset=(0,0,0), spawn_box=False))
    # generate_run(RunArgs(folder="d1_test",  cls="nopeg" , offset=(0,0,0), spawn_box=True))

    d_train = get_dataset_view_artificial_dataset(folder="d1_train")
    d_test = get_dataset_view_artificial_dataset(folder="d1_test", y_cls=d_train.y_cls)

    datasets.append([d_train, d_test, "Anomaly Box, 2cls"])

    return datasets
