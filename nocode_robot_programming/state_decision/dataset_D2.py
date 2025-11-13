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

def get_dataset_view_artificial_dataset(folder: str, y_cls: List[str] | None = None) -> ImageDatasetView:
    """ Fileloader used to create dataset
        1. Image preprocessing
        2. Returns as a ImageDatasetView object

    y_cls = ["label1", "label2"] - classification labels
    if y_cls is None, it searches for labels (can shuffle them)
    """
    X_parts, Xt_parts, y_int_parts, y_name_parts = [], [], [], []
    
    if y_cls is None:
        y_cls_do = True
        y_cls = []
    else:
        y_cls_do = False

    p = Path(trajectory_data.package_path) / 'trajectories' / 'artificial_dataset' / folder
    
    for c, dir_ in enumerate(p.iterdir()):
        images = np.load(f"{dir_}/grayscale_uint8.npz")["images"]  # (N, H, W), uint8
        images = torch.tensor(images)
        nsamples, H, W = images.shape
        cls = dir_.name

        if y_cls_do:
            y_cls.append(cls)

        for i, img in enumerate(images):
            img_post = saved_img_processing(img.squeeze())  # (nsamples, H, W)
            
            X_parts.append(img_post)
            Xt_parts.append(i)
            y_int_parts.append(c)
            y_name_parts.append(cls)
    X  = torch.cat(X_parts, dim=0).squeeze()  # (total_samples, H, W)
    Xt = torch.tensor(Xt_parts)  # (total_samples,)
    y_int = torch.tensor(y_int_parts)  # (total_samples,)

    return ImageDatasetView(X=X, Xt=Xt, y_int=y_int, y_names=y_name_parts, y_cls=y_cls)


def d1_move(dummy: None):
    datasets = []
    # clear folder 
    dsfolder = Path(trajectory_data.package_path) / 'trajectories' / 'artificial_dataset'
    assert dsfolder.is_dir()
    shutil.rmtree(dsfolder)

    generate_run(RunArgs(folder="d1_train", cls="peg", offset=(0,0,0), mesh="taskboard.stl"))
    generate_run(RunArgs(folder="d1_test",  cls="peg" , offset=(8,0,0), mesh="taskboard.stl"))
    generate_run(RunArgs(folder="d1_train", cls="nopeg", offset=(0,0,0), mesh="taskboard_nopeg.stl"))
    generate_run(RunArgs(folder="d1_test",  cls="nopeg" , offset=(8,0,0), mesh="taskboard_nopeg.stl"))

    d_train = get_dataset_view_artificial_dataset(folder="d1_train")
    d_test = get_dataset_view_artificial_dataset(folder="d1_test", y_cls=d_train.y_cls)

    datasets.append([d_train, d_test, "Artificial Approach"])

    return datasets
