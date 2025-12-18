import torch
from copy import deepcopy

import trajectory_data
from nocode_robot_programming.state_decision.utils import Filename
from nocode_robot_programming.state_decision_dataset_prepare.dataloader import TrajectoryDataset, ImageDatasetView, saved_img_processing
from nocode_robot_programming.state_decision_dataset_prepare.decision_state_clustering import cluster
from typing import List, Tuple
import numpy as np

def get_auto_dataset_view(datafileloader, file_names: list[str], relevant_parts: list[str], at: slice = slice(None,None), anomaly=False) -> ImageDatasetView:
    """ NO TAGS!    
        Fileloader used to create dataset
        1. Image preprocessing
        2. Mask only a portion of the samples
        3. Returns as a ImageDatasetView object
    """
    X_parts, Xt_parts, y_int_parts, y_name_parts = [], [], [], []

    start = at.start if at.start is not None else -float("inf")
    stop  = at.stop  if at.stop  is not None else  float("inf")

    if anomaly:
        offsets = [Filename(part).offset for part in relevant_parts]
        root_part = relevant_parts[np.argmin(offsets)]
        relevant_parts = [root_part]
    print(f"{relevant_parts=}")

    for file in file_names:
        f = Filename(file)
        idx = datafileloader.files.index(f"{datafileloader.dir}/{file}.npz")
        imgs = saved_img_processing(datafileloader[idx]['img'].squeeze()).squeeze()   # (nsamples, H, W)
        if imgs.ndim == 2: # if nsmaples == 1, squeeze squeezes too hard, we need ndim=3
            imgs = imgs.unsqueeze(0)
        nsamples = imgs.shape[0]
        
        if anomaly:
            if f.part_name == root_part:
                i = 0
                label_name = f.part_name
            else:
                i = 1
                label_name = "anomaly"
        else:
            i = relevant_parts.index(f.part_name)
            label_name = f.part_name
        print(f"{label_name=}")
        xt = torch.arange(f.offset, f.offset + nsamples)  # shape (nsamples,)
        
        mask = (xt >= start) & (xt < stop)                # safe, no OOB
        if mask.any():
            imgs_sub = imgs[mask]
            xt_sub   = xt[mask]

            y_int_sub   = torch.full((imgs_sub.shape[0],), i, dtype=torch.int)
            y_names_sub = [label_name] * imgs_sub.shape[0]

            X_parts.append(imgs_sub)
            Xt_parts.append(xt_sub)
            y_int_parts.append(y_int_sub)
            y_name_parts.extend(y_names_sub)

    if len(X_parts) == 0:
        return None

    X  = torch.cat(X_parts, dim=0)             # (total_samples, H, W)
    Xt = torch.cat(Xt_parts, dim=0)            # (total_samples,)
    y_int = torch.cat(y_int_parts, dim=0)      # (total_samples,)
    y_names = y_name_parts

    return ImageDatasetView(X=X, Xt=Xt, y_int=y_int, y_names=y_names, y_cls=relevant_parts)


# def load_anom(loader, task_name: str, e: int = 10) -> List[Tuple[ImageDatasetView, ImageDatasetView, str]]:
#     return load_eval(loader, task_name, e, anomaly=True)

def load_eval(loader, task_name: str, e: int = 10, anomaly: bool = False) -> List[Tuple[ImageDatasetView, ImageDatasetView, str]]:
    """ Returns list of dataset tuples, each tuple has train dataset, test dataset and text description. """
    decision_states = cluster(loader.tasks[task_name], e)
    print("Decision states: ", decision_states)

    index = loader.tasks[task_name]

    datasets = []
    for ds in decision_states:

        file_names = []
        for name in index['names']:
            f = Filename(name)
            if f.part_name in ds['relevant_parts']:
                file_names.append(name) 
        print("filenames: ", file_names)

        # train/test split based on if it is a demonstration or not
        if anomaly:
            train_file_names, test_file_names = [], []
            for name in file_names:
                f = Filename(name)
                if f.offset == 0:
                    train_file_names.append(name)
                else:
                    test_file_names.append(name)
        else:
            train_file_names, test_file_names = [], []
            for name in file_names:
                f = Filename(name)
                if f.is_demo or f.trial == 0:
                    train_file_names.append(name)
                else:
                    test_file_names.append(name)


        d_train = get_auto_dataset_view(loader, relevant_parts=ds['relevant_parts'], at=slice(ds['start'], ds['end']), file_names=train_file_names, anomaly=anomaly)
        d_test = get_auto_dataset_view(loader, relevant_parts=ds['relevant_parts'], at=slice(ds['start'], ds['end']), file_names=test_file_names, anomaly=anomaly)
        
        
        if d_train is None:
            continue
        
        elif d_test is None:
            d_test = d_train

        f = Filename(ds['relevant_parts'][0])

        datasets.append((d_train, d_test, f"{f.task_userstudy} {f.modality} {f.person}, window={e}, train={len(train_file_names)}, test={len(test_file_names)}"))


    return datasets


def load_deploy(loader, task_name: str, e: int = 10) -> tuple[list[ImageDatasetView], ImageDatasetView]:
    """ Returns list of dataset tuples, each tuple has train dataset, test dataset and text description. 
    
        - Each dataset tuple includes a single DS, exception is the last that contains all images.
    """

    # cluster decision states
    # [{'start': 8, 'end': 18, 'relevant_parts': ['ttst_kin_test', 'ttst_kin_test_branch_from_0_at_8']}, # DS1
    #   ... # DS2
    #   ... # DS3
    # ]
    decision_states = cluster(loader.tasks[task_name], e)
    print("Decision states: ", decision_states)

    index = loader.tasks[task_name]

    datasets = []
    for ds in decision_states:

        file_names = []
        for name in index['names']:
            if Filename(name).part_name in ds['relevant_parts']:
                file_names.append(name) 
        d = get_auto_dataset_view(loader, relevant_parts=ds['relevant_parts'], at=slice(ds['start'], ds['end']), file_names=file_names)
        if d is not None:
            datasets.append(d)

    file_names = loader.tasks[task_name]['names']
    relevant_parts = []
    for name in file_names:
        if Filename(name).is_demo:
            relevant_parts.append(name)

    all_images_dataset = get_auto_dataset_view(loader, relevant_parts=relevant_parts, at=slice(None, None), file_names=file_names)

    return datasets, all_images_dataset

