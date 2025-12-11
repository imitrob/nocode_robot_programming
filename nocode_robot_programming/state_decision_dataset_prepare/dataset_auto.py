import torch
from copy import deepcopy

import trajectory_data
from nocode_robot_programming.state_decision.utils import Filename
from nocode_robot_programming.state_decision_dataset_prepare.dataloader import TrajectoryDataset, ImageDatasetView, saved_img_processing
from nocode_robot_programming.state_decision_dataset_prepare.decision_state_clustering import cluster

def get_auto_dataset_view(datafileloader, file_names: list[str], relevant_parts: list[str], at: slice = slice(None,None)) -> ImageDatasetView:
    """ NO TAGS!    
        Fileloader used to create dataset
        1. Image preprocessing
        2. Mask only a portion of the samples
        3. Returns as a ImageDatasetView object
    """
    X_parts, Xt_parts, y_int_parts, y_name_parts = [], [], [], []

    start = at.start if at.start is not None else -float("inf")
    stop  = at.stop  if at.stop  is not None else  float("inf")

    for file in file_names:
        f = Filename(file)
        idx = datafileloader.files.index(f"{datafileloader.dir}/{file}.npz")
        imgs = saved_img_processing(datafileloader[idx]['img'].squeeze()).squeeze()   # (nsamples, H, W)
        if imgs.ndim == 2: # if nsmaples == 1, squeeze squeezes too hard, we need ndim=3
            imgs = imgs.unsqueeze(0)
        nsamples = imgs.shape[0]
        
        i = relevant_parts.index(f.before_trial_suffix)

        xt = torch.arange(f.offset, f.offset + nsamples)  # shape (nsamples,)
        
        mask = (xt >= start) & (xt < stop)                # safe, no OOB
        if mask.any():
            imgs_sub = imgs[mask]
            xt_sub   = xt[mask]

            y_int_sub   = torch.full((imgs_sub.shape[0],), i, dtype=torch.int)
            y_names_sub = [f.before_trial_suffix] * imgs_sub.shape[0]

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

def load_dataset(loader, task_name: str, e: int = 10) -> tuple[list[ImageDatasetView], ImageDatasetView]:
    """Returns two datasets: First selects images at each DS, Second contains all images
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
            if Filename(name).before_trial_suffix in ds['relevant_parts']:
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

