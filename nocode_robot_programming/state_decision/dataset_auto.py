from nocode_robot_programming.state_decision.utils import Filename
from nocode_robot_programming.state_decision.dataloader import ImageDatasetView, saved_img_processing
import torch
from copy import deepcopy
from nocode_robot_programming.state_decision.dataloader import TrajectoryDataset
import trajectory_data

def get_dataset_view(datafileloader, file_names: list[str], tags: list[str] = [], at: slice = slice(35,85)) -> ImageDatasetView:
    """ Fileloader used to create dataset
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
        nsamples = imgs.shape[0]
        
        tag = datafileloader[file]['tag']
        i = tags.index(tag)
        xt = torch.arange(f.offset, f.offset + nsamples)  # shape (nsamples,)
        
        mask = (xt >= start) & (xt < stop)                # safe, no OOB
        if mask.any():
            imgs_sub = imgs[mask]
            xt_sub   = xt[mask]

            y_int_sub   = torch.full((imgs_sub.shape[0],), i, dtype=torch.int)
            y_names_sub = [tag] * imgs_sub.shape[0]

            X_parts.append(imgs_sub)
            Xt_parts.append(xt_sub)
            y_int_parts.append(y_int_sub)
            y_name_parts.extend(y_names_sub)

    X  = torch.cat(X_parts, dim=0)             # (total_samples, H, W)
    Xt = torch.cat(Xt_parts, dim=0)            # (total_samples,)
    y_int = torch.cat(y_int_parts, dim=0)      # (total_samples,)
    y_names = y_name_parts

    return ImageDatasetView(X=X, Xt=Xt, y_int=y_int, y_names=y_names, y_cls=tags)

def auto_load(task_name: str, e: int = 10):
    raise Exception("TODO")
    loader = TrajectoryDataset(trajectory_data.package_path)
    
    file_names = loader.get_all_names(task_name)

    index = loader.tasks['task_name']

    # set of all branches
    tags = 

    branch_offset: int = 49
    tags = ['d1_peg_pick', 'd1_peg_pick_branch_at_49'] # label: 0, 1  

    at = slice(branch_offset-e/2.0, branch_offset+e/2.0)

    return get_dataset_view(loader, tags=tags, at=at,
        file_names=file_names)

