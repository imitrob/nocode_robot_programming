from nocode_robot_programming.state_decision.utils import Filename
from nocode_robot_programming.state_decision.dataloader import ImageDatasetView, saved_img_processing
import torch
from copy import deepcopy

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

def dupl(dataset, n=5):
    dataset_dupl = deepcopy(dataset)
    dataset_dupl.X = torch.vstack([dataset.X] * n)
    dataset_dupl.y_int = torch.tile(dataset.y_int, dims=(n,))
    dataset_dupl.y_names = [*dataset.y_names] * n
    return dataset_dupl

def d1_peg_pick(
        loader,
        window: int = 10, 
        branch_offset: int = 49, 
        tags = ['d1_peg_pick', 'd1_peg_pick_branch_at_49'] # label: 0, 1  
    ):
    datasets = []
    at = slice(branch_offset-window/2.0, branch_offset+window/2.0)

    d_train = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d1_peg_pick', 
                    'd1_peg_pick_branch_at_49', 
                    'd1_peg_pick_trial_3'])
    d_test = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d1_peg_pick_trial_0',
                    'd1_peg_pick_trial_1',
                    'd1_peg_pick_trial_2',
                    'd1_peg_pick_trial_4',
                    'd1_peg_pick_trial_5',
                    'd1_peg_pick_trial_6',
                    'd1_peg_pick_trial_7',
                    'd1_peg_pick_trial_8'])

    d2_train = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d1_peg_pick', 
                    'd1_peg_pick_branch_at_49', 
                    'd1_peg_pick_trial_0', 
                    'd1_peg_pick_trial_4', 
                    'd1_peg_pick_trial_5', 
                    'd1_peg_pick_trial_1'])

    d2_test = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d1_peg_pick_trial_2', 
                    'd1_peg_pick_trial_3', 
                    'd1_peg_pick_trial_7', 
                    'd1_peg_pick_trial_8', 
                    'd1_peg_pick_trial_6'])

    # 1. Smallest train, biggest test    
    datasets.append([d_train, d_test, "Peg Pick dataset, 2 train trials, 8 test trials, window=10"])
    # 2. 50% train, 50% test
    datasets.append([d2_train, d2_test, "Peg Pick dataset, 5 train trials, 5 test trials, window=10"])
    # 3. biggest train, smallest test
    datasets.append([d_test, d_train, "Peg Pick dataset, 8 train trials, 2 test trials, window=10"])

    return datasets

def d2_peg_pick(
        loader,
        window: int = 10, 
        branch_offset: int = 76, 
        tags = ['d2_peg_pick', 'd2_peg_pick_branch_at_76'] # label: 0, 1  
    ):
    datasets = []
    at = slice(branch_offset-window/2.0, branch_offset+window/2.0)

    d_train = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d2_peg_pick', 
                    'd2_peg_pick_branch_at_76', 
                    'd2_peg_pick_trial_2'])
    d_test = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d2_peg_pick_trial_0',
                    'd2_peg_pick_trial_1',
                    'd2_peg_pick_trial_3',
                    'd2_peg_pick_trial_4'])

    d2_train = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d2_peg_pick', 
                    'd2_peg_pick_branch_at_76', 
                    'd2_peg_pick_trial_2', 
                    'd2_peg_pick_trial_0'])

    d2_test = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d2_peg_pick_trial_1', 
                    'd2_peg_pick_trial_3',
                    'd2_peg_pick_trial_4'])

    # 1. Smallest train, biggest test    
    datasets.append([d_train, d_test, "Peg Pick dataset, 2 train trials, 4 test trials, window=10"])
    # 2. 50% train, 50% test
    datasets.append([d2_train, d2_test, "Peg Pick dataset, 3 train trials, 3 test trials, window=10"])
    # 3. biggest train, smallest test
    datasets.append([d_test, d_train, "Peg Pick dataset, 4 train trials, 2 test trials, window=10"])

    return datasets


def d1_probe(
        loader,
        window: int = 10, 
        branch_offset: int = 51, 
        tags = ['d1_probe', 'd1_probe_branch_at_51'] # label: 0, 1  
    ):
    datasets = []
    at = slice(branch_offset-window/2.0, branch_offset+window/2.0)

    d_train = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d1_probe', 
                    'd1_probe_branch_at_51',
                    'd1_probe_trial_6'])
    d_test = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d1_probe_trial_0',
                    'd1_probe_trial_1',
                    'd1_probe_trial_2',
                    'd1_probe_trial_3',
                    'd1_probe_trial_4',
                    'd1_probe_trial_5'])

    d2_train = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d1_probe', 
                    'd1_probe_branch_at_51', 
                    'd1_probe_trial_6',
                    'd1_probe_trial_0', 
                    'd1_probe_trial_1', 
                    'd1_probe_trial_2'])

    d2_test = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d1_probe_trial_3', 
                    'd1_probe_trial_4', 
                    'd1_probe_trial_5'])

    # 1. Smallest train, biggest test    
    datasets.append([d_train, d_test, "Probe dataset, 2 train trials, 6 test trials, window=10"])
    # 2. 50% train, 50% test
    datasets.append([d2_train, d2_test, "Probe dataset, 5 train trials, 3 test trials, window=10"])
    # 3. biggest train, smallest test
    datasets.append([d_test, d_train, "Probe dataset, 6 train trials, 2 test trials, window=10"])

    return datasets

def d2_probe(
        loader,
        window: int = 10, 
        branch_offset: int = 103, 
        tags = ['d2_probe', 'd2_probe_branch_at_103'] # label: 0, 1  
    ):
    datasets = []
    at = slice(branch_offset-window/2.0, branch_offset+window/2.0)

    d_train = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d2_probe', 
                    'd2_probe_branch_at_103',
                    'd2_probe_trial_6'])
    d_test = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d2_probe_trial_0',
                    'd2_probe_trial_1',
                    'd2_probe_trial_2',
                    'd2_probe_trial_3',
                    'd2_probe_trial_4',
                    'd2_probe_trial_5'])

    d2_train = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d2_probe', 
                    'd2_probe_branch_at_103', 
                    'd2_probe_trial_6',
                    'd2_probe_trial_0', 
                    'd2_probe_trial_3'])

    d2_test = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d2_probe_trial_1', 
                    'd2_probe_trial_2', 
                    'd2_probe_trial_5',
                    'd2_probe_trial_4', 
                    ])

    # 1. Smallest train, biggest test    
    datasets.append([d_train, d_test, "Probe dataset, 2 train trials, 6 test trials, window=10"])
    # 2. 50% train, 50% test
    datasets.append([d2_train, d2_test, "Probe dataset, 4 train trials, 4 test trials, window=10"])
    # 3. biggest train, smallest test
    datasets.append([d_test, d_train, "Probe dataset, 6 train trials, 2 test trials, window=10"])

    return datasets