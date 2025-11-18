from nocode_robot_programming.state_decision.utils import Filename
from nocode_robot_programming.state_decision.dataloader import ImageDatasetView, saved_img_processing
import torch
from copy import deepcopy

from nocode_robot_programming.state_decision.dataset_D1 import get_dataset_view

def d1_anomaly_peg_pick(
        loader,
        window: int = 10, 
        branch_offset: int = 49, 
        tags = ['d1_peg_pick', ] # label: 0,
    ):
    datasets = []
    at = slice(branch_offset-window/2.0, branch_offset+window/2.0)

    da_train = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d1_peg_pick'])
    da_test = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d1_peg_pick_branch_at_49', 
                    'd1_peg_pick_trial_0',
                    'd1_peg_pick_trial_1',
                    'd1_peg_pick_trial_2',
                    'd1_peg_pick_trial_3',
                    'd1_peg_pick_trial_4',
                    'd1_peg_pick_trial_5',
                    'd1_peg_pick_trial_6',
                    'd1_peg_pick_trial_7',
                    'd1_peg_pick_trial_8'])

    db_train = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d1_peg_pick', 
                    'd1_peg_pick_trial_4'])

    db_test = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d1_peg_pick_trial_2', 
                    'd1_peg_pick_trial_3', 
                    'd1_peg_pick_trial_7', 
                    'd1_peg_pick_trial_8', 
                    'd1_peg_pick_trial_6',
                    'd1_peg_pick_branch_at_49', 
                    'd1_peg_pick_trial_0', 
                    'd1_peg_pick_trial_5', 
                    'd1_peg_pick_trial_1'])

    dc_train = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d1_peg_pick', 
                    'd1_peg_pick_trial_4',
                    'd1_peg_pick_trial_5', 
                    'd1_peg_pick_trial_6'])

    dc_test = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d1_peg_pick_branch_at_49', 
                    'd1_peg_pick_trial_0', 
                    'd1_peg_pick_trial_1',
                    'd1_peg_pick_trial_2',
                    'd1_peg_pick_trial_3',
                    'd1_peg_pick_trial_7',
                    'd1_peg_pick_trial_8'])

    # 1. Smallest train, biggest test    
    datasets.append([da_train, da_test, "Peg Pick anomaly kin hard, 1 train trials, 10 test trials, window=10"])
    # 2. 50% train, 50% test
    datasets.append([db_train, db_test, "Peg Pick anomaly kin medium, 2 train trials, 9 test trials, window=10"])
    # 3. biggest train, smallest test
    datasets.append([dc_train, dc_test, "Peg Pick anomaly kin easy, 4 train trials, 7 test trials, window=10"])

    return datasets

def d2_anomaly_peg_pick(
        loader,
        window: int = 10, 
        branch_offset: int = 76, 
        tags = ['d2_peg_pick', ] # label: 0,
    ):
    datasets = []
    at = slice(branch_offset-window/2.0, branch_offset+window/2.0)

    da_train = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d2_peg_pick',])
    da_test = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d2_peg_pick_trial_0',
                    'd2_peg_pick_trial_1',
                    'd2_peg_pick_trial_3',
                    'd2_peg_pick_trial_4',
                    'd2_peg_pick_branch_at_76', 
                    'd2_peg_pick_trial_2'])

    db_train = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d2_peg_pick', 
                    'd2_peg_pick_trial_0'])

    db_test = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d2_peg_pick_trial_1', 
                    'd2_peg_pick_trial_3',
                    'd2_peg_pick_trial_4',
                    'd2_peg_pick_branch_at_76', 
                    'd2_peg_pick_trial_2', ])

    dc_train = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d2_peg_pick', 
                    'd2_peg_pick_trial_0',
                    'd2_peg_pick_trial_1'])

    dc_test = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d2_peg_pick_trial_3',
                    'd2_peg_pick_trial_4',
                    'd2_peg_pick_branch_at_76', 
                    'd2_peg_pick_trial_2', ])

    # 1. Smallest train, biggest test    
    datasets.append([da_train, da_test, "Peg Pick anomaly joy hard, 1 train trials, 6 test trials, window=10"])
    # 2. 50% train, 50% test
    datasets.append([db_train, db_test, "Peg Pick anomaly joy medium, 2 train trials, 5 test trials, window=10"])
    # 3. biggest train, smallest test
    datasets.append([dc_train, dc_test, "Peg Pick anomaly joy easy, 3 train trials, 4 test trials, window=10"])

    return datasets

def d3_anomaly_peg_pick(        
        loader,
        window: int = 10, 
        branch_offset: int = 189, 
        tags = ['d3_peg_pick', ] # label: 0,
    ):
    datasets = []
    at = slice(branch_offset-window/2.0, branch_offset+window/2.0)

    da_train = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d3_peg_pick', ])
    da_test = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d3_peg_pick_trial_0',
                    'd3_peg_pick_trial_2',
                    'd3_peg_pick_branch_at_189', 
                    'd3_peg_pick_trial_1'])

    db_train = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d3_peg_pick', 
                    'd3_peg_pick_trial_0', ])

    db_test = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d3_peg_pick_trial_2',
                    'd3_peg_pick_branch_at_189', 
                    'd3_peg_pick_trial_1', ])

    dc_train = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d3_peg_pick', 
                    'd3_peg_pick_trial_0', ])

    dc_test = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d3_peg_pick_branch_at_189', 
                    'd3_peg_pick_trial_1', 
                    'd3_peg_pick_trial_2', ])

    # 1. Smallest train, biggest test    
    datasets.append([da_train, da_test, "Peg Pick gest hard, 1 train trials, 4 test trials, window=10"])
    # 2. 50% train, 50% test
    datasets.append([db_train, db_test, "Peg Pick gest easy/medium(a), 2 train trials, 3 test trials, window=10"])
    # 3. biggest train, smallest test
    datasets.append([dc_train, dc_test, "Peg Pick gest easy/medium(b), 2 train trials, 3 test trials, window=10"])

    return datasets

def d1_anomaly_probe(
        loader,
        window: int = 10, 
        branch_offset: int = 51, 
        tags = ['d1_probe', ] # label: 0,   
    ):
    datasets = []
    at = slice(branch_offset-window/2.0, branch_offset+window/2.0)

    da_train = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d1_probe', ])
    da_test = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d1_probe_trial_0',
                    'd1_probe_trial_1',
                    'd1_probe_trial_2',
                    'd1_probe_trial_3',
                    'd1_probe_trial_4',
                    'd1_probe_trial_5',
                    'd1_probe_branch_at_51',
                    'd1_probe_trial_6', ])

    db_train = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d1_probe', 
                    'd1_probe_trial_0', 
                    ])

    db_test = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d1_probe_trial_3', 
                    'd1_probe_trial_4', 
                    'd1_probe_trial_5',
                    'd1_probe_branch_at_51', 
                    'd1_probe_trial_6',
                    'd1_probe_trial_1', 
                    'd1_probe_trial_2', ])

    dc_train = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d1_probe', 
                    'd1_probe_trial_0', 
                    'd1_probe_trial_4', 
                    'd1_probe_trial_5',
                    ])

    dc_test = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d1_probe_trial_3', 
                    'd1_probe_branch_at_51', 
                    'd1_probe_trial_6',
                    'd1_probe_trial_1', 
                    'd1_probe_trial_2', ])

    # 1. Smallest train, biggest test    
    datasets.append([da_train, da_test, "Probe anomaly kin hard, 1 train trials, 8 test trials, window=10"])
    # 2. 50% train, 50% test
    datasets.append([db_train, db_test, "Probe anomaly kin medium, 2 train trials, 7 test trials, window=10"])
    # 3. biggest train, smallest test
    datasets.append([dc_train, dc_test, "Probe anomaly kin easy, 4 train trials, 5 test trials, window=10"])

    return datasets

def d2_anomaly_probe(
        loader,
        window: int = 10, 
        branch_offset: int = 103, 
        tags = ['d2_probe', ] # label: 0,  
    ):
    datasets = []
    at = slice(branch_offset-window/2.0, branch_offset+window/2.0)

    da_train = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d2_probe', ])
    da_test = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d2_probe_trial_0',
                    'd2_probe_trial_1',
                    'd2_probe_trial_2',
                    'd2_probe_trial_3',
                    'd2_probe_trial_4',
                    'd2_probe_trial_5',
                    'd2_probe_trial_6',
                    'd2_probe_branch_at_103',
                    ])

    db_train = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d2_probe', 
                    'd2_probe_trial_0', ])

    db_test = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d2_probe_trial_1', 
                    'd2_probe_trial_2', 
                    'd2_probe_trial_5',
                    'd2_probe_trial_4', 
                    'd2_probe_branch_at_103', 
                    'd2_probe_trial_6',
                    'd2_probe_trial_3', ])

    dc_train = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d2_probe', 
                    'd2_probe_trial_0', 
                    'd2_probe_trial_1', 
                    'd2_probe_trial_2', ])

    dc_test = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d2_probe_trial_5',
                    'd2_probe_trial_4', 
                    'd2_probe_branch_at_103', 
                    'd2_probe_trial_6',
                    'd2_probe_trial_3', ])

    # 1. Smallest train, biggest test    
    datasets.append([da_train, da_test, "Probe anomaly joy hard, 1 train trials, 8 test trials, window=10"])
    # 2. 50% train, 50% test
    datasets.append([db_train, db_test, "Probe anomaly joy medium, 2 train trials, 7 test trials, window=10"])
    # 3. biggest train, smallest test
    datasets.append([dc_train, dc_test, "Probe anomaly joy easy, 4 train trials, 5 test trials, window=10"])

    return datasets


def d3_anomaly_probe(
        loader,
        window: int = 10, 
        branch_offset: int = 118, 
        tags = ['d3_probe', ] # label: 0,
    ):
    datasets = []
    at = slice(branch_offset-window/2.0, branch_offset+window/2.0)

    da_train = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d3_probe', ])
    da_test = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d3_probe_trial_0',
                    'd3_probe_trial_1',
                    'd3_probe_branch_at_118',
                    'd3_probe_trial_2'])

    db_train = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d3_probe', 
                    'd3_probe_trial_0'])

    db_test = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d3_probe_trial_1',
                    'd3_probe_branch_at_118', 
                    'd3_probe_trial_2',])

    dc_train = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d3_probe', 
                    'd3_probe_trial_0'])

    dc_test = get_dataset_view(loader, tags=tags, at=at,
        file_names=['d3_probe_trial_1',
                    'd3_probe_branch_at_118', 
                    'd3_probe_trial_2',])

    # 1. Smallest train, biggest test    
    datasets.append([da_train, da_test, "Probe gest hard, 1 train trials, 4 test trials, window=10"])
    # 2. 50% train, 50% test
    datasets.append([db_train, db_test, "Probe gest easy/medium(a), 2 train trials, 3 test trials, window=10"])
    # 3. biggest train, smallest test
    datasets.append([dc_train, dc_test, "Probe gest easy/medium(b), 2 train trials, 3 test trials, window=10"])

    return datasets