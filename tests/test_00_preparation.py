
def test_addition():
    assert 1 + 1 == 2

def test_import_modules():
    import rclpy
    from skills_manager.risk_aware_lfd.ralfd import RALfD
    from skills_manager.ros_param_manager import set_remote_parameters
    import spatialmath as sm
    from geometry_msgs.msg import PoseStamped

def test_import_modules2():
    import trajectory_data
    import matplotlib.pyplot as plt
    from nocode_robot_programming.state_decision_dataset_prepare.dataloader import TrajectoryDataset
    from nocode_robot_programming.state_decision.dino_model import DINOFeaturePresence
    from nocode_robot_programming.state_decision.SIFT_model import StateDeciderSIFT
    from nocode_robot_programming.state_decision.AEGP_model import AEGP
    from nocode_robot_programming.state_decision.state_decider import StateDeciderBase
    from gesture_detector.utils import pretty_confusion_matrix
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    from trajectory_data.skill_visualizer import play_video
    from nocode_robot_programming.state_decision.utils import add_tag
    from nocode_robot_programming.state_decision_dataset_prepare.dataloader import ImageDatasetView, saved_img_processing

    seed = 50
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    from nocode_robot_programming.state_decision_dataset_prepare.dataset_D1 import d1_peg_pick, dupl
    datafileloader = TrajectoryDataset(trajectory_data.package_path)
    datasets = d1_peg_pick(datafileloader)


def test_d1_dataset_loaded():
    from nocode_robot_programming.state_decision_dataset_prepare.dataloader import TrajectoryDataset
    import trajectory_data
    import torch
    from nocode_robot_programming.state_decision_dataset_prepare.dataset_D1 import d1_peg_pick, dupl
    datafileloader = TrajectoryDataset(trajectory_data.package_path)

    datasets = d1_peg_pick(datafileloader)
    d_train, d_test, d_text = datasets[0] # hard mix
    print(f"{d_text=},\n{d_train.X.shape=},\n{d_train.y_int.shape=},\n{len(d_train.y_names)=},\n{d_train.y_cls=}")


    d_text='Peg Pick d1 hard, 2 train trials, 8 test trials, window=10',
    
    assert d_train.X.shape == torch.Size([21, 224, 224])
    assert d_train.y_int.shape == torch.Size([21])
    assert len(d_train.y_names) == 21
    assert d_train.y_cls == ['d1_peg_pick', 'd1_peg_pick_branch_at_49']