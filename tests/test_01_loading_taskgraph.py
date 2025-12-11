from nocode_robot_programming.state_decision.utils import Filename

import pytest

def test_root_demo_without_extension():
    f = Filename("p0_peg_pick")
    assert f.filename == "p0_peg_pick.npz"
    assert f.name == "p0_peg_pick"
    assert f.task == "p0_peg_pick"
    assert f.trial == -1
    assert f.offset == 0
    assert f.parent_offset == 0
    assert f.is_demo is True
    assert f.to_str() == "p0_peg_pick"

def test_root_demo_with_extension():
    f = Filename("p0_peg_pick.npz")
    assert f.filename == "p0_peg_pick.npz"
    assert f.name == "p0_peg_pick"
    assert f.task == "p0_peg_pick"
    assert f.trial == -1
    assert f.offset == 0
    assert f.parent_offset == 0
    assert f.is_demo is True
    assert f.to_str() == "p0_peg_pick"


@pytest.mark.parametrize("trial_id", [0, 1, 3, 4, 5])
def test_trial_execution(trial_id: int):
    fname = f"p0_peg_pick_trial_{trial_id}"
    f = Filename(fname)

    assert f.filename == fname + ".npz"
    assert f.name == fname
    assert f.task == "p0_peg_pick"
    assert f.trial == trial_id
    assert f.offset == 0        # no branch, so still 0
    assert f.parent_offset == 0
    assert f.is_demo is False
    assert f.to_str() == fname

def test_trial_execution_with_extension():
    f = Filename("p0_peg_pick_trial_2.npz")

    assert f.filename == "p0_peg_pick_trial_2.npz"
    assert f.name == "p0_peg_pick_trial_2"
    assert f.task == "p0_peg_pick"
    assert f.trial == 2
    assert f.offset == 0
    assert f.parent_offset == 0
    assert f.is_demo is False
    assert f.to_str() == "p0_peg_pick_trial_2"

def test_legacy_branch_at():
    f = Filename("p0_peg_pick_branch_at_39")

    assert f.filename == "p0_peg_pick_branch_at_39.npz"
    assert f.name == "p0_peg_pick_branch_at_39"
    assert f.task == "p0_peg_pick"
    assert f.trial == -1
    assert f.offset == 39
    # Old format: treat as branch from root demonstration
    assert f.parent_offset == 0
    assert f.is_demo is True  # still a demonstration (no trial)
    assert f.to_str() == "p0_peg_pick_branch_from_0_at_39"

def test_legacy_branch_at_with_trial():
    f = Filename("p0_peg_pick_branch_at_39_trial_2")

    assert f.filename == "p0_peg_pick_branch_at_39_trial_2.npz"
    assert f.name == "p0_peg_pick_branch_at_39_trial_2"
    assert f.task == "p0_peg_pick"
    assert f.trial == 2
    assert f.offset == 39
    assert f.parent_offset == 0
    assert f.is_demo is False
    assert f.to_str() == "p0_peg_pick_branch_from_0_at_39_trial_2"

def test_new_branch_from_nonzero_parent():
    f = Filename("p0_peg_pick_branch_from_29_at_158")

    assert f.filename == "p0_peg_pick_branch_from_29_at_158.npz"
    assert f.name == "p0_peg_pick_branch_from_29_at_158"
    assert f.task == "p0_peg_pick"
    assert f.trial == -1
    assert f.parent_offset == 29
    assert f.offset == 158
    assert f.is_demo is True
    assert f.to_str() == "p0_peg_pick_branch_from_29_at_158"

def test_new_branch_from_zero_parent():
    f = Filename("p0_peg_pick_branch_from_0_at_158")

    assert f.filename == "p0_peg_pick_branch_from_0_at_158.npz"
    assert f.name == "p0_peg_pick_branch_from_0_at_158"
    assert f.task == "p0_peg_pick"
    assert f.trial == -1
    assert f.parent_offset == 0
    assert f.offset == 158
    assert f.is_demo is True
    assert f.to_str() == "p0_peg_pick_branch_from_0_at_158"

def test_new_branch_from_with_trial_and_extension():
    f = Filename("p0_peg_pick_branch_from_29_at_158_trial_3.npz")

    assert f.filename == "p0_peg_pick_branch_from_29_at_158_trial_3.npz"
    assert f.name == "p0_peg_pick_branch_from_29_at_158_trial_3"
    assert f.task == "p0_peg_pick"
    assert f.trial == 3
    assert f.parent_offset == 29
    assert f.offset == 158
    assert f.is_demo is False
    assert f.to_str() == "p0_peg_pick_branch_from_29_at_158_trial_3"

def test_filename_from_params():
    f = Filename("p0_peg_pick", offset=0, parent_offset=0, trial=-1)
    assert f.to_str() == "p0_peg_pick"

    f = Filename("p0_peg_pick", offset=0, parent_offset=0, trial=10)
    assert f.to_str() == "p0_peg_pick_trial_10"

    f = Filename("p0_peg_pick", offset=10, parent_offset=0, trial=-1)
    assert f.to_str() == "p0_peg_pick_branch_from_0_at_10"

    f = Filename("p0_peg_pick", offset=10, parent_offset=0, trial=10)
    assert f.to_str() == "p0_peg_pick_branch_from_0_at_10_trial_10"

    