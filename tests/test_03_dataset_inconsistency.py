import pytest
from nocode_robot_programming.state_decision_dataset_prepare.dataloader import TrajectoryDataset

def _make_ds(task_index):
    """Create a TrajectoryDataset instance without running __init__."""
    ds = object.__new__(TrajectoryDataset)
    ds._task_index = task_index
    return ds

def _rec(names):
    """Minimal record; function under test only uses rec['names']."""
    return {"names": names}

@pytest.mark.parametrize(
    "name, expected_substrings",
    [
        (
            "missing trial parent",
            ["[missing trial parent]", "base 'jan_kin_peg_pick'", "Present trials: [0]"],
        ),
        (
            "missing previous trial",
            ["[missing previous trial]", "jan_kin_peg_pick", "Missing: [0]"],
        ),
        (
            "missing original for branch_at",
            ["[missing original for branch]", "branch_at_30", "jan_kin_peg_pick"],
        ),
        (
            "missing original for branch_from_0",
            ["[missing original for branch]", "branch_from_0_at_30", "jan_kin_peg_pick"],
        ),
        (
            "missing branch parent for branch_from_10",
            ["[missing branch parent]", "branch_from_10_at_30", "at 10", "branch_from_<any>_at_10"],
        ),
    ],
)
def test_warn_incomplete_trials_and_branches_flags_expected(name, expected_substrings):
    # Build cases that each trigger exactly one *kind* of warning.
    if name == "missing trial parent":
        task_index = {
            "jan_kin_peg_pick": _rec(["jan_kin_peg_pick_trial_0"])  # base missing
        }

    elif name == "missing previous trial":
        task_index = {
            "jan_kin_peg_pick": _rec(["jan_kin_peg_pick", "jan_kin_peg_pick_trial_1"])
        }

    elif name == "missing original for branch_at":
        task_index = {
            "jan_kin_peg_pick": _rec(["jan_kin_peg_pick_branch_at_30"])
        }

    elif name == "missing original for branch_from_0":
        task_index = {
            "jan_kin_peg_pick": _rec(["jan_kin_peg_pick_branch_from_0_at_30"])
        }

    elif name == "missing branch parent for branch_from_10":
        task_index = {
            "jan_kin_peg_pick": _rec(
                ["jan_kin_peg_pick", "jan_kin_peg_pick_branch_from_10_at_30"]
            )
        }
    else:
        raise RuntimeError("unknown case")

    ds = _make_ds(task_index)
    warnings = ds.warn_incomplete_trials_and_branches(print_warnings=False)
    joined = "\n".join(warnings)

    # Ensure we got at least one warning and it contains the expected information
    assert warnings, "Expected warnings but got none."
    for s in expected_substrings:
        assert s in joined


def test_warn_incomplete_trials_and_branches_no_warnings_when_consistent():
    # Fully consistent example:
    # - original exists
    # - trials contiguous
    # - branch_from_0_at_10 has original
    # - branch_from_10_at_30 has a parent branch ending at 10 (any-from, at=10)
    task_index = {
        "jan_kin_peg_pick": _rec(
            [
                "jan_kin_peg_pick",
                "jan_kin_peg_pick_trial_0",
                "jan_kin_peg_pick_trial_1",
                "jan_kin_peg_pick_branch_from_0_at_10",
                "jan_kin_peg_pick_branch_from_0_at_10_trial_0",
                "jan_kin_peg_pick_branch_from_10_at_30",
                "jan_kin_peg_pick_branch_from_10_at_30_trial_0",
            ]
        )
    }
    ds = _make_ds(task_index)
    warnings = ds.warn_incomplete_trials_and_branches(print_warnings=False)
    assert warnings == []


def test_warn_incomplete_trials_and_branches_branch_trials_require_branch_base():
    # Branch trial exists but branch base missing => missing trial parent (base = branch name)
    task_index = {
        "jan_joy_peg_pick": _rec(
            [
                "jan_joy_peg_pick",
                "jan_joy_peg_pick_branch_at_114_trial_0",  # base missing
            ]
        )
    }
    ds = _make_ds(task_index)
    warnings = ds.warn_incomplete_trials_and_branches(print_warnings=False)
    joined = "\n".join(warnings)
    assert "[missing trial parent]" in joined
    assert "jan_joy_peg_pick_branch_at_114" in joined
