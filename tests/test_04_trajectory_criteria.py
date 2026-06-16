import csv
from types import SimpleNamespace

import numpy as np

from nocode_robot_programming.state_decision_dataset_prepare.trajectory_criteria import (
    TrajectoryCriteriaEditor,
    criteria_status,
    discard,
    filter_trajectory_files,
    include,
    resolve_trajectory_criteria_filenames,
    sync_trajectory_criteria,
)


def _write_npz(path):
    np.savez_compressed(
        path,
        grip=np.zeros((1, 3)),
        traj=np.zeros((3, 1)),
        img=np.zeros((3, 4, 4), dtype=np.uint8),
    )


def _write_rows(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _fake_dataset_builder(names):
    return SimpleNamespace(
        names=names,
        tasks={
            "jan_kin_probe": {
                "names": names,
            }
        },
    )


def test_sync_trajectory_criteria_creates_keep_rows(tmp_path):
    files = [tmp_path / "jan_kin_probe.npz", tmp_path / "jan_kin_probe_trial_0.npz"]
    for file in files:
        _write_npz(file)

    criteria_path = sync_trajectory_criteria(tmp_path / "trajectory_criteria.csv", files)

    with criteria_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert [row["filename"] for row in rows] == ["jan_kin_probe.npz", "jan_kin_probe_trial_0.npz"]
    assert [row["use"] for row in rows] == ["1", "1"]
    assert set(rows[0]) == {"filename", "use", "criterion", "discard_reason"}


def test_sync_trajectory_criteria_preserves_manual_decisions(tmp_path):
    first_file = tmp_path / "jan_kin_probe.npz"
    second_file = tmp_path / "jan_kin_probe_trial_0.npz"
    for file in [first_file, second_file]:
        _write_npz(file)

    criteria_path = sync_trajectory_criteria(tmp_path / "trajectory_criteria.csv", [first_file])
    with criteria_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0]["use"] = "0"
    rows[0]["criterion"] = "manual_quality_check"
    rows[0]["discard_reason"] = "corrupted image sequence"
    _write_rows(criteria_path, rows)

    sync_trajectory_criteria(criteria_path, [first_file, second_file])

    with criteria_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["filename"] for row in rows] == ["jan_kin_probe.npz", "jan_kin_probe_trial_0.npz"]
    assert rows[0]["use"] == "0"
    assert rows[0]["discard_reason"] == "corrupted image sequence"
    assert rows[1]["use"] == "1"


def test_resolve_trajectory_criteria_filenames_supports_task_part_and_trial():
    names = [
        "jan_kin_probe",
        "jan_kin_probe_trial_0",
        "jan_kin_probe_branch_from_0_at_13",
        "jan_kin_probe_branch_from_0_at_13_trial_0",
    ]
    dataset_builder = _fake_dataset_builder(names)

    assert resolve_trajectory_criteria_filenames(dataset_builder, "jan_kin_probe") == [
        "jan_kin_probe.npz",
        "jan_kin_probe_branch_from_0_at_13.npz",
        "jan_kin_probe_branch_from_0_at_13_trial_0.npz",
        "jan_kin_probe_trial_0.npz",
    ]
    assert resolve_trajectory_criteria_filenames(
        dataset_builder,
        "jan_kin_probe_branch_from_0_at_13",
        task="jan_kin_probe",
    ) == [
        "jan_kin_probe_branch_from_0_at_13.npz",
        "jan_kin_probe_branch_from_0_at_13_trial_0.npz",
    ]
    assert resolve_trajectory_criteria_filenames(
        dataset_builder,
        "jan_kin_probe_branch_from_0_at_13_trial_0",
        task="jan_kin_probe",
    ) == ["jan_kin_probe_branch_from_0_at_13_trial_0.npz"]


def test_discard_include_and_status_edit_criteria_csv(tmp_path):
    names = [
        "jan_kin_probe",
        "jan_kin_probe_trial_0",
        "jan_kin_probe_branch_from_0_at_13",
        "jan_kin_probe_branch_from_0_at_13_trial_0",
    ]
    dataset_builder = _fake_dataset_builder(names)
    for name in names:
        _write_npz(tmp_path / f"{name}.npz")
    criteria_path = tmp_path / "trajectory_criteria.csv"

    discarded = discard(
        dataset_builder,
        criteria_path,
        "jan_kin_probe_branch_from_0_at_13",
        reason="bad video",
        task="jan_kin_probe",
        print_report=False,
    )
    assert discarded == [
        "jan_kin_probe_branch_from_0_at_13.npz",
        "jan_kin_probe_branch_from_0_at_13_trial_0.npz",
    ]
    assert criteria_status(
        dataset_builder,
        criteria_path,
        "jan_kin_probe_branch_from_0_at_13",
        task="jan_kin_probe",
    ) == [
        {
            "filename": "jan_kin_probe_branch_from_0_at_13.npz",
            "use": 0,
            "criteria": ["manual_quality_check"],
            "discard_reason": "bad video",
        },
        {
            "filename": "jan_kin_probe_branch_from_0_at_13_trial_0.npz",
            "use": 0,
            "criteria": ["manual_quality_check"],
            "discard_reason": "bad video",
        },
    ]

    included = include(
        dataset_builder,
        criteria_path,
        "jan_kin_probe_branch_from_0_at_13_trial_0",
        task="jan_kin_probe",
        print_report=False,
    )
    assert included == ["jan_kin_probe_branch_from_0_at_13_trial_0.npz"]
    assert criteria_status(
        dataset_builder,
        criteria_path,
        "jan_kin_probe_branch_from_0_at_13_trial_0",
        task="jan_kin_probe",
    )[0]["use"] == 1


def test_trajectory_criteria_editor_uses_dynamic_default_task(tmp_path):
    names = ["jan_kin_probe", "jan_kin_probe_trial_0"]
    dataset_builder = _fake_dataset_builder(names)
    for name in names:
        _write_npz(tmp_path / f"{name}.npz")
    criteria_path = tmp_path / "trajectory_criteria.csv"
    task_name = "jan_kin_probe"
    editor = TrajectoryCriteriaEditor(dataset_builder, criteria_path, default_task=lambda: task_name)

    assert editor.discard(reason="bad task", print_report=False) == [
        "jan_kin_probe.npz",
        "jan_kin_probe_trial_0.npz",
    ]
    assert {row["use"] for row in editor.status()} == {0}


def test_filter_trajectory_files_discards_rows_marked_false(tmp_path):
    keep_file = tmp_path / "jan_kin_probe.npz"
    discard_file = tmp_path / "jan_kin_probe_trial_0.npz"
    for file in [keep_file, discard_file]:
        _write_npz(file)

    criteria_path = tmp_path / "trajectory_criteria.csv"
    sync_trajectory_criteria(criteria_path, [keep_file, discard_file])
    with criteria_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows[1]["use"] = "0"
    rows[1]["criterion"] = "manual_quality_check"
    rows[1]["discard_reason"] = "empty recording"
    _write_rows(criteria_path, rows)

    included, report = filter_trajectory_files([keep_file, discard_file], criteria_path)

    assert included == [str(keep_file)]
    assert [decision.filename for decision in report.discarded] == ["jan_kin_probe_trial_0.npz"]
    assert report.discarded[0].criteria == frozenset({"manual_quality_check"})
    assert report.discarded[0].discard_reason == "empty recording"


def test_trajectory_dataset_applies_criteria_csv(tmp_path):
    from nocode_robot_programming.state_decision_dataset_prepare.dataloader import TrajectoryDataset

    keep_file = tmp_path / "jan_kin_probe.npz"
    discard_file = tmp_path / "jan_kin_probe_trial_0.npz"
    for file in [keep_file, discard_file]:
        _write_npz(file)

    criteria_path = tmp_path / "trajectory_criteria.csv"
    sync_trajectory_criteria(criteria_path, [keep_file, discard_file])
    with criteria_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows[1]["use"] = "0"
    rows[1]["discard_reason"] = "bad video"
    _write_rows(criteria_path, rows)

    dataset = TrajectoryDataset(package_path=str(tmp_path), criteria_csv=criteria_path)

    assert dataset.names == ["jan_kin_probe"]
    assert dataset.criteria_report.included == 1
    assert dataset.criteria_report.discarded[0].discard_reason == "bad video"


def test_trajectory_dataset_syncs_relative_criteria_csv(tmp_path):
    from nocode_robot_programming.state_decision_dataset_prepare.dataloader import TrajectoryDataset

    trajectory_file = tmp_path / "jan_kin_probe.npz"
    _write_npz(trajectory_file)

    dataset = TrajectoryDataset(
        package_path=str(tmp_path),
        criteria_csv="trajectory_criteria.csv",
        sync_criteria_csv=True,
    )

    criteria_path = tmp_path / "trajectory_criteria.csv"
    assert criteria_path.exists()
    assert dataset.criteria_csv == criteria_path
    assert dataset.names == ["jan_kin_probe"]
    assert dataset.criteria_report.included == 1
