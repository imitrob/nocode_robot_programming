"""Helpers for a manually edited CSV that excludes bad trajectory .npz files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


DEFAULT_TRAJECTORY_CRITERIA_FILENAME = "trajectory_criteria.csv"
TRAJECTORY_CRITERIA_COLUMNS = ("filename", "use", "criterion", "discard_reason")

_TRUE_VALUES = {"1", "true", "yes", "y", "use", "keep", "load", "include"}
_FALSE_VALUES = {"0", "false", "no", "n", "skip", "drop", "discard", "exclude"}


@dataclass(frozen=True)
class TrajectoryCriteriaDecision:
    filename: str
    use: bool
    criterion: str = ""
    discard_reason: str = ""


@dataclass(frozen=True)
class TrajectoryCriteriaReport:
    path: Path
    total_files: int
    included: int
    discarded: tuple[TrajectoryCriteriaDecision, ...]
    missing_rows: tuple[str, ...]
    stale_rows: tuple[str, ...]

    def __str__(self) -> str:
        lines = [
            f"Trajectory criteria: {self.path}",
            f"  included: {self.included}/{self.total_files}",
            f"  discarded: {len(self.discarded)}",
        ]
        if self.missing_rows:
            lines.append(f"  missing CSV rows, included by default: {len(self.missing_rows)}")
        if self.stale_rows:
            lines.append(f"  stale CSV rows without .npz file: {len(self.stale_rows)}")
        if self.discarded:
            lines.append("  discarded files:")
            for decision in self.discarded:
                reason = decision.discard_reason or "(no discard_reason)"
                criterion = f" [{decision.criterion}]" if decision.criterion else ""
                lines.append(f"    - {decision.filename}{criterion}: {reason}")
        return "\n".join(lines)


@dataclass
class TrajectoryCriteriaEditor:
    """Notebook-friendly wrapper around manual trajectory criteria edits."""

    dataset_builder: Any
    criteria_path: str | Path
    default_task: str | Callable[[], str] | None = None

    def target_filenames(self, target: object | None = None, *, task: str | None = None) -> list[str]:
        return resolve_trajectory_criteria_filenames(
            self.dataset_builder,
            target,
            task=task,
            default_task=self.default_task,
        )

    def set(
        self,
        target: object | None = None,
        *,
        use: bool,
        criterion: str = "manual_quality_check",
        discard_reason: str = "",
        task: str | None = None,
        print_report: bool = True,
    ) -> list[str]:
        return set_trajectory_criteria(
            self.dataset_builder,
            self.criteria_path,
            target,
            use=use,
            criterion=criterion,
            discard_reason=discard_reason,
            task=task,
            default_task=self.default_task,
            print_report=print_report,
        )

    def discard(
        self,
        target: object | None = None,
        *,
        reason: str = "corrupted image sequence",
        criterion: str = "manual_quality_check",
        task: str | None = None,
        print_report: bool = True,
    ) -> list[str]:
        return discard(
            self.dataset_builder,
            self.criteria_path,
            target,
            reason=reason,
            criterion=criterion,
            task=task,
            default_task=self.default_task,
            print_report=print_report,
        )

    def include(
        self,
        target: object | None = None,
        *,
        task: str | None = None,
        print_report: bool = True,
    ) -> list[str]:
        return include(
            self.dataset_builder,
            self.criteria_path,
            target,
            task=task,
            default_task=self.default_task,
            print_report=print_report,
        )

    def status(self, target: object | None = None, *, task: str | None = None) -> list[dict[str, object]]:
        return criteria_status(
            self.dataset_builder,
            self.criteria_path,
            target,
            task=task,
            default_task=self.default_task,
        )


def normalize_trajectory_filename(filename: str) -> str:
    name = Path(str(filename).strip()).name
    if not name:
        return ""
    if not name.endswith(".npz"):
        name = f"{name}.npz"
    return name


def _resolve_default_task(default_task: str | Callable[[], str] | None) -> str | None:
    return default_task() if callable(default_task) else default_task


def _task_trajectory_names(dataset_builder: Any, task_name: str) -> list[str]:
    return list(dataset_builder.tasks[task_name]["names"])


def _all_trajectory_files(criteria_path: str | Path) -> list[Path]:
    return sorted(Path(criteria_path).parent.glob("*.npz"))


def resolve_trajectory_criteria_filenames(
    dataset_builder: Any,
    target: object | None = None,
    *,
    task: str | None = None,
    default_task: str | Callable[[], str] | None = None,
) -> list[str]:
    """Resolve a task, skill part, trial name, or list of those to CSV filenames."""
    default_task_name = _resolve_default_task(default_task)
    target = default_task_name if target is None else target
    task = default_task_name if task is None else task

    if target is None:
        raise ValueError("Provide target or default_task before editing trajectory criteria")

    if isinstance(target, (list, tuple, set)):
        filenames = []
        for item in target:
            filenames.extend(
                resolve_trajectory_criteria_filenames(
                    dataset_builder,
                    item,
                    task=task,
                    default_task=default_task,
                )
            )
        return sorted(set(filenames))

    target_name = str(target)
    if target_name in dataset_builder.tasks:
        return sorted(normalize_trajectory_filename(name) for name in _task_trajectory_names(dataset_builder, target_name))

    if task in dataset_builder.tasks:
        search_names = _task_trajectory_names(dataset_builder, task)
    else:
        search_names = list(dataset_builder.names)

    from nocode_robot_programming.state_decision.utils import Filename

    part_matches = [name for name in search_names if Filename(name).part_name == target_name]
    if part_matches:
        return sorted(normalize_trajectory_filename(name) for name in part_matches)

    return [normalize_trajectory_filename(target_name)]


def set_trajectory_criteria(
    dataset_builder: Any,
    criteria_path: str | Path,
    target: object | None = None,
    *,
    use: bool,
    criterion: str = "manual_quality_check",
    discard_reason: str = "",
    task: str | None = None,
    default_task: str | Callable[[], str] | None = None,
    print_report: bool = True,
    trajectory_files: Iterable[str | Path] | None = None,
) -> list[str]:
    """Set use=1/0 for a resolved target in a trajectory criteria CSV."""
    csv_path = Path(criteria_path)
    if trajectory_files is None:
        trajectory_files = _all_trajectory_files(csv_path)
    sync_trajectory_criteria(csv_path, trajectory_files)

    decisions = load_trajectory_criteria(csv_path)
    selected = set(
        resolve_trajectory_criteria_filenames(
            dataset_builder,
            target,
            task=task,
            default_task=default_task,
        )
    )
    missing = sorted(selected - set(decisions))
    if missing:
        raise KeyError(f"Not present in {csv_path}: {missing[:10]}")

    rows = []
    for filename, decision in decisions.items():
        row = {
            "filename": filename,
            "use": "1" if decision.use else "0",
            "criterion": decision.criterion,
            "discard_reason": decision.discard_reason,
        }
        if filename in selected:
            row["use"] = "1" if use else "0"
            row["criterion"] = "" if use else criterion
            row["discard_reason"] = "" if use else discard_reason
        rows.append(row)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRAJECTORY_CRITERIA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    selected_filenames = sorted(selected)
    if print_report:
        action = "Included" if use else "Discarded"
        print(f"{action} {len(selected_filenames)} trajectory row(s) in {csv_path}")
    return selected_filenames


def discard(
    dataset_builder: Any,
    criteria_path: str | Path,
    target: object | None = None,
    *,
    reason: str = "corrupted image sequence",
    criterion: str = "manual_quality_check",
    task: str | None = None,
    default_task: str | Callable[[], str] | None = None,
    print_report: bool = True,
    trajectory_files: Iterable[str | Path] | None = None,
) -> list[str]:
    return set_trajectory_criteria(
        dataset_builder,
        criteria_path,
        target,
        use=False,
        criterion=criterion,
        discard_reason=reason,
        task=task,
        default_task=default_task,
        print_report=print_report,
        trajectory_files=trajectory_files,
    )


def include(
    dataset_builder: Any,
    criteria_path: str | Path,
    target: object | None = None,
    *,
    task: str | None = None,
    default_task: str | Callable[[], str] | None = None,
    print_report: bool = True,
    trajectory_files: Iterable[str | Path] | None = None,
) -> list[str]:
    return set_trajectory_criteria(
        dataset_builder,
        criteria_path,
        target,
        use=True,
        task=task,
        default_task=default_task,
        print_report=print_report,
        trajectory_files=trajectory_files,
    )


def criteria_status(
    dataset_builder: Any,
    criteria_path: str | Path,
    target: object | None = None,
    *,
    task: str | None = None,
    default_task: str | Callable[[], str] | None = None,
    trajectory_files: Iterable[str | Path] | None = None,
) -> list[dict[str, object]]:
    csv_path = Path(criteria_path)
    if trajectory_files is None:
        trajectory_files = _all_trajectory_files(csv_path)
    sync_trajectory_criteria(csv_path, trajectory_files)
    decisions = load_trajectory_criteria(csv_path)
    return [
        {
            "filename": filename,
            "use": int(decisions[filename].use),
            "criterion": decisions[filename].criterion,
            "discard_reason": decisions[filename].discard_reason,
        }
        for filename in resolve_trajectory_criteria_filenames(
            dataset_builder,
            target,
            task=task,
            default_task=default_task,
        )
        if filename in decisions
    ]


def parse_use_value(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    accepted = sorted(_TRUE_VALUES | _FALSE_VALUES)
    raise ValueError(f"Unknown trajectory criteria use value {value!r}. Accepted values: {accepted}")


def load_trajectory_criteria(path: str | Path) -> dict[str, TrajectoryCriteriaDecision]:
    csv_path = Path(path)
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = set(TRAJECTORY_CRITERIA_COLUMNS[:2]) - fieldnames
        if missing_columns:
            raise ValueError(f"{csv_path} is missing required column(s): {sorted(missing_columns)}")

        decisions = {}
        for line_number, row in enumerate(reader, start=2):
            filename = normalize_trajectory_filename(row.get("filename", ""))
            if not filename:
                continue
            if filename in decisions:
                raise ValueError(f"{csv_path}:{line_number} duplicates filename {filename!r}")
            decisions[filename] = TrajectoryCriteriaDecision(
                filename=filename,
                use=parse_use_value(row.get("use", "")),
                criterion=(row.get("criterion") or "").strip(),
                discard_reason=(row.get("discard_reason") or "").strip(),
            )
    return decisions


def sync_trajectory_criteria(path: str | Path, trajectory_files: Iterable[str | Path]) -> Path:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    current_filenames = sorted(
        filename
        for filename in {normalize_trajectory_filename(str(file)) for file in trajectory_files}
        if filename
    )

    existing = load_trajectory_criteria(csv_path) if csv_path.exists() else {}
    stale_filenames = sorted(set(existing) - set(current_filenames))

    rows = []
    for filename in [*current_filenames, *stale_filenames]:
        decision = existing.get(
            filename,
            TrajectoryCriteriaDecision(filename=filename, use=True),
        )
        rows.append(
            {
                "filename": filename,
                "use": "1" if decision.use else "0",
                "criterion": decision.criterion,
                "discard_reason": decision.discard_reason,
            }
        )

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRAJECTORY_CRITERIA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def filter_trajectory_files(
    trajectory_files: Iterable[str | Path],
    criteria_path: str | Path,
    *,
    require_rows: bool = False,
) -> tuple[list[str], TrajectoryCriteriaReport]:
    files = sorted(str(file) for file in trajectory_files)
    filenames = {normalize_trajectory_filename(file) for file in files}
    decisions = load_trajectory_criteria(criteria_path)
    missing_rows = tuple(sorted(filenames - set(decisions)))
    stale_rows = tuple(sorted(set(decisions) - filenames))

    if require_rows and missing_rows:
        raise ValueError(
            f"{criteria_path} is missing rows for {len(missing_rows)} trajectory file(s): "
            + ", ".join(missing_rows[:10])
        )

    included_files = []
    discarded = []
    for file in files:
        filename = normalize_trajectory_filename(file)
        decision = decisions.get(filename)
        if decision is None or decision.use:
            included_files.append(file)
        else:
            discarded.append(decision)

    report = TrajectoryCriteriaReport(
        path=Path(criteria_path),
        total_files=len(files),
        included=len(included_files),
        discarded=tuple(discarded),
        missing_rows=missing_rows,
        stale_rows=stale_rows,
    )
    return included_files, report
