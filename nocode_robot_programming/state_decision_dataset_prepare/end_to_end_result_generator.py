from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from nocode_robot_programming.state_decision.utils import Filename

@dataclass
class E2EResults:
    entry_df: pd.DataFrame
    rollout_df: pd.DataFrame
    summary_df: pd.DataFrame
    overall_df: pd.DataFrame
    pivots: dict[str, pd.DataFrame]

def compute_e2e_results(
    per_rollout_results: dict[tuple[str, int], dict[str, Any]],
    *,
    task_labels: dict[str, str] | None = None,
    modality_order: list[str] | None = None,
    task_order: list[str] | None = None,
    strict_success_consistency: bool = True,
) -> tuple[E2EResults, str]:
    """
    Computes rollout-level estimated autonomous success.

    Input:
        per_rollout_results[(rollout_name, ds)] = {
            "state_acc": float,
            "is_success": int,
            "e2e_acc": float,
        }

    Preferred rollout-level construction:
        state_acc_joint = product of state_acc over all DS entries of the same rollout
        e2e_acc = is_success * state_acc_joint

    Returns:
        E2EResults with:
            entry_df   : one row per (rollout_name, ds)
            rollout_df : one row per rollout_name
            summary_df : grouped Task x Modality averages
            pivots     : Table-II-style pivot tables
    """

    task_labels = task_labels or {
        "peg pick": "Peg Pick",
        "probe": "Probe measure",
        "wrap": "Cable wrap",
    }

    modality_order = modality_order or ["kin", "joy", "gst"]
    task_order = task_order or ["Peg Pick", "Probe measure", "Cable wrap"]

    entry_rows = []

    for (rollout_name, ds), value in per_rollout_results.items():
        fn = Filename(rollout_name)

        state_acc = float(value["state_acc"])
        is_success = int(value["is_success"])
        e2e_acc_recomputed = state_acc * is_success
        e2e_acc_given = float(value.get("e2e_acc", np.nan))

        task_raw = fn.task_userstudy
        task = task_labels.get(task_raw, task_raw)

        entry_rows.append(
            {
                "rollout_name": rollout_name,
                "ds": int(ds),
                "person": fn.person,
                "modality": fn.modality,
                "task": task,
                "task_raw": task_raw,
                "trial": fn.trial,
                "offset": fn.offset,
                "parent_offset": fn.parent_offset,
                "is_branch": fn.offset != 0,
                "state_acc": state_acc,
                "is_success": is_success,
                "e2e_acc_recomputed": e2e_acc_recomputed,
                "e2e_acc_given": e2e_acc_given,
            }
        )

    entry_df = pd.DataFrame(entry_rows)

    if entry_df.empty:
        empty = pd.DataFrame()
        return E2EResults(
            entry_df=empty,
            rollout_df=empty,
            summary_df=empty,
            pivots={},
            overall_df=empty,
        ), ""

    entry_df = entry_df.sort_values(
        ["modality", "task", "person", "rollout_name", "ds"]
    ).reset_index(drop=True)

    rollout_rows = []

    for rollout_name, group in entry_df.groupby("rollout_name", sort=True):
        group = group.sort_values("ds")
        first = group.iloc[0]

        success_values = group["is_success"].unique()

        if len(success_values) != 1:
            msg = (
                f"Inconsistent is_success values for rollout '{rollout_name}': "
                f"{success_values.tolist()}"
            )
            if strict_success_consistency:
                raise ValueError(msg)
            replay_success = int(group["is_success"].min())
        else:
            replay_success = int(success_values[0])

        state_acc_joint = float(np.prod(group["state_acc"].to_numpy()))
        e2e_acc = replay_success * state_acc_joint

        rollout_rows.append(
            {
                "rollout_name": rollout_name,
                "person": first["person"],
                "modality": first["modality"],
                "task": first["task"],
                "task_raw": first["task_raw"],
                "trial": first["trial"],
                "is_branch": bool(first["is_branch"]),
                "num_ds": len(group),
                "ds_list": tuple(group["ds"].tolist()),
                "is_success": replay_success,
                "state_acc_joint": state_acc_joint,
                "e2e_acc": e2e_acc,
            }
        )

    rollout_df = pd.DataFrame(rollout_rows).sort_values(
        ["modality", "task", "person", "rollout_name"]
    ).reset_index(drop=True)

    summary_df = (
        rollout_df.groupby(["modality", "task"], as_index=False)
        .agg(
            n_rollouts=("rollout_name", "count"),
            replay_success=("is_success", "mean"),
            switcher_joint_acc=("state_acc_joint", "mean"),
            estimated_autonomous_success=("e2e_acc", "mean"),
            mean_num_ds=("num_ds", "mean"),
        )
    )

    overall_df = pd.DataFrame(
        [
            {
                "n_rollouts": len(rollout_df),
                "replay_success": rollout_df["is_success"].mean(),
                "switcher_joint_acc": rollout_df["state_acc_joint"].mean(),
                "estimated_autonomous_success": rollout_df["e2e_acc"].mean(),
                "mean_num_ds": rollout_df["num_ds"].mean(),
            }
        ]
    )

    for col in [
        "replay_success",
        "switcher_joint_acc",
        "estimated_autonomous_success",
    ]:
        overall_df[f"{col}_pct"] = 100.0 * overall_df[col]

    for col in [
        "replay_success",
        "switcher_joint_acc",
        "estimated_autonomous_success",
    ]:
        summary_df[f"{col}_pct"] = 100.0 * summary_df[col]

    summary_df["modality"] = pd.Categorical(
        summary_df["modality"],
        categories=modality_order,
        ordered=True,
    )
    summary_df["task"] = pd.Categorical(
        summary_df["task"],
        categories=task_order,
        ordered=True,
    )

    summary_df = summary_df.sort_values(["modality", "task"]).reset_index(drop=True)

    pivots = {}

    for metric in [
        "replay_success_pct",
        "switcher_joint_acc_pct",
        "estimated_autonomous_success_pct",
        "n_rollouts",
        "mean_num_ds",
    ]:
        pivots[metric] = (
            summary_df.pivot(index="modality", columns="task", values=metric)
            .reindex(index=modality_order, columns=task_order)
        )

    def make_table2_success_rows(results, decimals=1):
        modalities = ["kin", "joy", "gst"]
        tasks = ["Peg Pick", "Probe measure", "Cable wrap"]

        replay = results.pivots["replay_success_pct"].loc[modalities, tasks]
        auto = results.pivots["estimated_autonomous_success_pct"].loc[modalities, tasks]

        replay_r = replay.round(decimals)
        auto_r = auto.round(decimals)

        # Bold best estimated autonomous success per task, after rounding.
        best_auto = auto_r.max(axis=0)

        lines = []
        for modality in modalities:
            cells = []
            for task in tasks:
                replay_txt = f"{replay_r.loc[modality, task]:.{decimals}f}"
                auto_txt = f"{auto_r.loc[modality, task]:.{decimals}f}"

                if auto_r.loc[modality, task] == best_auto.loc[task]:
                    auto_txt = rf"\textbf{{{auto_txt}}}"

                cells.append(f"{replay_txt} ({auto_txt})")

            line = (
                rf"\texttt{{{modality}}} & "
                + " & ".join(cells)
                + r" \\"
            )
            lines.append(line)

        return "\n".join(lines)

    ret = E2EResults(
        entry_df=entry_df,
        rollout_df=rollout_df,
        summary_df=summary_df,
        overall_df=overall_df,
        pivots=pivots,
    )
    return ret, make_table2_success_rows(ret)