
from typing import List, Sequence, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import os
import re
from nocode_robot_programming.jupyter_plot import jupyter_plot as ipt

def _ensure_arrays(train_2d: Sequence[Sequence[float]],
                   test_2d: Sequence[Sequence[float]],
                   model_names: Sequence[str],
                   task_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    train = np.array(train_2d, dtype=float)
    test = np.array(test_2d, dtype=float)
    models = list(model_names)
    tasks = list(task_names)

    if train.shape != test.shape:
        raise ValueError(f"Train/test shapes differ: {train.shape} vs {test.shape}")
    if train.ndim != 2:
        raise ValueError(f"Expected 2D lists. Got array of ndim={train.ndim}")

    m, n = train.shape
    # Try to align axes with provided names
    if len(models) == m and len(tasks) == n:
        return train, test, models, tasks
    if len(models) == n and len(tasks) == m:
        return train.T, test.T, models, tasks
    if len(models) == m:
        if len(tasks) != n:
            raise ValueError("task_names length must match number of columns in the arrays.")
        return train, test, models, tasks
    if len(tasks) == n:
        if len(models) != m:
            raise ValueError("model_names length must match number of rows in the arrays.")
        return train, test, models, tasks

    raise ValueError(
        f"Could not align names with shapes. "
        f"Array shape is {train.shape}, model_names={len(models)}, task_names={len(tasks)}."
    )


def _safe_dir(dirname: str) -> Path:
    p = Path("auto_fig_generator") / dirname
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def plot_heatmap(matrix: np.ndarray, x_labels: List[str], y_labels: List[str], title: str, save_path: Path, jupyter_plot: bool):
    fig = plt.figure(figsize=(max(6, len(x_labels)*0.8), max(4.5, len(y_labels)*0.6)))
    ax = fig.add_subplot(111)
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Tasks")
    ax.set_ylabel("Models")
    ax.set_title(title)
    # annotate
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()

    if jupyter_plot:
        ipt.save()
    else:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close(fig)


def plot_grouped_bars_per_task(train: np.ndarray, test: np.ndarray, models: List[str],
                               task: str, task_idx: int, save_path: Path, jupyter_plot: bool):
    tr = train[:, task_idx]
    te = test[:, task_idx]
    x = np.arange(len(models))
    width = 0.35

    fig = plt.figure(figsize=(max(6, len(models)*0.8), 4.8))
    ax = fig.add_subplot(111)
    ax.bar(x - width/2, tr, width, label="Train")
    ax.bar(x + width/2, te, width, label="Test")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{task}: Train vs Test by Model")
    ax.legend()
    fig.tight_layout()
    
    if jupyter_plot:
        ipt.save()
    else:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close(fig)


def plot_bars_per_model(values: np.ndarray, x_labels: List[str], model_name: str, title: str, save_path: Path, jupyter_plot: bool):
    x = np.arange(len(x_labels))
    fig = plt.figure(figsize=(max(6, len(x_labels)*0.8), 4.8))
    ax = fig.add_subplot(111)
    
    x = list(x)
    x.append(x[-1]+1)
    mean = values.mean()
    values = list(values)
    values.append(mean)
    x_labels = list(x_labels)
    x_labels.append("mean")

    colors = ["blue"] * len(x_labels)
    colors[-1] = "orange"

    ax.bar(x, values, color=colors)
    fig.text(0.93, 0.9, f'{mean:.0f}%', ha='center', va='bottom')


    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{title} — {model_name}")
    fig.tight_layout()

    if jupyter_plot:
        ipt.save()
    else:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close(fig)


def visualize_accuracies(train_2d: Sequence[Sequence[float]],
                         test_2d: Sequence[Sequence[float]],
                         model_names: Sequence[str],
                         task_names: Sequence[str],
                         out_dir: str = "accuracy_viz",
                         jupyter_plot: bool = True):
    """Create visualizations and a CSV summary.

    Returns
    -------
    dict with keys:
      - 'out_dir': directory path containing figures
      - 'zip_path': path to ZIP with all figures
      - 'csv_path': path to CSV summary
    """
    train, test, models, tasks = _ensure_arrays(train_2d, test_2d, model_names, task_names)
    out_path = _safe_dir(out_dir)

    difficulty_group = []
    modality_group = []
    task_group = []
    diff_mod_group = []
    for task_name in task_names:
        difficulty_group.append( task_name.split(" ")[-1] )
        modality_group.append( task_name.split(" ")[-2] )
        task_group.append( " ".join(task_name.split(" ")[:-2]) )
        diff_mod_group.append( " ".join(task_name.split(" ")[-2:]) )

    difficulty_means = []
    modality_means = []
    task_means = []
    diff_mod_means = []

    a_idxs = None
    b_idxs = None
    c_idxs = None
    d_idxs = None
    for model_values in test_2d:
        a_ = pd.Series(np.array(model_values)).groupby(difficulty_group).mean()
        if a_idxs is not None:
            assert a_.index.to_list() == a_idxs, f"{a_.index.to_list()} != {difficulty_group}"
        a_idxs = a_.index.to_list()

        b_ = pd.Series(np.array(model_values)).groupby(modality_group).mean()
        if b_idxs is not None:    
            assert b_.index.to_list() == b_idxs, f"{b_.index.to_list()} != {modality_group}"
        b_idxs = b_.index.to_list()

        c_ = pd.Series(np.array(model_values)).groupby(task_group).mean()
        if c_idxs is not None:
            assert c_.index.to_list() == c_idxs, f"{c_.index.to_list()} != {task_group}"
        c_idxs = c_.index.to_list()

        d_ = pd.Series(np.array(model_values)).groupby(diff_mod_group).mean()
        if d_idxs is not None:
            assert d_.index.to_list() == d_idxs, f"{d_.index.to_list()} != {diff_mod_group}"
        d_idxs = d_.index.to_list()

        difficulty_means.append(a_.to_list())
        modality_means.append(b_.to_list())
        task_means.append(c_.to_list())
        diff_mod_means.append(d_.to_list())
    
    print("   difficulty_means  ", difficulty_means)
    
    plot_heatmap(np.array(difficulty_means), a_idxs, models, "Test Accuracy (%)", out_path / "heatmap_test_dfcly_group.pdf", jupyter_plot)
    plot_heatmap(np.array(modality_means), b_idxs, models, "Test Accuracy (%)", out_path / "heatmap_test_mdlt_group.pdf", jupyter_plot)
    if jupyter_plot:
        ipt.show()
    else:
        ipt.delete()
    plot_heatmap(np.array(task_means), c_idxs, models, "Test Accuracy (%)", out_path / "heatmap_test_task_group.pdf", jupyter_plot)
    
    plot_heatmap(np.hstack([np.array(task_means), np.array(modality_means)]), np.hstack([np.array(c_idxs), np.array(b_idxs)]), models, "Test Accuracy (%)", out_path / "heatmap_test_mdltandtask_group.pdf", jupyter_plot)
    
    plot_heatmap(np.array(diff_mod_means), d_idxs, models, "Test Accuracy (%)", out_path / "heatmap_test_dfclt_task_group.pdf", jupyter_plot)
    if jupyter_plot:
        ipt.show()
    else:
        ipt.delete()

    # Summary DataFrame

    cols = pd.MultiIndex.from_product([["Train", "Test"], tasks], names=["Split", "Task"])
    data = np.concatenate([train, test], axis=1)  # (n_models, 2*n_tasks)
    df = pd.DataFrame(data, index=models, columns=cols)
    csv_path = out_path / "summary_train_test.csv"

    df.to_csv(csv_path)

    # Heatmaps
    plot_heatmap(train, tasks, models, "Train Accuracy (%)", out_path / "heatmap_train.pdf", jupyter_plot)
    plot_heatmap(test, tasks, models, "Test Accuracy (%)", out_path / "heatmap_test.pdf", jupyter_plot)
    # plot_heatmap(test - train, tasks, models, "Generalization Gap (Test - Train, pp)", out_path / "heatmap_gap.pdf", jupyter_plot)
    if jupyter_plot:
        ipt.show()
    else:
        ipt.delete()

    # Per-task grouped bars
    for j, t in enumerate(tasks):
        plot_grouped_bars_per_task(train, test, models, t, j, out_path / f"grouped_{j:02d}_{_safe_name(t)}.pdf", jupyter_plot)
        if jupyter_plot and j%4==3:
            ipt.show()

    if jupyter_plot:
        ipt.show()
    else:
        ipt.delete()

    # Per-model bars (test across tasks)
    for i, m in enumerate(models):
        plot_bars_per_model(test[i, :], tasks, m, "Test Accuracy by Task", out_path / f"per_model_test_{i:02d}_{_safe_name(m)}.pdf", jupyter_plot)
        if jupyter_plot and i%4==3:
            ipt.show()
        else:
            ipt.delete()


    if jupyter_plot:
        ipt.show()
    else:
        ipt.delete()

        # ZIP
        zip_path = out_path.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for pdf in sorted(out_path.glob("*.pdf")):
                zf.write(pdf, arcname=pdf.name)
        return {"out_dir": str(out_path), "zip_path": str(zip_path), "csv_path": str(csv_path)}




