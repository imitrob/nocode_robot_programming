
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Dict


def plot_task_graph(
    skill_name: str,
    branches: List[Dict[str, int]],
    e: int,
    title: str = "Task Graph with Global e-Step Branching Window",
    ax: plt.Axes | None = None,
):
    """
    Visualize a branching task graph with a global e-step branching window.

    Parameters
    ----------
    skill_name : str
        Name of the original (root) branch.
    branches : list of dict
        Each dict must contain:
          - 'name'  (str): label for the branch (e.g. 'B1')
          - 'start' (int): timestep when the branch starts
          - 'length'(int): number of timesteps for this branch
        (Optional extra keys are ignored.)
    e : int
        Size of the branching window. For a branch starting at time t, we:
          - Shade [t, t+e] as a global "branching possible" band.
          - Draw arrows from every branch whose active interval intersects [t-e, t+e].
    title : str
        Plot title.
    ax : matplotlib.axes.Axes | None
        Existing axes to draw on. If None, a new figure/axes is created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the visualization.
    """
    # Collect all branches, including original
    all_branches = [
        {"name": str(b["name"]), "start": int(b["start"]), "length": int(b["length"])}
        for b in branches
    ]

    # Stable/top-down ordering: by start time then name
    sorted_branches = sorted(all_branches, key=lambda b: (b["start"], b["name"]))
    for idx, b in enumerate(sorted_branches):
        b["y"] = len(sorted_branches) - idx  # highest y for earliest start

    def active_interval(b):
        """Return (start, end) for a branch."""
        return b["start"], b["start"] + b["length"]

    x_max = max(b["start"] + b["length"] for b in sorted_branches)
    y_min, y_max = 0, len(sorted_branches) + 1

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        created_fig = True

    # Draw branch segments and labels
    for b in sorted_branches:
        s, end = active_interval(b)
        y = b["y"]
        ax.plot([s, end], [y, y], linewidth=3)
        ax.plot(s, y, marker="o")
        ax.plot(end, y, marker="|")
        ax.text(end + 0.3, y, f'{b["name"]}  (len={b["length"]})', va="center")

    # Shade global branching windows [t, t+e] for each non-original branch start
    band_y0, band_y1 = (y_min + 0.25, y_max - 0.25)
    for b in sorted_branches:
        if b["name"] == "Original":
            continue
        t = b["start"]
        band_start, band_end = t, t + e
        rect = Rectangle(
            (band_start, band_y0),
            band_end - band_start,
            band_y1 - band_y0,
            alpha=0.12,
            linewidth=0,
        )
        ax.add_patch(rect)
        ax.text(
            (band_start + band_end) / 2.0,
            y_max - 0.2,
            f"[t, t+e]\n({t}, {t+e})",
            ha="center",
            va="top",
            fontsize=8,
        )

    # Draw arrows from any branch active in [t-e, t+e] to the new branch
    for b_new in sorted_branches:
        if b_new["name"] == "Original":
            continue
        t_new = b_new["start"]
        window_left, window_right = t_new - e, t_new + e

        for b_prev in sorted_branches:
            if b_prev is b_new:
                continue
            prev_start, prev_end = active_interval(b_prev)

            # Intersect([prev_start, prev_end], [t_new - e, t_new + e]) non-empty?
            overlap_left = max(prev_start, window_left)
            overlap_right = min(prev_end, window_right)
            if overlap_left <= overlap_right:
                # Tail near t_new, clamped to predecessor's active interval
                x_tail = max(prev_start, min(t_new, prev_end))
                y_tail = b_prev["y"]
                x_head, y_head = t_new, b_new["y"]
                if x_tail == x_head:
                    x_tail -= 0.2  # slight offset to avoid perfectly vertical arrow

                ax.annotate(
                    "",
                    xy=(x_head, y_head),
                    xytext=(x_tail, y_tail),
                    arrowprops=dict(arrowstyle="->", shrinkA=4, shrinkB=4, lw=1),
                )

    # Axes cosmetics
    ax.set_title(title)
    ax.set_xlabel("Time (timesteps)")
    ax.set_ylabel("Branches")
    ax.set_xlim(-1, x_max + 5)
    ax.set_ylim(y_min, y_max + 0.5)
    ax.set_yticks([b["y"] for b in sorted_branches])
    ax.set_yticklabels([b["name"] for b in sorted_branches])
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    if created_fig:
        plt.tight_layout()

    return ax

if __name__ == "__main__":
    # --- Demo usage (replace with your data) ---
    # Example data:
    example_branches = [
        {"name": "OG", "start": 0,  "length": 30},
        {"name": "B1", "start": 8,  "length": 15},
        {"name": "B2", "start": 12, "length": 10},
        {"name": "B3", "start": 20, "length": 8},
        {"name": "B4", "start": 24, "length": 12},
    ]
    e = 3

    # Plot
    plot_task_graph("OG", example_branches, e, title="Task Graph (Demo)")

    plt.show()
    # Show the image in the notebook output and also confirm saved path
    # from IPython.display import Image, display
    # display(Image("/mnt/data/task_graph.png"))
