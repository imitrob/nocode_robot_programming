import ipywidgets as widgets
from IPython.display import display
import asyncio

import tkinter as tk
import time
from nocode_robot_programming.state_decision_dataset_prepare.dataloader import TrajectoryDataset
import trajectory_data
import nocode_robot_programming
import numpy as np

def choose_with_popup(options, title="Choose target", master=None):
    """
    Modal popup with up to 9 large buttons arranged in a 3x3 grid.
    Blocks until THIS window is closed, then returns the chosen option
    (or None if closed via the window X).

    options : list
        List of payloads (labels or any objects; str() is used for text).
    """
    if not options:
        raise ValueError("options must be a non-empty list")

    options = list(options)#[:9]

    owns_root = False

    # Reuse existing root if possible; only create a new hidden root if needed
    if master is None:
        if tk._default_root is None:
            master = tk.Tk()
            master.withdraw()
            owns_root = True
        else:
            master = tk._default_root

    chosen = {"value": None}

    # Create the dialog window
    dialog = tk.Toplevel(master)
    dialog.title(title)

    # Make it fairly large and centered
    width, height = 720, 420
    dialog.update_idletasks()
    sw = dialog.winfo_screenwidth()
    sh = dialog.winfo_screenheight()
    x = (sw - width) // 2
    y = (sh - height) // 3
    dialog.geometry(f"{width}x{height}+{x+1000}+{y}")
    dialog.minsize(width, height)

    # Handle closing via window X
    def on_close():
        dialog.destroy()
    dialog.protocol("WM_DELETE_WINDOW", on_close)

    dialog.configure(padx=20, pady=20)

    # Title label
    title_label = tk.Label(
        dialog,
        text=title,
        font=("Arial", 16, "bold"),
        anchor="center",
        justify="center",
        wraplength=width - 40,
    )
    title_label.pack(pady=(0, 15), fill="x")

    # Frame for buttons
    btn_frame = tk.Frame(dialog)
    btn_frame.pack(fill="both", expand=True)

    def on_click(value):
        chosen["value"] = value
        dialog.destroy()  # ends wait_window

    # Configure 3x3 grid to expand nicely
    for r in range(3):
        btn_frame.grid_rowconfigure(r, weight=1, uniform="row")
    for c in range(3):
        btn_frame.grid_columnconfigure(c, weight=1, uniform="col")

    # Create buttons in a 3x3 grid
    for i, opt in enumerate(options):
        r, c = divmod(i, 3)
        txt = str(opt)
        btn = tk.Button(
            btn_frame,
            text=txt,
            font=("Arial", 13),
            anchor="center",
            justify="center",
            wraplength=200,
            command=lambda v=opt: on_click(v),
            padx=12,
            pady=12,
        )
        btn.grid(row=r, column=c, sticky="nsew", padx=10, pady=10)

    dialog.lift()
    dialog.focus_force()

    # Block until THIS dialog is closed
    master.wait_window(dialog)

    # Clean up only if we created our own hidden root
    if owns_root and master.winfo_exists():
        master.destroy()

    return chosen["value"]


_busy = False

def user_study_widget(lfd):
    def single_run(handler):
        """Ignore clicks while handler is still running; disable button + input controls."""
        def wrapped(btn):
            global _busy
            if _busy:
                return

            _busy = True
            btn.disabled = True

            prev_states = [w.disabled for w in _controls_to_lock]
            for w in _controls_to_lock:
                w.disabled = True

            try:
                handler(btn)
            finally:
                # restore previous disabled states
                for w, st in zip(_controls_to_lock, prev_states):
                    w.disabled = st

                _busy = False
                btn.disabled = False
        return wrapped

    def new_run_log(title: str, log_accordion) -> widgets.Output:
        """
        Create a new collapsible Output panel for a single record/play/show run.
        Returns the Output widget so you can 'with out:' print into it.
        """
        nonlocal _run_counter

        out = widgets.Output(
            layout=widgets.Layout(
                max_height="600px",
                overflow="auto",
                border="1px solid #eee",
                padding="4px",
            )
        )

        children = list(log_accordion.children)
        children.append(out)
        log_accordion.children = children

        idx = len(children) - 1
        log_accordion.set_title(idx, f"{_run_counter:02d}: {title}")
        log_accordion.selected_index = idx

        _run_counter += 1
        return out

    log_accordion = widgets.Accordion(children=[])
    _run_counter = 0  # simple counter (you can fix this if you need real numbering)

    def human_record(task_name: str):
        lfd.home_gripper()
        lfd.move_template_start()
        lfd.traj_rec()
        suc = lfd.save(task_name)
        if suc:
            lfd.show(task_name)
        lfd.move_template_start()
        return suc

    def normalize_person(p: str) -> str:
        """
        Normalize person input so user can type 'p1' or '1'
        and you always get something like 'p1'.
        """
        p = p.strip()
        if not p:
            return "p1"
        return p

    def build_task_name() -> str:
        """
        Build task_name according to current config.
        - If include_modality=True:   p1_kin_peg_pick (depending on modality)
        """
        person_norm = normalize_person(person_text.value)
        np.save(nocode_robot_programming.package_path+"/user_name.npy", person_text.value) # saves the last user name
        task_key = task_toggle.value       # 'test', 'peg_pick', 'probe', 'wrap'
        return f"{person_norm}{modality_toggle.value}_{task_key}"
        
    def list_available_tasks() -> list[str]:
        """
        Return all available task names (without extensions).
        """
        import os, glob
        base_dir = trajectory_data.package_path+"/trajectories"
        paths = glob.glob(os.path.join(base_dir, "*.npz"))
        return [
            os.path.splitext(os.path.basename(p))[0]
            for p in paths
        ]

    def task_file_exists(task_name: str) -> bool:
        """
        True if `task_name` exists among known tasks.
        """
        return task_name in list_available_tasks()

    try:
        person_name_placeholder = str(np.load(nocode_robot_programming.package_path+"/user_name.npy").item()) # tries to load last user name
    except FileNotFoundError:
        person_name_placeholder = "p1"

    person_text = widgets.Text(
        value=person_name_placeholder,
        description="Person:",
        placeholder="e.g. p1",
        style={"description_width": "80px"},
        layout=widgets.Layout(width="200px"),
    )

    modality_toggle = widgets.ToggleButtons(
        options=[
            ("Joystick (joy)", "_joy"),
            ("Kinesthetic (kin)", "_kin"),
            ("Gesture teleop (gst)", "_gst"),
        ],
        value="_kin",
        description="Modality (tag in task name, not checking what modality used):",
        style={"description_width": "80px"},
    )

    task_toggle = widgets.ToggleButtons(
        options=[
            ("Test", "test"),
            ("Peg pick", "peg_pick"),
            ("Probe measure", "probe"),
            ("Cable wrap", "wrap"),
        ],
        value="test",
        description="Task:",
        style={"description_width": "80px"},
    )

    file_status_label = widgets.HTML(
        value="",
        layout=widgets.Layout(margin="4px 0px 0px 0px"),
    )

    matching_files_label = widgets.HTML(
        value="",
        layout=widgets.Layout(
            margin="4px 0px 0px 0px",
            max_height="140px",
            overflow="auto",
        ),
    )

    matching_files_actions = widgets.VBox(
        [],
        layout=widgets.Layout(
            margin="4px 0px 0px 0px",
            max_height="180px",
            overflow="auto",
            border="1px dashed #ddd",
            padding="4px",
        ),
    )


    config_box = widgets.VBox(
        [
            widgets.HTML("<b>Configuration</b>"),
            person_text,
            modality_toggle,
            task_toggle,
            file_status_label,
            matching_files_label,
            matching_files_actions,
        ],
        layout=widgets.Layout(
            border="1px solid #ccc",
            padding="8px",
            margin="4px",
        ),
    )

    log_out = widgets.Output(
        layout=widgets.Layout(border="1px solid #eee", padding="4px")
    )

    btn_final_record = widgets.Button(
        description="● Record teaching",
        tooltip="Run human_record for final configuration (with modality)",
        layout=widgets.Layout(width="250px"),
    )

    btn_play_final = widgets.Button(
        description="▶ Play",
        tooltip="Play recorded final task",
        layout=widgets.Layout(width="250px"),
        disabled=True,   # will be updated based on file existence
    )

    btn_taskgraph_final = widgets.Button(
        description="Task graph",
        tooltip="Generate recorded task",
        layout=widgets.Layout(width="250px"),
    )

    teaching_box = widgets.VBox(
        [
            widgets.HTML(""),
            widgets.HBox(
                [btn_final_record, btn_play_final, btn_taskgraph_final]
            ),
        ],
        layout=widgets.Layout(
            border="1px solid #ccc",
            padding="8px",
            margin="4px",
        ),
    )


    _controls_to_lock = [person_text, modality_toggle, task_toggle]

    def update_task_status():
        """
        Called whenever person/modality/task changes, or after recording/training.
        - Updates 'file exists' label.
        - Enables / disables Play button depending on file existence.
        - Updates 'available files' list matching the base task name.
        """
        full_name = build_task_name()

        # Get list of all tasks from storage
        all_tasks = list_available_tasks()

        # File exists if storage knows it OR we just recorded it as last_final_task
        exists = (full_name in all_tasks)

        # 1) 'File exists' indicator
        if exists:
            file_status_label.value = (
                f"<span style='color:green;'>"
                f"File exists for <b>{full_name}</b>"
                f"</span>"
            )
            btn_final_record.disabled = True
            # Enable Play and set proper label
            btn_play_final.disabled = False
            # btn_play_final.description = f"▶ Play {full_name}"
            btn_taskgraph_final.disabled = False
            # btn_taskgraph_final.description = f"Task graph of {full_name}"

        else:
            file_status_label.value = (
                f"<span style='color:red;'>"
                f"No file for <b>{full_name}</b>"
                f"</span>"
            )
            btn_final_record.disabled = False
            btn_play_final.disabled = True
            # btn_play_final.description = "▶ Play"
            btn_taskgraph_final.disabled = True
            # btn_taskgraph_final.description = f"Task graph"

            
        # 3) list all available files that match the base name (prefix match)
        matching = [t for t in all_tasks if (t.startswith(full_name) and "_trial_" not in t)]
        
        matching_files_actions.children = ()


        if matching:
            matching_sorted = sorted(matching)
            matching_files_label.value = (
                f"<b>Available files matching prefix {full_name}</b>"
                f" <span style='color:#666'>(click Show to visualize)</span>"
            )

            rows = []
            for t in matching_sorted:

                matching = len([t_ for t_ in all_tasks if (t_.startswith(t) and "_trial_" in t_)])

                name_html = widgets.HTML(f"<code>{t}</code> ({matching})")

                btn_show = widgets.Button(
                    description="Show",
                    tooltip=f"lfd.show('{t}')",
                    layout=widgets.Layout(width="90px"),
                )

                # IMPORTANT: bind `t` as a default argument to avoid late-binding bugs
                @single_run
                def _on_show_clicked(_btn, task=t):
                    run_out = new_run_log(f"Show {task}", log_accordion)
                    with run_out:
                        run_out.clear_output()
                        print(f"[show] {task}")
                        lfd.show(task)
                        print(f"[show] done")

                btn_show.on_click(_on_show_clicked)

                rows.append(widgets.HBox([name_html, btn_show], layout=widgets.Layout(justify_content="space-between")))
            matching_files_actions.children = tuple(rows)

        else:
            matching_files_label.value = f"<b>No files found matching prefix {full_name}</b>"


    @single_run
    def on_record_clicked(_):
        task_name = build_task_name()

        run_out = new_run_log(f"Record {task_name}", log_accordion)
        with run_out:
            run_out.clear_output()
            print(f"[teaching] Recording task: {task_name}")
            suc = human_record(task_name)
            if suc:
                print(f"[teaching] Finished")
            else:
                print(f"")
                print(f"[teaching] Finished, No skill is saved!")


        # Newly recorded file now exists; refresh status labels & Play button
        update_task_status()

    @single_run
    def on_play_clicked(_):
        task_name = build_task_name()
        if task_name is None:
            run_out = new_run_log("Play (no task)", log_accordion)
            with run_out:
                run_out.clear_output()
                print("[play] nothing to play.")
            return

        plot_out = widgets.Output(layout=widgets.Layout(border="1px solid #ddd", padding="4px", margin="6px 0 0 0"))

        run_log = new_run_log(f"Play {task_name}", log_accordion)
        with run_log:
            print(f"[training] Started ({task_name})")
            lfd.retrain(task_name)
            print(f"[training] Finished")
            print(f"[play] Playing: {task_name}")
            display(plot_out)
            
            # reset plot state + history for this run
            if hasattr(lfd, "_exec_plot_state"):
                delattr(lfd, "_exec_plot_state")
            for k in exec_history:
                exec_history[k].clear()

            lfd.execution_plot_out = plot_out
            lfd.ui_progress_callback = ui_progress_callback

            lfd.play_skill(task_name, None, localize_box=False)
            lfd.move_template_start()
            print(f"[play] Finished")
            
    @single_run
    def on_taskgraph_clicked(_):
        task_name = build_task_name()
        if task_name is None:
            run_out = new_run_log("Task graph (no task)", log_accordion)
            with run_out:
                run_out.clear_output()
                print("[task graph] nothing to plot.")
            return

        run_log = new_run_log(f"Task graph {task_name}", log_accordion)
        with run_log:
            loader = TrajectoryDataset()
            loader.plot_task_graph(task_name)
            print(f"[task graph] Finished")

    btn_final_record.on_click(on_record_clicked)
    btn_play_final.on_click(on_play_clicked)
    btn_taskgraph_final.on_click(on_taskgraph_clicked)

    def on_config_changed(change):
        update_task_status()

    person_text.observe(on_config_changed, names="value")
    modality_toggle.observe(on_config_changed, names="value")
    task_toggle.observe(on_config_changed, names="value")

    # Initial status
    update_task_status()

    # Dashboard layout
    dashboard = widgets.VBox(
        [
            widgets.HTML("<h3>Human Teaching Dashboard</h3>"),
            config_box,
            teaching_box,
            widgets.HTML("<b>Log</b>"),
            log_out,
            log_accordion,
        ]
    )

    return dashboard  # last expression: renders inline

import io
import numpy as np
import ipywidgets as widgets
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Tuned for 500 typical, up to 4000 rare
UPDATE_EVERY = 10          # redraw every N steps (try 5 if you want smoother)
MAX_DRAW_POINTS = 700      # max points drawn for the line (downsample if more)
JPEG_QUALITY = 5          # lower = faster uglier (5..15 range is good)
FIG_W_PX, FIG_H_PX = 1000, 200
DPI = 200

# Keep a short rolling history (to avoid plotting tens of thousands of points)
exec_history = {
    "gstep": [],         # monotonically increasing step we compute
    "curr_branch": [],   # int id
    "anomaly": [],       # bool
}

def _short_label(s: str, maxlen: int = 40) -> str:
    s = str(s)
    return s if len(s) <= maxlen else (s[: maxlen - 1] + "…")


def _apply_yticks_and_margin(ax):
    mapping = getattr(branch_to_int, "_mapping", {})
    if not mapping:
        return 0
    items = sorted(mapping.items(), key=lambda kv: kv[1])  # (name, id)
    labels = [_short_label(k, 18) for k, _ in items]
    ids = [v for _, v in items]

    ax.set_yticks(ids)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", labelsize=7, pad=2)
    ax.tick_params(axis="x", labelsize=7)

    maxlen = max((len(l) for l in labels), default=0)
    left = min(0.32, 0.10 + 0.010 * maxlen)
    ax.set_position([left, 0.26, 0.98 - left, 0.70])
    return len(items)



def branch_to_int(branch_name: str) -> int:
    # Simple encoding so you can plot categorical branches as lines.
    # Adjust mapping to your real branch names.
    # Example: {'nominal':0, 'fallback':1, 'anomaly':2, ...}
    mapping = getattr(branch_to_int, "_mapping", {})
    if branch_name not in mapping:
        mapping[branch_name] = len(mapping)
        branch_to_int._mapping = mapping
    return mapping[branch_name]

def _update_branch_yticks(ax):
    """Set y ticks/labels from branch_to_int mapping (name -> id)."""
    mapping = getattr(branch_to_int, "_mapping", {})
    if not mapping:
        return 0
    # sort by id
    items = sorted(mapping.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in items]
    ids = [v for _, v in items]
    ax.set_yticks(ids)
    ax.set_yticklabels(labels)
    return len(items)

def init_exec_plot_fast(lfd):
    if hasattr(lfd, "_exec_plot_state"):
        return

    img = widgets.Image(format="jpeg")
    buf = io.BytesIO()

    fig = Figure(figsize=(FIG_W_PX / DPI, FIG_H_PX / DPI), dpi=DPI)
    canvas = FigureCanvas(fig)
    ax = fig.add_axes([0.18, 0.26, 0.80, 0.70])

    # ONE executed path line
    (line_curr,) = ax.plot([], [], lw=1, antialiased=False)

    # anomaly as obvious star event
    anom_scatter = ax.scatter([], [], s=160, marker="*", facecolors="red",
                          edgecolors="black", linewidths=0.7, zorder=5)


    ax.set_xlabel("step")
    ax.set_ylabel("branch")
    ax.grid(True, alpha=0.12)

    last_map_size = _apply_yticks_and_margin(ax)

    with lfd.execution_plot_out:
        display(img)

    lfd._exec_plot_state = dict(
        img=img, buf=buf,
        fig=fig, canvas=canvas, ax=ax,
        line_curr=line_curr,
        anom_scatter=anom_scatter,
        last_map_size=last_map_size,

        # for global-step reconstruction
        step_offset=0,
        prev_raw_step=None,
    )


def ui_progress_callback(lfd, step, fps, curr_branch, target_state, suggested_branch, anomaly_flag):
    init_exec_plot_fast(lfd)
    st = lfd._exec_plot_state

    # Update mapping so y-ticks include any branches that appear
    curr_id = branch_to_int(curr_branch)
    _ = branch_to_int(suggested_branch)  # for tick labels only, not plotted

    # --- global step reconstruction: when step goes backwards, add offset ---
    if st["prev_raw_step"] is None:
        st["prev_raw_step"] = step
    else:
        if step < st["prev_raw_step"]:
            st["step_offset"] += st["prev_raw_step"]  # branch reset happened
        st["prev_raw_step"] = step

    gstep = st["step_offset"] + step

    exec_history["gstep"].append(gstep)
    exec_history["curr_branch"].append(curr_id)
    exec_history["anomaly"].append(bool(anomaly_flag))

    # redraw cadence: always redraw on anomaly so the star shows
    if (not anomaly_flag) and (gstep % UPDATE_EVERY != 0):
        return

    ax = st["ax"]

    # refresh y ticks + margin when mapping grows
    map_size = len(getattr(branch_to_int, "_mapping", {}))
    if map_size != st["last_map_size"]:
        st["last_map_size"] = _apply_yticks_and_margin(ax)

    steps = np.asarray(exec_history["gstep"], dtype=float)
    curr_b = np.asarray(exec_history["curr_branch"], dtype=float)
    anom = np.asarray(exec_history["anomaly"], dtype=bool)

    n = steps.size
    if n == 0:
        return

    # downsample line for speed, but keep full run shown on x-axis
    stride = max(1, n // MAX_DRAW_POINTS)
    st["line_curr"].set_data(steps[::stride], curr_b[::stride])

    # anomaly event markers at (global_step, current_branch_id)
    if anom.any():
        st["anom_scatter"].set_offsets(np.c_[steps[anom], curr_b[anom]])
    else:
        st["anom_scatter"].set_offsets(np.empty((0, 2)))

    # show ALL steps (global)
    ax.set_xlim(float(steps[0]), float(steps[-1]))

    # categorical y-range
    if map_size > 0:
        ax.set_ylim(-0.5, map_size - 0.5)
    else:
        ax.set_ylim(-0.5, float(curr_b.max()) + 0.5)

    # render → JPEG bytes
    st["canvas"].draw()
    w, h = st["canvas"].get_width_height()
    rgb = np.frombuffer(st["canvas"].tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    im = Image.fromarray(rgb, mode="RGB")

    st["buf"].seek(0)
    st["buf"].truncate(0)
    im.save(st["buf"], format="JPEG", quality=JPEG_QUALITY, optimize=False)
    st["img"].value = st["buf"].getvalue()