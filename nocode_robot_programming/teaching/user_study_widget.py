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
    Modal popup with 2–5 large buttons.
    Blocks until THIS window is closed, then returns the chosen option
    (or None if closed via the window X).

    options : list
        List of payloads (labels or any objects; str() is used for text).
    title   : str
        Window title and heading text.
    master  : tk.Tk or tk.Toplevel or None
        Existing root; if None, an existing default root is reused if present,
        otherwise a hidden root is created temporarily.
    """
    if not options:
        raise ValueError("options must be a non-empty list")

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
    width, height = 600, 300
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

    # Vertical list of big buttons
    for opt in options:
        txt = str(opt)
        btn = tk.Button(
            btn_frame,
            text=txt,
            font=("Arial", 13),
            width=30,          # good for ~30-char labels
            anchor="w",        # left-align text
            command=lambda v=opt: on_click(v),
            padx=10,
            pady=5,
        )
        btn.pack(fill="x", pady=5)

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

    def new_run_log(title: str, _run_counter, log_accordion) -> widgets.Output:
        """
        Create a new collapsible Output panel for a single record/play run.
        Returns the Output widget so you can 'with out:' print into it.
        """
        out = widgets.Output(
            layout=widgets.Layout(
                max_height="600px",   # log area scrolls internally
                overflow="auto",
                border="1px solid #eee",
                padding="4px",
            )
        )

        # Append new Output as a child of the Accordion
        children = list(log_accordion.children)
        children.append(out)
        log_accordion.children = children

        # Set title (e.g., "03: Record p1k_peg_pick")
        idx = len(children) - 1
        log_accordion.set_title(idx, f"{_run_counter:02d}: {title}")

        # Optionally auto-open the newest run and collapse others
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

    config_box = widgets.VBox(
        [
            widgets.HTML("<b>Configuration</b>"),
            person_text,
            modality_toggle,
            task_toggle,
            file_status_label,
            matching_files_label,
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
        if matching:
            matching_sorted = sorted(matching)
            items = "<br>".join(matching_sorted)
            matching_files_label.value = (
                f"<b>Available files matching prefix {full_name}</b><br>{items}"
            )
        else:
            matching_files_label.value = (
                f"<b>No files found matching prefix {full_name}</b>"
            )

    # ------------------------------------------------
    # Button callbacks
    # ------------------------------------------------
    @single_run
    def on_record_clicked(_):
        task_name = build_task_name()

        run_out = new_run_log(f"Record {task_name}", _run_counter, log_accordion)
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
            run_out = new_run_log("Play (no task)", _run_counter, log_accordion)
            with run_out:
                run_out.clear_output()
                print("[play] nothing to play.")
            return

        run_log = new_run_log(f"Play {task_name}", _run_counter, log_accordion)
        with run_log:
            print(f"[training] Started ({task_name})")
            lfd.retrain(task_name)
            print(f"[training] Finished")
            print(f"[play] Playing: {task_name}")
            # lfd.ui_progress_callback = ui_progress_callback
            # lfd.execution_plot_out = execution_plot_out
            lfd.play_skill(task_name, None, localize_box=False)
            lfd.move_template_start()
            print(f"[play] Finished")
            
    @single_run
    def on_taskgraph_clicked(_):
        task_name = build_task_name()
        if task_name is None:
            run_out = new_run_log("Task graph (no task)", _run_counter, log_accordion)
            with run_out:
                run_out.clear_output()
                print("[task graph] nothing to plot.")
            return

        run_log = new_run_log(f"Task graph {task_name}", _run_counter, log_accordion)
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

    execution_plot_out = widgets.Output(
    layout=widgets.Layout(
        border="1px solid #ccc",
        padding="4px",
        max_height="400px",
        overflow="auto",
    )
    )


    # Dashboard layout
    dashboard = widgets.VBox(
        [
            widgets.HTML("<h3>Human Teaching Dashboard</h3>"),
            config_box,
            teaching_box,
            # widgets.HTML("<b>Execution plot (live)</b>"),
            # execution_plot_out,
            widgets.HTML("<b>Log</b>"),
            log_out,
            log_accordion,
        ]
    )

    return dashboard  # last expression: renders inline

# import matplotlib.pyplot as plt
# from IPython.display import clear_output

# # Keep a short rolling history (to avoid plotting tens of thousands of points)
# from collections import deque
# history_len = 300
# exec_history = {
#     "step": deque(maxlen=history_len),
#     "curr_branch": deque(maxlen=history_len),
#     "suggested_branch": deque(maxlen=history_len),
#     "anomaly": deque(maxlen=history_len),
# }

# def branch_to_int(branch_name: str) -> int:
#     # Simple encoding so you can plot categorical branches as lines.
#     # Adjust mapping to your real branch names.
#     # Example: {'nominal':0, 'fallback':1, 'anomaly':2, ...}
#     mapping = getattr(branch_to_int, "_mapping", {})
#     if branch_name not in mapping:
#         mapping[branch_name] = len(mapping)
#         branch_to_int._mapping = mapping
#     return mapping[branch_name]

# def ui_progress_callback(lfd, step, fps, curr_branch, target_state, suggested_branch, anomaly_flag):
#     """
#     Called from inside lfd.play_skill() loop.
#     Updates a small execution plot in execution_plot_out.
#     """
#     exec_history["step"].append(step)
#     exec_history["curr_branch"].append(branch_to_int(curr_branch))
#     exec_history["suggested_branch"].append(branch_to_int(suggested_branch))
#     exec_history["anomaly"].append(1 if anomaly_flag else 0)

#     # For speed, you may want to only update every N steps:
#     if step % 10 != 0:
#         return

#     with lfd.execution_plot_out:
#         clear_output(wait=True)
#         fig, ax1 = plt.subplots()

#         steps = list(exec_history["step"])
#         curr_b = list(exec_history["curr_branch"])
#         sugg_b = list(exec_history["suggested_branch"])
#         anomaly = list(exec_history["anomaly"])

#         ax1.plot(steps, curr_b, label="current branch")
#         ax1.plot(steps, sugg_b, linestyle="--", label="suggested branch")
#         ax1.set_xlabel("step")
#         ax1.set_ylabel("branch id")
#         ax1.legend(loc="upper left")

#         # Optional: show anomaly as vertical markers
#         for s, a in zip(steps, anomaly):
#             if a:
#                 ax1.axvline(s, linestyle=":", alpha=0.5)

#         plt.tight_layout()
#         plt.show()
#         plt.close(fig)