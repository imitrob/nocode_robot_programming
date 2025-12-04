import ipywidgets as widgets
from IPython.display import display
import asyncio

import tkinter as tk

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
    dialog.geometry(f"{width}x{height}+{x}+{y}")
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




def user_study_widget(lfd):

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

    # ------------------------------------------------
    # 1) Collapsible log manager (put this in a setup cell)
    # ------------------------------------------------
    log_accordion = widgets.Accordion(children=[])
    # display(log_accordion)   # or insert into your main dashboard layout instead

    _run_counter = 0  # simple global counter

    def human_record(task_name: str):
        lfd.home_gripper(); lfd.move_template_start()
        lfd.traj_rec()
        lfd.save(task_name)
        lfd.show(task_name)
        lfd.move_template_start()
    
    def normalize_person(p: str) -> str:
        """
        Normalize person input so user can type 'p1' or '1'
        and you always get something like 'p1'.
        """
        p = p.strip()
        if not p:
            return "p1"
        return p if p.startswith("p") else f"p{p}"


    def build_task_name(include_modality: bool) -> str:
        """
        Build task_name according to current config.
        - If include_modality=False:  p1_test
        - If include_modality=True:   p1k_peg_pick
        """
        person_norm = normalize_person(person_text.value)
        task_key = task_toggle.value       # 'test', 'peg_pick', 'probe', 'wrap'
        if include_modality:
            return f"{person_norm}{modality_toggle.value}_{task_key}"
        else:
            return f"{person_norm}_{task_key}"


    state = {
        "last_test_task": None,
        "last_final_task": None,
    }

    person_text = widgets.Text(
        value="p1",
        description="Person:",
        placeholder="e.g. p1",
        style={"description_width": "80px"},
        layout=widgets.Layout(width="200px"),
    )

    modality_toggle = widgets.ToggleButtons(
        options=[
            ("Joystick (j)", "j"),
            ("Kinesthetic (k)", "k"),
            ("Gesture teleop (g)", "g"),
        ],
        value="k",
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

    config_box = widgets.VBox(
        [
            widgets.HTML("<b>Configuration</b>"),
            person_text,
            modality_toggle,
            task_toggle,
        ],
        layout=widgets.Layout(
            border="1px solid #ccc",
            padding="8px",
            margin="4px",
        ),
    )

    log_out = widgets.Output(layout=widgets.Layout(border="1px solid #eee", padding="4px"))

    btn_final_record = widgets.Button(
        description="● Record teaching",
        tooltip="Run human_record for final configuration (with modality)",
        layout=widgets.Layout(width="250px"),
    )

    btn_retrain = widgets.Button(
        description="Retrain model",
        tooltip="Runs training procedure on all collected skill parts and trial recordings.",
        layout=widgets.Layout(width="250px"),
        disabled=True,
    )

    btn_play_final = widgets.Button(
        description="▶ Play last",
        tooltip="Play last recorded final task",
        layout=widgets.Layout(width="250px"),
        disabled=True,
    )

    def on_final_record_clicked(_):
        task_name = build_task_name(include_modality=True)   # e.g. p1k_peg_pick
        state["last_final_task"] = task_name

        log_out = new_run_log(f"Record {task_name}", _run_counter, log_accordion)
        with log_out:
            log_out.clear_output()
            print(f"[teaching] Recording task: {task_name}")
            human_record(task_name)

        # Enable play button for this recording
        btn_retrain.description = f"Retrain on {task_name}"
        btn_retrain.disabled = False

    def on_retrain_clicked(_):
        task_name = state["last_final_task"]

        log_out = new_run_log(f"Retraining {task_name}", _run_counter, log_accordion)
        with log_out:
            log_out.clear_output()
            print(f"[retraining] Recording task: {task_name}")
            lfd.retrain(task_name)

        # Enable play button for this recording
        btn_play_final.description = f"▶ Play {task_name}"
        btn_play_final.disabled = False


    async def on_play_final_clicked(_):
        task_name = state["last_final_task"]
        if task_name is None:
            return
        run_log = new_run_log(f"Play {task_name}", _run_counter, log_accordion)
        with run_log:
            print(f"[PLAY] Playing: {task_name}")
            lfd.play_skill(task_name, None, localize_box=False)

    btn_final_record.on_click(on_final_record_clicked)
    btn_retrain.on_click(on_retrain_clicked)
    btn_play_final.on_click(on_play_final_clicked)

    teaching_box = widgets.VBox(
        [
            widgets.HTML(""),
            widgets.HBox([btn_final_record, btn_retrain, btn_play_final]),
        ],
        layout=widgets.Layout(
            border="1px solid #ccc",
            padding="8px",
            margin="4px",
        ),
    )


    def on_load_name_clicked(_):
        task_name = build_task_name(include_modality=True)   # e.g. p1k_peg_pick
        playtask_text.value = task_name

    def on_play_name_clicked(_):
        
        task_name = playtask_text.value
        state["last_final_task"] = task_name
        
        log_out = new_run_log(f"Play {task_name}", _run_counter, log_accordion)
        with log_out:
            log_out.clear_output()
            print(f"[PLAY] Playing: {task_name}")
            lfd.play_skill(task_name, None, localize_box=False)


    def on_retrain_name_clicked(_):
        
        task_name = playtask_text.value
        state["last_final_task"] = task_name
        
        log_out = new_run_log(f"Retrain {task_name}", _run_counter, log_accordion)
        with log_out:
            log_out.clear_output()
            print(f"[Retraining]: {task_name}")
            lfd.retrain(task_name)



    btn_load_name = widgets.Button(
        description="Play custom",
        tooltip="Run task based on name on the left",
        layout=widgets.Layout(width="250px"),
    )
    btn_load_name.on_click(on_load_name_clicked)

    playtask_text = widgets.Text(
        value="p0k_peg_pick",
        description="Task name:",
        placeholder="e.g. p0k_peg_pick",
        style={"description_width": "80px"},
        layout=widgets.Layout(width="200px"),
    )
    

    btn_play_task = widgets.Button(
        description="Play custom",
        tooltip="Run task based on name on the left",
        layout=widgets.Layout(width="250px"),
    )
    btn_retrain_task = widgets.Button(
        description="Retrain custom",
        tooltip="Retrain task based on name on the left",
        layout=widgets.Layout(width="250px"),
    )

    btn_play_task.on_click(on_play_name_clicked)
    btn_retrain_task.on_click(on_retrain_name_clicked)

    play_box = widgets.VBox(
        [
            widgets.HTML(""),
            widgets.HBox([btn_load_name, playtask_text, btn_retrain_task, btn_play_task]),
        ],
        layout=widgets.Layout(
            border="1px solid #ccc",
            padding="8px",
            margin="4px",
        ),
    )

    dashboard = widgets.VBox(
        [
            widgets.HTML("<h3>Human Teaching Dashboard</h3>"),
            config_box,
            teaching_box,
            play_box, 
            widgets.HTML("<b>Log</b>"),
            log_out,
            log_accordion,
        ]
    )

    return dashboard  # last expression: renders inline
