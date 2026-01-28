

from nocode_robot_programming.state_decision.utils import Filename

from nocode_robot_programming.task_graph.task_graph_pyplot import plot_task_graph as plot_task_graph_pyplot
from nocode_robot_programming.task_graph.task_graph_dashplot import visualize_task_graph as plot_task_graph_dash

class TaskGraph:

    def plot_task_graph(self, task_name: str, branch_window: int = 3):
        # names: "user_0_kine_peg_pick", "user_0_kine_peg_pick_0", "user_0_kine_peg_pick_1", "user_0_kine_peg_pick_2", ...
        filenames = self.names
        assert task_name in filenames, "Not found root skill variant"

        # filter the skills with this name
        plot_branches = []
        for file in filenames:
            # the same root name, e.g., "user_0_kine_peg_pick..."
            if not (task_name in file):
                continue
            f = Filename(file)
            plot_branches.append({"name": file, "start": f.offset, "length": self.get_length(file)})

        plot_task_graph_pyplot(task_name, plot_branches, branch_window, title=f"Task Graph: {task_name}")

        raw_names = self.tasks[task_name]['names']
        length_lookup = {}
        for raw_name in raw_names:
            if "trial" not in raw_name:
                length_lookup[raw_name] = self[raw_name]['length']

        plot_task_graph_dash(raw_names, length_lookup)