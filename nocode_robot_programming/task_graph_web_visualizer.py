
from nocode_robot_programming.state_decision.utils import Filename

from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional

import dash
from dash import Dash, dcc, html
import plotly.graph_objects as go


def build_task_graph(
    filenames: List[Filename],
    length_lookup: Optional[Dict[str, int]] = None,
):
    """
    Build nodes and edges from a list of Filename objects.

    - Only filenames with trial == -1 (demos/branches) become nodes.
    - Trials are counted per node but not included as nodes.
    - Each node optionally has a 'length' (number of timesteps).
    """

    def demo_key(f: Filename) -> Tuple[str, Optional[int], int]:
        # (task, parent_offset, offset) uniquely identifies a demonstration/branch
        return (f.task, getattr(f, "parent_offset", None), f.offset)

    demos = [f for f in filenames if f.trial == -1]
    trials = [f for f in filenames if f.trial != -1]

    key_to_idx: Dict[Tuple[str, Optional[int], int], int] = {}
    nodes: List[Dict] = []

    # 1) Create nodes
    for f in demos:
        key = demo_key(f)
        if key in key_to_idx:
            continue

        # try to get length from object, then from lookup
        length = getattr(f, "length", None)
        if length is None and length_lookup is not None:
            length = length_lookup.get(f.name)

        idx = len(nodes)
        key_to_idx[key] = idx
        nodes.append(
            {
                "id": idx,
                "task": f.task,
                "offset": f.offset,
                "parent_offset": getattr(f, "parent_offset", None),
                "label": f.name,
                "trials": 0,
                "length": length,  # may be None
            }
        )

    # 2) Count trials per demo node
    for t in trials:
        key = demo_key(t)
        if key in key_to_idx:
            nodes[key_to_idx[key]]["trials"] += 1

    # 3) Build edges (parent -> child) using (task, offset) to find parents
    edges: List[Tuple[int, int]] = []
    offset_to_idx: Dict[Tuple[str, int], int] = {}

    for n in nodes:
        offset_to_idx[(n["task"], n["offset"])] = n["id"]

    for n in nodes:
        parent_offset = n["parent_offset"]
        if parent_offset is None:
            continue  # root demo, no parent
        parent_key = (n["task"], parent_offset)
        parent_idx = offset_to_idx.get(parent_key)
        if parent_idx is not None:
            edges.append((parent_idx, n["id"]))

    # 4) Compute depth per node (for layout / hierarchy)
    children = defaultdict(list)
    for u, v in edges:
        children[u].append(v)

    depths: Dict[int, int] = {n["id"]: 0 for n in nodes}
    roots = [n["id"] for n in nodes if nodes[n["id"]]["parent_offset"] is None]

    for root_id in roots:
        depths[root_id] = 0
        q = deque([root_id])
        while q:
            u = q.popleft()
            for v in children[u]:
                depths[v] = depths[u] + 1
                q.append(v)

    return nodes, edges, depths


def build_topology_figure(
    nodes: List[Dict],
    edges: List[Tuple[int, int]],
    depths: Dict[int, int],
) -> go.Figure:
    """
    Graph 1: Topology view
    - No timestep axis.
    - Nodes float in space in a hierarchical left-to-right layout.
    - Initial sample is on the left center; branches fan out to the right.
    """

    # Assign positions based on depth: x = depth, y spaced within each depth
    depth_to_nodes: Dict[int, List[int]] = defaultdict(list)
    for n in nodes:
        nid = n["id"]
        depth = depths.get(nid, 0)
        depth_to_nodes[depth].append(nid)

    for depth, ids in depth_to_nodes.items():
        count = len(ids)
        for j, nid in enumerate(ids):
            # center them vertically around 0
            y = (j - (count - 1) / 2.0) * 1.5
            nodes[nid]["topo_x"] = depth  # root depth=0 => left side
            nodes[nid]["topo_y"] = y

    # Edges
    edge_x = []
    edge_y = []
    for u, v in edges:
        x0, y0 = nodes[u]["topo_x"], nodes[u]["topo_y"]
        x1, y1 = nodes[v]["topo_x"], nodes[v]["topo_y"]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=2, color="rgba(150,150,150,0.5)"),
        hoverinfo="none",
    )

    # Nodes
    node_x = [n["topo_x"] for n in nodes]
    node_y = [n["topo_y"] for n in nodes]

    hover_text = []
    display_text = []
    sizes = []

    for n in nodes:
        length = n["length"]
        end_ts = n["offset"] + length if length is not None else None
        trials = n["trials"]

        ht = [
            n["label"],
            f"task={n['task']}",
            f"offset={n['offset']}",
            f"parent_offset={n['parent_offset']}",
            f"trials={trials}",
        ]
        if length is not None:
            ht.append(f"length={length}")
            ht.append(f"ends_at={end_ts}")
        hover_text.append("<br>".join(ht))

        if trials > 0:
            display_text.append(f"{n['offset']} (x{trials})")
        else:
            display_text.append(str(n["offset"]))

        sizes.append(10 + 3 * trials)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=display_text,
        textposition="top center",
        hovertext=hover_text,
        hoverinfo="text",
        marker=dict(
            size=sizes,
            color="cornflowerblue",
            line=dict(width=1, color="black"),
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Task Branch Graph – Topology View",
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor="white",
        hovermode="closest",
    )
    return fig


def build_timeline_figure(
    nodes: List[Dict],
    edges: List[Tuple[int, int]],
    depths: Dict[int, int],
) -> go.Figure:
    """
    Graph 2: Timeline view
    - x-axis = timestep (offset).
    - Each demo node sits at its starting offset.
    - Each demo has a horizontal bar from offset to offset + length (if known).
    - Branching shown at a single timestep with a vertical, directed edge (arrow).
    """

    # Give each node a distinct y within its depth band to avoid overlapping bars
    depth_to_nodes: Dict[int, List[int]] = defaultdict(list)
    for n in nodes:
        nid = n["id"]
        depth = depths.get(nid, 0)
        depth_to_nodes[depth].append(nid)

    for depth, ids in depth_to_nodes.items():
        count = len(ids)
        base = -depth * 3.0  # separate depths vertically
        for j, nid in enumerate(ids):
            # spread nodes at this depth around the base line
            y = base + (j - (count - 1) / 2.0) * 0.8
            nodes[nid]["time_x"] = nodes[nid]["offset"]
            nodes[nid]["time_y"] = y

    # Demo length bars (one per node, at its y)
    seg_x = []
    seg_y = []
    for n in nodes:
        length = n["length"]
        if length is None:
            continue
        y = n["time_y"]
        x_start = n["offset"]
        x_end = n["offset"] + length
        seg_x += [x_start, x_end, None]
        seg_y += [y, y, None]

    seg_trace = go.Scatter(
        x=seg_x,
        y=seg_y,
        mode="lines",
        line=dict(width=4, color="rgba(50,50,200,0.6)"),
        hoverinfo="none",
        name="demo length",
    )

    # Node markers
    node_x = [n["time_x"] for n in nodes]
    node_y = [n["time_y"] for n in nodes]

    hover_text = []
    display_text = []
    sizes = []

    for n in nodes:
        length = n["length"]
        end_ts = n["offset"] + length if length is not None else None
        trials = n["trials"]

        ht = [
            n["label"],
            f"task={n['task']}",
            f"offset={n['offset']}",
            f"parent_offset={n['parent_offset']}",
            f"trials={trials}",
        ]
        if length is not None:
            ht.append(f"length={length}")
            ht.append(f"ends_at={end_ts}")
        hover_text.append("<br>".join(ht))

        if trials > 0:
            display_text.append(f"{n['offset']} (x{trials})")
        else:
            display_text.append(str(n["offset"]))

        sizes.append(10 + 3 * trials)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=display_text,
        textposition="bottom center",
        hovertext=hover_text,
        hoverinfo="text",
        marker=dict(
            size=sizes,
            color="orange",
            line=dict(width=1, color="black"),
        ),
        name="demos",
    )

    fig = go.Figure(data=[seg_trace, node_trace])

    # Directed vertical edges (arrows):
    #   For edge u->v, use a single x (child's timestep) and different y's.
    # group edges by x-position (timestep), not by child id
    edges_by_x = defaultdict(list)
    for u, v in edges:
        child_x = nodes[v]["time_x"]
        edges_by_x[child_x].append((u, v))

    arrow_dx = 15  # horizontal spacing between arrows that share the same timestep

    # Directed edges (arrows), with slight horizontal offsets to avoid overlap
    for x_val, edge_list in edges_by_x.items():
        n_at_x = len(edge_list)
        for i, (u, v) in enumerate(edge_list):
            child_x = x_val
            parent_y = nodes[u]["time_y"]
            child_y = nodes[v]["time_y"]

            # center offsets around x_val so multiple arrows at same timestep don't overlap
            offset = (i - (n_at_x - 1) / 2.0) * arrow_dx

            fig.add_annotation(
                x=child_x,                # head at exact timestep
                y=child_y,
                ax=child_x + offset,      # tail slightly shifted left/right
                ay=parent_y,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=3,
                arrowwidth=1,
                arrowcolor="rgba(100,100,100,0.8)",
                hovertext=f"{nodes[u]['label']} → {nodes[v]['label']}",
                hoverlabel=dict(bgcolor="white"),
            )

    fig.update_layout(
        title="Task Branch Graph – Timeline View (with Timestep & Length)",
        showlegend=True,
        xaxis=dict(
            title="Timestep (offset)",
            zeroline=False,
            showgrid=True,
        ),
        yaxis=dict(
            title="Branch depth / variants",
            zeroline=False,
            showticklabels=False,
            showgrid=False,
        ),
        margin=dict(l=60, r=40, t=60, b=60),
        plot_bgcolor="white",
        hovermode="closest",
    )
    return fig


def create_app(
    filenames: List[Filename],
    length_lookup: Optional[Dict[str, int]] = None,
) -> Dash:
    nodes, edges, depths = build_task_graph(filenames, length_lookup)

    topo_fig = build_topology_figure(nodes, edges, depths)
    timeline_fig = build_timeline_figure(nodes, edges, depths)

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.Div(
                [
                    # Left: smaller topology graph (~400x400)
                    html.Div(
                        [
                            dcc.Graph(
                                id="task-graph-topology",
                                figure=topo_fig,
                                style={"width": "100%", "height": "100%"},
                            ),
                        ],
                        style={"width": "400px", "height": "400px"},
                    ),

                    # Right: larger timeline graph (~800x400)
                    html.Div(
                        [
                            dcc.Graph(
                                id="task-graph-timeline",
                                figure=timeline_fig,
                                style={"width": "100%", "height": "100%"},
                            ),
                        ],
                        style={
                            "width": "800px",
                            "height": "400px",
                            "marginLeft": "20px",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "row",
                    "justifyContent": "center",
                    "alignItems": "center",
                },
            ),
        ],
        style={"maxWidth": "1300px", "margin": "0 auto"},
    )
    return app


def visualize_taskgraph(task_name: str, loader, inline: bool = True):
    raw_names = loader.tasks[task_name]['names']
    length_lookup = {}
    for raw_name in raw_names:
        if "trial" not in raw_name:
            length_lookup[raw_name] = loader[raw_name]['length']
    filenames = [Filename(name) for name in raw_names]

    run_kwargs = dict(
        debug=False,
        host="127.0.0.1", port=8090,
        jupyter_width="1200px",      # optional
        dev_tools_ui=False,         # <-- hide the bottom Dev Tools panel
        dev_tools_hot_reload=False  # avoid hijacking previous iframes
    )

    app = create_app(filenames, length_lookup=length_lookup)
    if inline:
        app.run(jupyter_mode="inline", jupyter_height=400, **run_kwargs)
    else:
        app.run(jupyter_mode="tab", **run_kwargs)

if __name__ == "__main__":
    # Example usage with your sample names:
    raw_names = [
        "p0_peg_pick",
        "p0_peg_pick_trial_0",
        "p0_peg_pick_trial_1",
        "p0_peg_pick_trial_3",
        "p0_peg_pick_trial_4",
        "p0_peg_pick_trial_2",
        "p0_peg_pick_branch_at_158",
        "p0_peg_pick_trial_5",
        "p0_peg_pick_branch_at_29",
        "p0_peg_pick_branch_at_39",
        "p0_peg_pick_branch_from_29_at_158",
        "p0_peg_pick_branch_from_0_at_158",
    ]

    filenames = [Filename(name) for name in raw_names]

    # Example demonstration lengths (replace with real values from your data)
    length_lookup = {
        "p0_peg_pick": 300,
        "p0_peg_pick_branch_at_29": 250,
        "p0_peg_pick_branch_at_39": 260,
        "p0_peg_pick_branch_at_158": 200,
        "p0_peg_pick_branch_from_29_at_158": 180,
        "p0_peg_pick_branch_from_0_at_158": 190,
    }

    app = create_app(filenames, length_lookup=length_lookup)
    app.run(debug=True)



