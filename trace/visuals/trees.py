import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from trace.core import TrajectoryManager
from trace.behavior import EmpiricalDistribution
from trace.visuals.utils import remove_redundancies, find_best_split, split_range


def policy_tree(model: EmpiricalDistribution|TrajectoryManager, max_depth:int=4, collapse:bool=True, title:str|None=None, color:str="#B85450"):
    if isinstance(model, TrajectoryManager):
        model = EmpiricalDistribution(model.metadata).fit(*model.conditioning_features()[:2])
    states, counts = zip(*model.counts.items())
    states, counts = np.array(states), np.array(counts)
    low, high = model.metadata['obs_low'], model.metadata['obs_high']
    rng = [[l, h] for l, h in zip(low, high)]

    nodes =  make_nodes(states, counts, depth=0, rng=rng, max_depth=max_depth)
    graph = nodes_to_graph(nodes, model.metadata['features'], model.metadata['actions'])
    if collapse: graph = remove_redundancies(graph)
    return visualize(graph, title=title, color=color, dst='dst' in model.metadata['file_prefix'])


def visualize(graph:nx.DiGraph, title=None, color:str="#B85450", dst:bool=True):
    fig, ax = plt.subplots(figsize=(8, 12) if dst else (10,12))
    pos = nx.drawing.nx_pydot.graphviz_layout(graph, prog="dot")

    args = {
        'node_size': 4000 if dst else 2000,
        'node_shape': 'o' if dst else '',
        'node_color': color if dst else 'white',
        'edgecolors': 'black' if dst else None,
        'with_labels': False
    }
    nx.draw(graph, pos=pos, ax=ax, **args)

    label_args = {'font_size': 18, 'font_weight': 'bold'}
    nx.draw_networkx_labels(
        graph, pos=pos, ax=ax,
        labels=nx.get_node_attributes(graph, "label"),
        font_color="white" if dst else color, **label_args
    )
    nx.draw_networkx_edge_labels(
        graph, pos=pos, ax=ax,
        edge_labels=nx.get_edge_attributes(graph, "label"),
        **label_args
    )
    ax.axis("off")
    if title: ax.set_title(title, fontsize=25)
    return fig


def make_nodes(states:np.ndarray, counts:np.ndarray, depth:int, rng:list, max_depth:int):
    total_counts = np.sum(counts, axis=0)
    if len(np.nonzero(total_counts)[0]) == 1 or depth >= max_depth:
        return {"type": "leaf", "action": np.argmax(total_counts), "depth": depth,}

    best_feature, best_split = find_best_split(states, counts, rng)
    if best_feature is None or best_split is None :
        action = int(np.argmax(total_counts))
        return {"type": "leaf", "action": action, "depth": depth,}

    mask = states[:, best_feature] < best_split
    less_states, less_counts = states[mask], counts[mask]
    more_states, more_counts = states[~mask], counts[~mask]
    left, right = split_range(rng, best_feature, best_split)

    return {
        "type": "split", "feature": best_feature, "split": best_split, "depth": depth, "rng": rng,
        "yes": make_nodes(less_states, less_counts, depth + 1, left, max_depth,),
        "no": make_nodes(more_states, more_counts, depth + 1, right, max_depth,),
    }


def nodes_to_graph(tree:dict, feature_names:list|None=None, action_names:dict|None=None):
    graph = nx.DiGraph()

    def add(node):
        node_id = id(node)

        if node["type"] == "leaf":
            label = f"A{node['action']}" if action_names is None \
                else action_names[node["action"]]
            graph.add_node(node_id, label=label)
            return

        label = f"x{node['feature']}<{node['split']:.1f}" if feature_names is None \
            else f"{feature_names[node['feature']]}<{node['split']:.1f}"

        graph.add_node(node_id, label=label)

        for child, edge in [(node["yes"], "yes"), (node["no"], "no")]:
            child_id = id(child)
            graph.add_edge(node_id, child_id, label=edge)
            add(child)

    add(tree)
    return graph