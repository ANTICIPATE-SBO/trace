from copy import deepcopy
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from trace.behavior import entropy
from trace.core import all_ints


def tsne_transform(data:list|np.ndarray, precomputed:bool=False, perplexity:int|None=None):
    from sklearn.manifold import TSNE

    data = np.array(data)
    if precomputed: assert data.shape[0] == data.shape[1], f"Invalid similarity matrix shape: {data.shape}"
    if perplexity is None: perplexity = max(30, len(data)//3)
    tsne = TSNE(
        n_components = 2,
        perplexity = min(perplexity, len(data)-1),
        metric = 'precomputed' if precomputed else 'euclidean',
        init = 'random' if precomputed else 'pca',
        learning_rate = "auto",
        random_state = 42
    )
    return tsne.fit_transform(data)


def env_frame(metadata:dict, abstr:bool=False):
    if abstr and 'frame_dir' in metadata: return plt.imread(metadata['frame_dir'])

    import mo_gymnasium as mo_gym
    render_env = mo_gym.make(metadata['env_id'], render_mode="rgb_array")
    render_env.reset()
    frame = render_env.render()
    return frame


def add_noise(x:list|np.ndarray):
    x = np.array(x)
    jitter = 0.005 * (np.max(x) - np.min(x))
    return x + np.random.normal(0, jitter, x.shape)


def masked_matrix(action_seq:list|np.ndarray, time_range:tuple|None, fill_value:int=-1):
    if time_range:
        start, end = time_range
        max_len = end - start
    else:
        max_len = max(len(seq) for seq in action_seq)
        start, end = 0, max_len - 1

    matrix = np.full((len(action_seq), max_len), fill_value=fill_value)
    for i, seq in enumerate(action_seq):
        for j in range(start, min(end + 1, len(seq))):
            matrix[i, j - start] = seq[j]
    return np.ma.masked_where(matrix == -1, matrix)


def remove_redundancies(graph: nx.DiGraph):
    while True:
        changed = False
        for node in list(graph.nodes):
            if node not in graph: continue
            children = list(graph.successors(node))

            if len(children) == 2 and all(graph.out_degree(c) == 0 for c in children):
                if graph.nodes[children[0]]["label"] == graph.nodes[children[1]]["label"]:
                    label = graph.nodes[children[0]]["label"]
                    graph.remove_nodes_from(children)
                    graph.nodes[node]["label"] = label
                    changed = True

        if not changed: break
    return graph


def find_best_split(states: np.ndarray, counts: np.ndarray, rng:list):
    best_score, best_feature, best_split = np.inf, None, None

    for feature in range(len(states[0])):
        feature_score, feature_split = np.inf, None
        splits = range(*rng[feature]) if all_ints(rng[feature]) else np.linspace(*rng[feature], 10)
        for split in splits:
            mask = states[:, feature] < split
            less, more = counts[mask].sum(axis=0), counts[~mask].sum(axis=0)
            if less.sum() == 0 or more.sum() == 0: continue

            e_less, e_more = entropy(less / less.sum()), entropy(more / more.sum())
            if (score := max(e_less, e_more)) < feature_score:
                feature_score, feature_split = score, split

        if feature_score < best_score:
            best_score, best_feature, best_split = feature_score, feature, feature_split

    return best_feature, best_split


def split_range(rng:list, feature:int, split:int):
    left, right = deepcopy(rng), deepcopy(rng)
    left[feature] = (rng[feature][0], split)
    right[feature] = (split, rng[feature][1],)
    return left, right
