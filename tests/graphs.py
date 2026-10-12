import os
import sys
import matplotlib.pyplot as plt

from tests.utils import load_cached, load_colors, clear_graph_dir
from trace.core import TrajectoryManager
from trace.visuals import (temporal_alignment, pareto_front, heatmap, trajectory_grid, policy_tree, pareto_clusters,
                           common_temporal, lineplot)
from trace.behavior import state_entropy

sys.path.insert(0, os.path.abspath('..'))
os.chdir('../')


def cluster_graphs(manager:TrajectoryManager, medoid:tuple, suffix:int|str='u',
                   color:str='blue', max_len:int|None=None, directory:str|None=None):
    time_range = (0, max_len) if max_len is not None else None

    title = f'Cluster {suffix}: {len(manager)} trajectories'
    figs = {
        f"trajectory{suffix}.png": trajectory_grid(manager, medoid=medoid[0], title=title, color=color),
        f'temporal{suffix}.png': temporal_alignment(manager, time_range=time_range, title=title),
        f"pareto{suffix}.png": pareto_front(manager, medoid=medoid[2], title=title, color=color),
        f'tree{suffix}.png': policy_tree(manager, title=title, color=color),
    }
    if 'dst' in manager.metadata['file_prefix']:
        figs[f"heatmap{suffix}.png"] = heatmap(*state_entropy(manager), title=title)

    if directory:
        for filename, fig in figs.items():
            fig.savefig(os.path.join(directory, filename), bbox_inches="tight", pad_inches=0)
            plt.close(fig)
    return figs


def integration(universal: TrajectoryManager, metric:str):
    labels, medoids, med_dist = load_cached(universal.metadata, metric)
    colors = load_colors()
    graph_dir = clear_graph_dir(universal.metadata)
    k = labels.max() + 1

    assert len(labels) == len(universal), f'Length mismatch {len(labels)} and {len(universal)}'
    clusters = [universal.subset(labels == l) for l in range(k)]
    pareto_clusters(clusters).savefig(graph_dir + 'paretoc.png')
    common_temporal(clusters).savefig(graph_dir + 'temporalc.png')
    if med_dist is not None: lineplot(med_dist).savefig(graph_dir + 'sensitivity.png')
    trajectory_grid(universal.subset(medoids)).savefig(graph_dir + 'trajectorym.png')

    max_len = max([len(t) for t in universal.sequence('observations')])
    for i, manager in enumerate(clusters + [universal]):
        suffix = i if i < k else 'u'
        medoid = universal[medoids[i]] if i < k else (None,None,None)
        cluster_graphs(manager, suffix=suffix, medoid=medoid, directory=graph_dir, color=colors[i], max_len=max_len)


if __name__ == '__main__':
    integration(TrajectoryManager('minecart').load('ipro'), metric='kl')