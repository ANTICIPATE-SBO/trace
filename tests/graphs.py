import os
os.chdir('../')
import numpy as np
from yaml import safe_load

from trace.core import TrajectoryManager
from trace.visuals import temporal_alignment, decision_tree, boxplot, pareto_front, heatmap, trajectory_grid
from trace.behavior import decisiveness

def cluster_graphs(manager:TrajectoryManager, suffix:int|str='u', medoid:tuple|None=None,
                   color:str='blue', max_len:int|None=None, directory:str|None=None):
    time_range = (0, max_len) if max_len is not None else None

    title = f'Cluster {suffix}: {len(manager)} trajectories'
    figs = {
        f"trajectory{suffix}.png": trajectory_grid(manager, abstr_frame=True, medoid=medoid, title=title, color=color),
        f'temporal{suffix}.png': temporal_alignment(manager, time_range=time_range, title=title),
        f'decision_tree{suffix}.png': decision_tree(manager, title=title),
        #f'boxplot{ind}.png': boxplot(rew, title=title),
        f"pareto{suffix}.png": pareto_front(manager, medoid=medoid, color=color, title=title),
        f"heatmap{suffix}.png": heatmap(*decisiveness(manager), title=title),
    }

    if directory:
        for filename, fig in figs.items():
            fig.savefig(os.path.join(directory, filename))
    return figs


def integration(universal: TrajectoryManager, metric:str, labels:list, medoids:list):
    colors = safe_load(open(f'trace/configs/colors.yaml'))['warm']

    file_prefix, policy = universal.metadata['file_prefix'], universal.metadata['policy']
    k = max(np.array(labels)) + 1
    graph_directory = f"plots/{file_prefix}/{policy}/{metric}"
    for file in os.listdir(graph_directory):
        path = os.path.join(graph_directory, file)
        if os.path.isfile(path): os.remove(path)

    assert len(labels) == len(universal), f'Length mismatch {len(labels)} and {len(universal)}'
    clusters = [universal.subset(np.array(labels) == l) for l in range(k)] + [universal]

    for i, manager in enumerate(clusters):
        suffix, medoid = (i, universal[medoids[i]]) if i < k else ('u', None)
        figs = cluster_graphs(manager, suffix=suffix, medoid=medoid, directory=graph_directory, color=colors[i])

        #figs = cluster_graphs(manager, medoid=medoid, suffix=suffix, color=colors[i])
        #for _, fig in figs.items(): fig.show()


if __name__ == '__main__':
    import trace.clustering.cached_labels.dst_k_medoids_kl as cached

    integration(TrajectoryManager('dst-conc').load('ground_truth'), metric='kl', labels=cached.labels,
                medoids=cached.medoid_indices)
