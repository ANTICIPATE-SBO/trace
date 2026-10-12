import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

from trace.core import TrajectoryManager
from trace.visuals.utils import tsne_transform, masked_matrix


def temporal_alignment(manager:TrajectoryManager, time_range:tuple|None=None, title:str|None=None, return_ax:bool=False):
    action_names = manager.metadata['actions']
    action_seq = manager.sequence(key='actions', per_point=False)
    action_mat = masked_matrix(action_seq, time_range)

    cmap = plt.colormaps['viridis'].resampled(len(action_names))
    cmap.set_bad(color='white')

    fig, ax = plt.subplots(figsize=(16, 6))
    norm = BoundaryNorm(np.arange(len(action_names) + 1) - 0.5, cmap.N)
    im = ax.imshow(action_mat, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')

    ticks, labels = list(action_names.keys()), list(action_names.values())
    cbar = fig.colorbar(im, ax=ax, ticks=ticks, orientation='horizontal', location='bottom')
    cbar.ax.set_xticklabels(labels, fontsize=25)

    ax.set_yticks([])
    ax.set_xlabel('Time steps', fontsize=25)
    ax.tick_params(axis='x', labelsize=15)

    if title: ax.set_title(title, fontsize=25)
    fig.tight_layout()
    return (fig, ax) if return_ax else fig


def common_temporal(clusters: list[TrajectoryManager]):
    time_range = (0, max(len(traj[0]) for cluster in clusters for traj in cluster))

    fig, new_axs = plt.subplots(len(clusters), ncols=1, sharex=True, figsize=(18, 3 * len(clusters)),)
    new_axs = np.atleast_1d(new_axs)

    for c, cluster in enumerate(clusters):
        old_fig, ax = temporal_alignment(cluster, time_range=time_range, return_ax=True,)

        if len(old_fig.axes) > 1: old_fig.delaxes(old_fig.axes[1])
        im = ax.images[0]
        new_axs[c].imshow(im.get_array(), aspect='auto', cmap=im.get_cmap(), norm=im.norm, interpolation='nearest',)
        new_axs[c].set_yticks([])

        new_axs[c].set_ylabel(f'Cluster {c}', fontsize=25)

        if c < len(clusters) - 1:
            new_axs[c].tick_params(axis='x', labelbottom=False)
        else:
            ticks = list(cluster.metadata['actions'].keys())
            labels = list(cluster.metadata['actions'].values())

            new_axs[c].set_xlabel('Time steps', fontsize=25)
            new_axs[c].tick_params(axis='x', labelsize=20)

        new_axs[c].set_xlim(ax.get_xlim())

    cbar = fig.colorbar(im, ax=new_axs, ticks=ticks, orientation='horizontal', location='bottom')
    cbar.ax.set_xticklabels(labels, fontsize=25)
    return fig



def cluster_scatter(features:np.ndarray, labels:list, colors:list, title:str|None=None, precomputed:bool=True):
    if features.shape[1] > 2: features = tsne_transform(features, precomputed=precomputed)
    fig, ax = plt.subplots(figsize=(8, 6))

    for l in set(labels):
        x, y = features[labels == l, 0], features[labels == l, 1]
        ax.scatter(x, y, label=f"Cluster {l + 1}", color=colors[l % len(colors)], linewidth=8)

    ax.legend()
    ax.grid()
    ax.tick_params(labelbottom=False, labelleft=False)
    if title: ax.set_title(title)
    fig.tight_layout()
    return fig


def heatmap(h:np.ndarray, visited:list|None=None, shape:tuple=(11, 11), title:str|None=None):
    if visited:
        assert len(visited) == len(h), f'Length mismatch {len(visited)} != {len(h)}.'
        heat = np.full(shape, np.nan)
        for s, e in zip(visited, h): heat[*s[:2]] = e
    else: heat = h

    square = shape[0] == shape[1]
    fig, ax = plt.subplots()
    sns.heatmap(heat, ax=ax, square=square, cbar=True)
    ax.set_xticks([])
    ax.set_yticks([])

    if title: ax.set_title(title)
    fig.tight_layout()
    return fig


def distribution_bar(arrays:np.ndarray|list, labels:list, y_range=None,
                     title=None, ax_args:dict|None=None):
    assert len(arrays) == len(labels), f'Length mismatch {len(arrays)} == {len(labels)}.'

    fig, ax = plt.subplots(figsize=(8, 8))
    for arr, label in zip(arrays, labels):
        sns.histplot(arr, kde=True, ax=ax, label=label,)
    if ax_args is None: ax_args = {'xlabel': 'Distance', 'ylabel': 'Pairs of trajectories'}
    ax.set(**ax_args)
    ax.legend()

    if title: ax.set_title('Distribution: ' + title)
    if y_range is not None: ax.set_ylim(*y_range)

    fig.tight_layout()
    return fig


def lineplot(med_dist:dict, title:str|None=None):
    keys = list(med_dist.keys())
    means = [np.mean(values) for values in med_dist.values()]
    stds = [np.std(values) for values in med_dist.values()]

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.lineplot(x=keys, y=means, ax=ax, label='Mean')
    ax.fill_between(keys, np.array(means) - stds, np.array(means) + stds, alpha=0.2, label='Standard Deviation')
    ax.xaxis.set_major_locator(MaxNLocator(10))
    ax.set_xlabel('Number of clusters', fontsize=18)
    ax.set_ylabel('Medoid Distance', fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(fontsize=16)

    if title: ax.set_title(title, fontsize=20)
    fig.tight_layout()
    return fig