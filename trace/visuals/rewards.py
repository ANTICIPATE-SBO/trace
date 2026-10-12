import matplotlib.pyplot as plt
import numpy as np

from trace.core import TrajectoryManager
from trace.visuals.utils import add_noise

def pareto_front(manager:TrajectoryManager, medoid:list|np.ndarray|None=None, noise:bool=True, color:str='red', title:str|None=None):
    rewards = manager.accrue('rewards')
    if noise: rewards = add_noise(rewards)

    if rewards.shape[1] == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
    elif rewards.shape[1] == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    else:
        raise ValueError(f'Visualization only supports 2D or 3D rewards, got {rewards.shape[1]} dimensions')

    if medoid is None:
        ax.scatter(*rewards.T, s=40, color=color, alpha=1.0)
    else:
        ax.scatter(*rewards.T, s=40, color='grey', alpha=0.7)
        ax.scatter(*medoid, s=40, color=color, alpha=1.0, label="Medoid")

    ax.grid()
    if title: ax.set_title(title)
    fig.tight_layout()
    return fig


def pareto_clusters(clusters:list[TrajectoryManager]):
    rewards = [cluster.accrue('rewards') for cluster in clusters]
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, reward in enumerate(rewards):
        jitter = 0.01 * (reward.max(axis=0) - reward.min(axis=0))
        reward = reward + np.random.normal(0, jitter, reward.shape)
        ax.scatter(reward[:, 0], reward[:, 1], s=200, label=f'Cluster {i}', alpha=0.3)
    ax.legend()
    ax.grid()
    fig.tight_layout()
    return fig
