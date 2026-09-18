import matplotlib.pyplot as plt
import numpy as np

from trace.core import TrajectoryManager

def pareto_front(manager:TrajectoryManager, medoid:tuple|None=None, noise:bool=True, color:str='red', title:str|None=None):
    _, _, rewards = manager.conditioning_features()

    if rewards.shape[1] == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
    elif rewards.shape[1] == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    else:
        raise ValueError(f'Visualization only supports 2D or 3D rewards, got {rewards.shape[1]} dimensions')

    if noise:
        jitter = 0.005 * (rewards.max(axis=0) - rewards.min(axis=0))
        rewards = rewards + np.random.normal(0, jitter, rewards.shape)

    if medoid is None:
        ax.scatter(rewards[:, 0], rewards[:, 1], *([rewards[:, 2]] if rewards.shape[1] == 3 else []),
                   s=40, color=color, alpha=1.0)
    else:
        ax.scatter(rewards[:, 0], rewards[:, 1], *([rewards[:, 2]] if rewards.shape[1] == 3 else []),
                   s=40, color='grey', alpha=1.0)
        medoid_reward = medoid[2]
        ax.scatter(medoid_reward[0], medoid_reward[1], *( [medoid_reward[2]] if rewards.shape[1] == 3 else []),
                   s=40, color=color, alpha=1.0, label="Medoid")

    ax.grid()
    if title: ax.set_title(title)
    fig.tight_layout()
    return fig


def boxplot(rewards: np.ndarray|list, title: str = "Reward Distribution by Dimension"):
    rewards = np.asarray(rewards)
    n_dims = rewards.shape[1]
    fig, ax = plt.subplots(figsize=(max(6, int(n_dims * 1.2)), 4))

    # tick_labels=[f"Dim {i}" for i in range(n_dims)]
    ax.boxplot([rewards[:, i] for i in range(n_dims)])
    ax.set(title=title, xlabel="Dimension", ylabel="Value")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return fig