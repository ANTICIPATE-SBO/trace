import numpy as np
from matplotlib import pyplot as plt

from trace.core import TrajectoryManager
from trace.visuals.utils import env_frame, add_noise


def trajectory_grid(manager:TrajectoryManager, abstr_frame:bool=True, medoid:list|None=None,
                      title:str|None=None, color:str|None=None):
    metadata = manager.metadata
    offset = 0.5 if 'deep-sea-treasure' in metadata['env_id'] else 0

    fig, ax = plt.subplots()
    h, w = np.array(metadata['obs_high'])[:2]
    ax.imshow(env_frame(metadata, abstr_frame), extent=(0, w, 0, h), origin='lower')

    obs, acs, _ = manager.conditioning_features()
    if medoid: ax = medoid_grid(obs, medoid, color, ax, offset)
    elif color is None: ax = cluster_grid(obs, ax, offset)
    else: ax = universal_grid(obs, ax, offset)

    if 'minecart' in metadata['env_id']: ax = minecart_grid(obs, acs, ax)

    ax.set(xlim=(0, w), ylim=(h, 0), aspect='equal')
    ax.axis('off')
    if title: ax.set_title(title)
    fig.tight_layout(pad=0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig


def minecart_grid(obs:list, acs:list, ax:plt.Axes):
    mining_states = [coords  for traj_obs, traj_acs in zip(obs, acs)
                     for coords, a in zip(traj_obs, traj_acs) if a == 0]
    y_mine, x_mine = np.asarray(mining_states)[:, :2].T
    ax.scatter(x_mine, y_mine, marker='o', color='yellow', s=40, label='Mining performed', zorder=10)

    finishing_states = [traj_obs[-1] for traj_obs in obs]
    y_finish, x_finish = list(zip(*finishing_states))[:2]
    ax.scatter(x_finish, y_finish, marker='x', color='blue', s=40, label='End of trajectory', zorder=10)

    ax.legend()
    return ax


def universal_grid(obs:list, ax:plt.Axes, offset:float|int):
    for i, traj in enumerate(obs):
        features = np.asarray(traj).T[:2]
        y, x = features[0] + offset, features[1] + offset
        ax.plot(x, y, linewidth=1.5, color='black', label='Trajectories' if i==0 else None)
    ax.legend()
    return ax


def cluster_grid(obs:list, ax:plt.Axes, offset:float|int):
    d = 0.01 if offset==0 else 0.1
    linewidth = 3 if offset==0 else 2

    for i in range(len(obs)):
        features = list(zip(*obs[i]))

        y, x = np.array(features[0]), np.array(features[1])
        y, x = y + offset - d*i, x + offset+ d*i

        linestyle = ':' if offset==0 else None
        ax.plot(x, y, linewidth=linewidth, linestyle=linestyle, label=f'Cluster {i}')

    ax.legend()
    return ax


def medoid_grid(obs:list, medoid:list, color:str|None, ax:plt.Axes, offset:float|int):
    if color is None: color = 'red'
    linestyle = 'dotted' if offset == 0 else None
    alpha = 0.2 if len(obs) < 200 else 0.01
    for i, traj in enumerate(obs):
        features = list(zip(*traj))
        y, x = add_noise(features[0]) + offset, add_noise(features[1]) + offset
        ax.plot(x, y, alpha=alpha, linewidth=1.5, color='black', label='Trajectories' if i==0 else None)
    y_med, x_med = np.asarray(medoid).T[:2]
    ax.plot(x_med + offset, y_med + offset, alpha=1, linewidth=3, linestyle=linestyle ,color=color, label='Medoid')
    ax.legend()
    return ax


def quantization_grid(ranges:list, metadata:dict, title:str|None=None, abstr_frame:bool=False):
    abstr_frame &= 'deep-sea-treasure' in metadata['env_id']
    h, w = np.array(metadata['obs_high'])[:2]

    fig, ax = plt.subplots()
    ax.imshow(env_frame(metadata, abstr_frame), extent=(0, w, 0, h), origin='lower')
    ax.vlines(ranges[0], 0, h, linestyle='dotted', color='blue', linewidth=0.5)
    ax.hlines(ranges[1], 0, w, linestyle='dotted', color='blue', linewidth=0.5)
    ax.set(xlim=(0, w), ylim=(h, 0), aspect='equal')

    plt.axis('off')
    if title is not None: ax.set_title(title)
    fig.tight_layout()
    return fig

