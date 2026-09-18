import numpy as np
from matplotlib import pyplot as plt

from trace.core import TrajectoryManager
from trace.visuals.utils import env_frame, add_noise


def trajectory_grid(manager:TrajectoryManager, abstr_frame:bool=False, medoid:tuple|None=None,
                      title: str|None=None, alpha: float|int=0.1, color: str='red'):
    metadata = manager.metadata
    offset = 0.5 if 'deep-sea-treasure' in metadata['env_id'] else 0

    fig, ax = plt.subplots()
    h, w = np.array(metadata['obs_high'])[:2]
    ax.imshow(env_frame(metadata, abstr_frame), extent=(0, w, 0, h), origin='lower')

    obs, acs, _ = manager.conditioning_features()
    features = list(zip(*obs))
    y, x = add_noise(features[0]) + offset, add_noise(features[1]) + offset
    ax.plot(x, y, alpha=alpha, linewidth=1.5, color='black' if medoid else color)

    if medoid:
        y_med, x_med = list(zip(*medoid[0]))[:2]
        ax.plot(x_med + offset, y_med + offset, alpha=1, linewidth=1.5, color=color)

    if 'minecart' in metadata['env_id']:
        mining_states = [
            [coords for coords, a in zip(traj_obs, traj_acs) if a == 0]
            for traj_obs, traj_acs in zip(obs, acs)
        ]
        y_mine, x_mine = list(zip(*mining_states))[:2]
        ax.scatter(y_mine + offset, x_mine + offset, marker='x', color='blue', s=20)

    ax.set(xlim=(0, w), ylim=(h, 0), aspect='equal')
    ax.axis('off')
    if title: ax.set_title(title)
    fig.tight_layout(rect=(0, 0, 1, 1))
    return fig


def quantization_grid(ranges:list, metadata:dict, title:str|None=None, abstr_frame:bool=False):
    abstr_frame &= 'deep-sea-treasure' in metadata['env_id']
    h, w = np.array(metadata['obs_high'])[:2]

    fig, ax = plt.subplots()
    ax.imshow(env_frame(metadata, abstr_frame), extent=(0, w, 0, h), origin='lower')
    ax.vlines(ranges[0], 0, h, linestyle='dashed', color='blue', linewidth=0.5)
    ax.hlines(ranges[1], 0, w, linestyle='dashed', color='blue', linewidth=0.5)
    ax.set(xlim=(0, w), ylim=(h, 0), aspect='equal')

    plt.axis('off')
    if title is not None: ax.set_title(title)
    fig.tight_layout()
    return fig

