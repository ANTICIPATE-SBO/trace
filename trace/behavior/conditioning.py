from typing import Hashable
from collections import defaultdict
import numpy as np


class EmpiricalDistribution:
    def __init__(self, metadata:dict, feature_mask:list[bool]|None=None):
        self.metadata = metadata
        self.n = len(metadata['actions'])
        self.counts = defaultdict(lambda: np.zeros(self.n, dtype=float))
        if feature_mask is not None:
            assert len(feature_mask) == len(metadata['observations_high']), 'Feature mask shape mismatch'
        self.mask = feature_mask
        self.trajectories_num = 0

    def visited(self):
        return list(self.counts.keys())

    def fit(self, obs:list|np.ndarray, acs:list|np.ndarray):
        if isinstance(obs[0][0], Hashable): obs, acs = [obs], [acs]
        self.trajectories_num = len(obs)

        for episode_obs, episode_acs in zip(obs, acs):
            episode_counts = defaultdict(lambda: np.zeros(self.n, dtype=float))

            for coords, action in zip(episode_obs, episode_acs):
                episode_counts[self.feature_map(coords)][action] += 1

            for state, count in episode_counts.items():
                counts_norm = count / sum(count)
                self.counts[state] += counts_norm

        return self

    def probs(self, coords:list|tuple, alpha:float|int=0.0):
        coords = self.feature_map(coords)
        if not coords in self.visited(): return np.array([1 / self.n] * self.n)

        probs = self.counts[coords]
        assert sum(probs) <= self.trajectories_num, f'{probs} >= {self.trajectories_num}'
        unvisited = (self.trajectories_num - sum(probs)) * np.array([1 / self.n] * self.n)
        return (probs + unvisited) / np.sum(probs + unvisited)




    def act(self, coords, deterministic:bool=True):
        probs = self.probs(coords)
        if deterministic: return int(probs.argmax())
        return int(np.random.choice(self.n, p=probs))


    def feature_map(self, coords:list|tuple):
        if self.mask is None: return tuple(coords)
        return (f for f, m in zip(coords, self.mask) if m)

