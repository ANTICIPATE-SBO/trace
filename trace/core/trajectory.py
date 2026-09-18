from json import load, dumps
from yaml import safe_load
from os import path

import numpy as np

from trace.core.maths import discount, pareto_filter


class TrajectoryManager:
    def __init__(self, metadata: dict|str):
        self.metadata = metadata if isinstance(metadata, dict) \
            else safe_load(open(f'trace/configs/{metadata}.yaml'))
        self.env_id = self.metadata['env_id']
        self.num_actions = len(self.metadata['actions'])
        self.gamma = self.metadata['gamma']
        self.trajectories = []

    def load(self, source:str|list, pareto:bool=True, duplicates:bool=False, split:bool=True):
        if isinstance(source, str):
            filepath = path.join('data/', self.metadata['file_prefix']) + f'_{source}.json'
            self.trajectories = load(open(filepath, 'rb'))
            self.metadata['policy'] = source
        elif isinstance(source, list): self.trajectories = source
        else: raise ValueError(f'Unknown source type: {type(source)}')

        if split: self.trajectories = [[traj] for point in self.trajectories for traj in point]
        if duplicates: self.trajectories = filter_duplicates(self.trajectories)
        if pareto: self.trajectories = [self.trajectories[ind] for ind in pareto_filter(self.accrue())]

        self._verify_data()
        return self

    def subset(self, labels: list|np.ndarray):
        if len(labels) < len(self):
            mask = [False] * len(self)
            for i in labels: mask[i] = True
        elif len(labels) == len(self): mask = labels
        else: raise ValueError(f'Length mismatch, too many labels')

        return TrajectoryManager(metadata=self.metadata).load([t for t, m in zip(self.trajectories, mask) if m])

    def _verify_data(self):
        action_seq = self.sequence(key='actions', per_point=False, pad=None)
        assert all(a in self.metadata['actions'] for ep in action_seq for a in ep)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self,i):
        states, actions, rewards = self.conditioning_features()
        return states[i], actions[i], rewards[i]

    def accrue(self, key: str='rewards', gamma: float|None=None):
        if gamma is None: gamma = self.gamma
        return np.array([
            np.mean([
                discount(trajectory[key], gamma) for trajectory in point
            ], axis=0)
            for point in self.trajectories
        ])

    def sequence(self, key:str='actions', per_point:bool=False):
        seq = [[t[key] for t in point] for point in self.trajectories] if per_point \
            else [t[key] for point in self.trajectories for t in point]
        return seq

    def conditioning_features(self, per_point:bool=False, accrue:bool=False, gamma:float|int|None=None, ):
        obs = self.sequence(key='observations', per_point=per_point)
        acs = self.sequence(key='actions', per_point=per_point)
        rew = self.accrue(key='rewards', gamma=gamma) if accrue \
            else self.sequence(key='rewards', per_point=per_point)
        return obs, acs, rew

    def save(self, filepath: str):
        with open(filepath, 'w') as f: f.write(dumps(self.trajectories, indent=2))


def filter_duplicates(array:list, sort:bool=True):
    filtered, seen = [], set()
    for element in array:
        if sort:
            if isinstance(element, dict): element = dict(sorted(element.items()))
            else: element = sorted(element)
        if (element_str := str(element)) not in seen:
            seen.add(element_str)
            filtered.append(element)
    return filtered
