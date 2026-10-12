from json import load, dumps
from yaml import safe_load
from os import path

import numpy as np


class TrajectoryManager:
    def __init__(self, metadata: dict|str):
        self.metadata = metadata if isinstance(metadata, dict) \
            else safe_load(open(f'trace/configs/{metadata}.yaml'))
        self.env_id = self.metadata['env_id']
        self.num_actions = len(self.metadata['actions'])
        self.gamma = self.metadata['gamma']
        self.trajectories, self.source = [], None

    def load(self, source:str|list, pareto:bool=True, duplicates:bool=True, split:bool=True):
        if isinstance(source, str):
            source = source.replace("-", "_")
            self.source = source
            filepath = path.join('data/', self.metadata['file_prefix']) + f'_{source}.json'
            self.trajectories = load(open(filepath, 'rb'))
            self.metadata['policy'] = source
        elif isinstance(source, list): self.trajectories = source
        else: raise ValueError(f'Unknown source type: {type(source)}')

        if split: self.trajectories = [[traj] for point in self.trajectories for traj in point]
        if duplicates: self.trajectories = duplicates_filter(self.trajectories)
        if pareto: self.trajectories = [self.trajectories[ind] for ind in pareto_filter(self.accrue())]

        self._verify_data()
        return self

    def subset(self, labels: list|np.ndarray):
        if len(labels) < len(self):
            trajectories = [self.trajectories[l] for l in labels]
        elif len(labels) == len(self):
            trajectories = [self.trajectories[i] for i, l in enumerate(labels) if l]
        else:
            raise ValueError(f'Length mismatch, too many labels')

        return TrajectoryManager(metadata=self.metadata).load(trajectories)

    def medoid_subset(self, medoids: list|np.ndarray):
        trajectories = [self.trajectories[m] for m in medoids]
        return TrajectoryManager(metadata=self.metadata).load(trajectories)

    def _verify_data(self):
        action_seq = self.sequence(key='actions', per_point=False)
        assert all(a in self.metadata['actions'] for ep in action_seq for a in ep)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self,i):
        states, actions, rewards = self.conditioning_features()
        return states[i], actions[i], rewards[i]

    def accrue(self, key: str='rewards', gamma: float|int|None=None):
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

    def conditioning_features(self, per_point:bool=False, accrue:bool=True, gamma:float|int|None=None, ):
        obs = self.sequence(key='observations', per_point=per_point)
        acs = self.sequence(key='actions', per_point=per_point)
        rew = self.accrue(key='rewards', gamma=gamma) if accrue \
            else self.sequence(key='rewards', per_point=per_point)
        return obs, acs, rew

    def save(self, filepath: str|None=None):
        if filepath is None:
            filepath = path.join('data/', self.metadata['file_prefix']) + f'_{self.source}.json'
        with open(filepath, 'w') as f: f.write(dumps(self.trajectories, indent=2))


def discount(ar: np.ndarray, gamma: float=0.99):
    if not isinstance(ar, np.ndarray): ar = np.array(ar)
    discounts = gamma ** np.arange(ar.shape[0], dtype=np.float32)
    return np.sum(ar * discounts[:, None], axis=0)


def pareto_filter(acc_rewards: np.ndarray):
    is_pareto = np.ones(len(acc_rewards), dtype=bool)

    for i in range(acc_rewards.shape[0]):
        if not is_pareto[i]: continue

        for j in range(len(acc_rewards)):
            if i == j: continue
            ri, rj = acc_rewards[i], acc_rewards[j]

            if np.all(rj >= ri) and np.any(rj > ri):
                is_pareto[i] = False
                break

    return np.where(is_pareto)[0]

def duplicates_filter(array:list, sort:bool=True):
    filtered, seen = [], set()
    for element in array:
        if sort:
            if isinstance(element, dict): element = dict(sorted(element.items()))
            else: element = sorted(element)
        if (element_str := str(element)) not in seen:
            seen.add(element_str)
            filtered.append(element)
    return filtered