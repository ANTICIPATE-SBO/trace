import numpy as np

from trace.behavior.conditioning import EmpiricalDistribution
from trace.core import TrajectoryManager


def state_entropy(model: EmpiricalDistribution|TrajectoryManager):
    if isinstance(model, TrajectoryManager):
        obs, acs, _ = model.conditioning_features()
        model = EmpiricalDistribution(model.metadata).fit(obs, acs)
    if len(model.counts) == 0: return 0.0

    entr, visited = [], model.visited()
    for state in visited:
        p = model.probs(state)
        dec = entropy(p)
        assert not np.isnan(dec), f'Probabilities {p} give NaN'
        entr.append(dec)

    return entr, visited


def medoid_distance(dist_mat:np.ndarray, labels:np.ndarray, medoids:list):
    assert max(labels) + 1 == len(medoids)
    med_dist = []

    for c_id, med_id in enumerate(medoids):
        cluster_points = np.where(labels == c_id)[0]
        dist_to_med = dist_mat[med_id, cluster_points]
        med_dist.append(np.mean(dist_to_med, dtype=float))

    return med_dist


def entropy(p:list|np.ndarray, eps:float|int=1e-16):
    p = np.asarray(p)
    max_h = np.log(len(p), dtype=np.float64)
    h = -np.sum(p * np.log(p + eps), dtype=np.float64)
    assert h < max_h, f'{h} is greater than {max_h}'
    return h / max_h