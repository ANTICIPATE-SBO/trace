import numpy as np

from trace.behavior.conditioning import EmpiricalDistribution
from trace.core import TrajectoryManager


def decisiveness(model: EmpiricalDistribution|TrajectoryManager, entropy:bool=True, alpha:float|int=0.1):
    if isinstance(model, TrajectoryManager):
        obs, acs, _ = model.conditioning_features()
        model = EmpiricalDistribution(model.metadata).fit(obs, acs)
    if len(model.counts) == 0: return 0.0
    dec_function = entropy_dist if entropy else np.std

    d, visited = [], model.visited()
    for state in visited:
        p = model.probs(state, alpha=alpha)
        dec = dec_function(p)
        assert not np.isnan(dec), f'Probabilities {p} give NaN decisiveness'
        d.append(dec)

    return d, visited


def compactness(dist_mat:np.ndarray, labels:np.ndarray, medoid_indices:list):
    assert max(labels) + 1 == len(medoid_indices)
    c = []

    for c_id, med_id in enumerate(medoid_indices):
        cluster_points = np.where(labels == c_id)[0]
        dist_to_med = dist_mat[med_id, cluster_points]
        c.append(np.mean(dist_to_med, dtype=float))

    return c


def entropy_dist(p:list|np.ndarray, eps:float|int=1e-16):
    p = np.asarray(p)
    max_h = np.log(len(p), dtype=np.float64)
    h = -np.sum(p * np.log(p + eps), dtype=np.float64)
    assert h < max_h, f'{h} is greater than {max_h}'
    return h / max_h
