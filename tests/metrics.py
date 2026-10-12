import os
import sys
import numpy as np

from tests.utils import load_cached
from trace.core import TrajectoryManager
from trace.behavior import (state_entropy, medoid_distance, EmpiricalDistribution, distance_matrix,
                            apply_quantization, quantization_mapping)

sys.path.insert(0, os.path.abspath('..'))
os.chdir('../')


def ablation(universal:TrajectoryManager, metric:str='kl'):
    trace_labels, trace_medoids, _ = load_cached(universal.metadata, metric)
    k = trace_labels.max() + 1

    rnd_labels = np.asarray(np.random.randint(0, k, size=len(trace_labels)))
    rnd_medoids = np.unique(rnd_labels, return_index=True)[1].tolist()

    obs, acs, _ = universal.conditioning_features()
    mapping = None if 'quant_bins' not in universal.metadata \
        else quantization_mapping(obs, method='eq', bins=universal.metadata['quant_bins'])

    models = [EmpiricalDistribution(universal.metadata).fit(o, a) for o, a in zip(obs, acs)]
    dist_mat = distance_matrix(models)
    medoid_report(dist_mat, results=(trace_labels, trace_medoids), method='trace')
    medoid_report(dist_mat, results=(rnd_labels, rnd_medoids), method='random')

    entropy_report([universal.subset(trace_labels == l) for l in range(k)], method='trace', mapping=mapping)
    entropy_report([universal], method='no clustering', mapping=mapping)
    entropy_report([universal.subset(rnd_labels == l) for l in range(k)], method='random', mapping=mapping)


def medoid_report(dist_mat:np.ndarray, results:tuple, method:str):
    c = medoid_distance(dist_mat, *results)
    print(f'\n Medoid distance ({method}): ${np.mean(c):.3f} \pm {np.std(c):.3f}$')


def entropy_report(clusters:list[TrajectoryManager], method:str, mapping:tuple|None=None):
    entr = []
    for c, cluster in enumerate(clusters):
        obs, acs, _ = cluster.conditioning_features(per_point=False)
        if mapping is not None: obs = apply_quantization(obs, mapping)
        model = EmpiricalDistribution(cluster.metadata).fit(obs, acs)
        entr.append(np.mean(state_entropy(model)[0]))

    print(f"\n State entropy ({method}): ${np.mean(entr):.4f} \pm {np.std(entr):.4f}$")


if __name__ == '__main__':
    ablation(universal=TrajectoryManager('minetrain').load('ground_truth'))