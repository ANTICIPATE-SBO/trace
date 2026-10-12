import os
import sys
import numpy as np

from json import dumps
from collections.abc import Callable

from trace.behavior import EmpiricalDistribution, distance_matrix, quantization_mapping, apply_quantization, medoid_distance
from trace.clustering import k_medoids
from trace.core import TrajectoryManager

sys.path.insert(0, os.path.abspath('..'))
os.chdir('../')


def behavior_clustering(manager:TrajectoryManager, cluster:Callable, k:int, metric:str='kl', save:bool=True):
    obs, acs, rew = manager.conditioning_features(per_point=False)
    if 'quant_bins' in manager.metadata:
        bins = manager.metadata['quant_bins']
        obs = apply_quantization(obs, quantization_mapping(obs, method='eq', bins=bins))

    behavior_models = [EmpiricalDistribution(manager.metadata).fit(o, a) for o, a in zip(obs, acs)]
    features = distance_matrix(behavior_models, metric=metric, norm=True)
    labels, medoid_indices = cluster(features, k=k, metric='precomputed', return_medoids=True)

    if save:
        prefix, source = manager.metadata['file_prefix'], manager.source
        with open(f'trace/clustering/cached/{prefix}_{source}_{metric}.py', 'w') as f:
            f.write('labels = ')
            f.write(str(labels.tolist()))
            f.write('\nmedoid_indices = ')
            f.write(str(medoid_indices.tolist()))

            print('Saved labels and medoid indices')

    return [manager.subset(labels == l) for l in range(k)], labels


def sensitivity_analysis(manager:TrajectoryManager, cluster:Callable, metric:str='kl', save:bool=True):
    k_samples = [i for i in range(1,41)]
    obs, acs, rew = manager.conditioning_features(per_point=False)
    if 'quant_bins' in manager.metadata:
        bins = manager.metadata['quant_bins']
        obs = apply_quantization(obs, quantization_mapping(obs, method='eq', bins=bins))
    behavior_models = [EmpiricalDistribution(manager.metadata).fit(o, a) for o, a in zip(obs, acs)]
    features = distance_matrix(behavior_models, metric=metric, norm=True)

    med_dist = dict()
    for k in k_samples:
        labels, medoids = cluster(features, k=k, metric='precomputed', return_medoids=True)
        med_dist[k] = medoid_distance(features, labels, medoids)
        print(f'\tk={k}: {np.mean(med_dist[k]):.3f} +- {np.std(med_dist[k]):.3f}')

    if save:
        prefix, source = manager.metadata['file_prefix'], manager.source
        with open(f'trace/clustering/cached/{prefix}_{source}_sensitivity.py', 'w') as f:
            f.write('med_dist = ')
            f.write(dumps(med_dist, indent=2))

            print('Saved medoid distance per k')


if __name__ == '__main__':
    behavior_clustering(TrajectoryManager('minetrain').load('ground_truth'), k_medoids, k=5)
    #sensitivity_analysis(TrajectoryManager('dst-conc').load('ground_truth'), k_medoids)
