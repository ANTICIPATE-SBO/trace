import os
os.chdir('../')

from collections.abc import Callable

from trace.behavior import EmpiricalDistribution, distance_matrix, quantization_mapping, apply_quantization
from trace.clustering import k_medoids
from trace.core import TrajectoryManager


def behavior_clustering(manager:TrajectoryManager, cluster:Callable, k:int, metric:str, save:bool=True):
    obs, acs, rew = manager.conditioning_features(per_point=False)
    #obs = apply_quantization(obs, quantization_mapping(obs, method='eq', bins=[100, 100, 6, 10, 10, 90, 90]))

    behavior_models = [EmpiricalDistribution(manager.metadata).fit(o, a) for o, a in zip(obs, acs)]
    features = distance_matrix(behavior_models, metric=metric, norm=True)
    labels, medoid_indices = cluster(features, k=k, metric='precomputed', return_medoids=True)

    if save:
        prefix, func = manager.metadata['file_prefix'], cluster.__name__
        with open(f'trace/clustering/cached_labels/{prefix}_{func}_{metric}.py', 'w') as f:
            f.write('labels = ')
            f.write(str(labels.tolist()))
            f.write('\nmedoid_indices = ')
            f.write(str(medoid_indices.tolist()))

            print('Saved labels and medoid indices')

    return [manager.subset(labels == l) for l in range(k)], labels

def silhouette():
    return NotImplemented

if __name__ == '__main__':
    behavior_clustering(TrajectoryManager('dst-conc').load('ground_truth'), k_medoids, k=10, metric='kl')