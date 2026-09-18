import os
os.chdir('../')

import numpy as np

from trace.behavior import EmpiricalDistribution, distance_matrix, apply_quantization, quantization_mapping
from trace.core import TrajectoryManager
from trace.visuals import heatmap, distribution_bar, quantization_grid

def overlap_actions(models:list[EmpiricalDistribution]):
    common_state_actions = []

    for i in range(len(models)):
        vi = set(models[i].visited())

        for j in range(i, len(models)):
            assert models[i].feature_mask == models[j].feature_mask, \
                f'State feature mask mismatch'
            vj = set(models[j].visited())

            common_state_actions.append(np.sum(
                [models[i].counts[v] + models[j].counts[v] for v in vi&vj]
            ))

    return common_state_actions


def quantization_comparison(manager:TrajectoryManager):
    og_obs, acs, _ = manager.conditioning_features(per_point=False)
    bins = manager.metadata['quant_bins']

    titles = ['without quantization', 'uniform quantization', 'equal quantization',]
    all_obs, all_ranges = [og_obs], [None]
    for method in ['uni', 'eq']:
        ranges, quants = quantization_mapping(og_obs, method=method, bins=bins)
        all_obs.append(apply_quantization(og_obs, (og_obs, ranges)))
        all_ranges.append(ranges)

    for obs, rng, title in zip(all_obs, all_ranges, titles):
        models = [EmpiricalDistribution(manager.metadata).fit(o, a) for o, a in zip(obs, acs)]
        features = distance_matrix(models, metric='kl')
        heatmap(features, title=title).show()
        distribution_bar(features.flatten(), title='Distances '+title).show()
        distribution_bar(overlap_actions(models), title='Actions overlap '+title).show()
        if rng is None: continue
        quantization_grid(rng, manager.metadata, title='State split '+title).show()


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    quantization_comparison(TrajectoryManager('dst-conc').load('ground_truth'))

