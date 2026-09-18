import os
os.chdir('../')

import numpy as np
np.set_printoptions(precision=4)

from trace.core import TrajectoryManager
from trace.behavior import decisiveness, compactness, EmpiricalDistribution, distance_matrix, apply_quantization, quantization_mapping


def ablation(universal:TrajectoryManager, trace_labels:list, trace_medoids:list|None=None):
    trace_labels, k = np.array(trace_labels), max(trace_labels) + 1
    rnd_labels = np.asarray(np.random.randint(0, k, size=len(trace_labels)))
    rnd_medoids = np.unique(rnd_labels, return_index=True)[1].tolist()

    obs, acs, _ = universal.conditioning_features()
    mapping = None if 'quant_bins' not in universal.metadata \
        else quantization_mapping(obs, method='eq', bins=universal.metadata['quant_bins'])

    if trace_medoids is not None:
        clusterings = (('trace', trace_labels, trace_medoids), ('rnd', rnd_labels, rnd_medoids))
        compactness_report(universal, clusterings, mapping)

    decisiveness_report([universal], method='universal', mapping=mapping)
    decisiveness_report([universal.subset(trace_labels == l) for l in range(k)], method='trace', mapping=mapping)
    decisiveness_report([universal.subset(rnd_labels == l) for l in range(k)], method='random', mapping=mapping)


def compactness_report(universal:TrajectoryManager, clusterings:tuple, mapping:tuple|None=None):
    obs, acs, _ = universal.conditioning_features()
    if mapping is not None: obs = apply_quantization(obs, mapping)

    models = [EmpiricalDistribution(universal.metadata).fit(o, a) for o, a in zip(obs, acs)]
    dist_mat = distance_matrix(models)
    for method, labels, medoids in clusterings:
        c = compactness(dist_mat, labels, medoids)
        print(f'Dist to medoids ({method}): ', c)
        print(f'\t-> avg: {np.mean(c):.3f} +- {np.std(c):.3f}')
    print()


def decisiveness_report(clusters:list[TrajectoryManager], method:str, mapping:tuple|None=None):
    print(f"\n{method.upper()}")
    print('per cluster: ', end='')
    cluster_ents = []
    for c, cluster in enumerate(clusters):
        obs, acs, _ = cluster.conditioning_features(per_point=False)
        if mapping is not None: obs = apply_quantization(obs, mapping)
        model = EmpiricalDistribution(cluster.metadata).fit(obs, acs)
        dec, _ = decisiveness(model, entropy=True)
        cluster_ents.append(dec)
        print(f'{np.mean(dec):.5f} ', end='')
    print()

    avg_ent = np.mean([e for c_ent in cluster_ents for e in c_ent])
    print(f"\t-> avg entr: {avg_ent:.5f}")

if __name__ == '__main__':
    from trace.clustering.cached_labels import dst_k_medoids_kl as cached

    ablation(universal=TrajectoryManager('dst-conc').load('ground_truth'),
             trace_labels=cached.labels, trace_medoids=cached.medoid_indices)