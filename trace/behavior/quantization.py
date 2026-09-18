import copy
from collections import Counter

import numpy as np


def quantization_mapping(obs: list|np.ndarray, method:str, bins:int|list):
    method_dict = {'eq': equal_quantization, 'uni': uniform_quantization}
    assert method in method_dict, f'Method {method} is not supported'

    n_features = len(obs[0][0])
    if isinstance(bins, int): bins = [bins] * n_features

    ranges, quants = [], []
    for n in range(n_features):
        values = [t[n] for trajectory in obs for t in trajectory]
        r, q = method_dict[method](values, bins[n])
        ranges.append(r)
        quants.append(q)

    return ranges, quants


def apply_quantization(obs:list|np.ndarray, mapping:tuple):
    quant_obs, n_features = copy.deepcopy(obs), len(obs[0][0])
    ranges, quants = mapping

    for n in range(n_features):
        for i, episode_obs in enumerate(obs):
            for j, coords in enumerate(episode_obs):
                idx = np.searchsorted(ranges[n], coords[n], side='right') - 1
                idx = np.clip(idx, 0, len(quants[n]) - 1)
                quant_obs[i][j][n] = quants[n][idx]
    return quant_obs


def uniform_quantization(values:list, bins: int):
    vmin, vmax = min(values), max(values)
    assert vmin != vmax, f'Redundant value set, {vmin} = {vmax}'

    step = (vmax - vmin) / bins
    nonzero = vmin if vmin != 0 else vmax
    decimals = len(str(nonzero).split('.')[1]) if '.' in str(nonzero) else 0
    steps = [round(vmin + b * step, decimals) for b in range(bins + 1)]
    quants = [round((steps[b] + steps[b + 1]) / 2, decimals) for b in range(bins)]

    return steps, quants


def equal_quantization(values:list|np.ndarray, bins:int):
    assert bins > 0 and (n := len(values)) > 0
    counts = Counter(values)

    if len((uniques := sorted(counts))) < bins:
        raise ValueError(f"{len(uniques)} unique values < {bins} bins")

    targets = [round(b * n / bins) for b in range(1, bins)]
    ranges, quants = [uniques[0]], []
    start, cumulative, target_idx, last_boundary = 0, 0, 0, -1

    for last_idx, value in enumerate(uniques):
        cumulative += counts[value]

        if target_idx < len(targets) and cumulative >= targets[target_idx] and last_idx > last_boundary:
            ranges.append(value)

            vals = uniques[start:last_idx + 1]
            quants.append(sum(v * counts[v] for v in vals) / sum(counts[v] for v in vals))

            start = last_idx + 1
            last_boundary = last_idx

            while target_idx < len(targets) and cumulative >= targets[target_idx]:
                target_idx += 1

    ranges.append(uniques[-1])

    if start < len(uniques):
        vals = uniques[start:]
        quants.append(sum(v * counts[v] for v in vals) / sum(counts[v] for v in vals))


    return ranges, quants
