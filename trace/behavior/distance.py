import numpy as np

from trace.behavior.conditioning import EmpiricalDistribution

def distance_matrix(models: list[EmpiricalDistribution], metric :str='kl', norm:bool=True, lam:float=0.5):
    assert metric in ('frobenius', 'wasserstein', 'kl'), f'Metric {metric} not supported'
    assert [m.metadata == models[0].metadata for m in models], f'Metadata mismatch'
    dist_mat = np.zeros((len(models), len(models)))

    for i in range(len(models)):
        if i % 100 == 0: print(f'{i}/{len(models)}')

        for j in range(i+1,len(models)):
            assert models[i].mask == models[j].mask, f'State feature mask mismatch'

            if metric == 'kl':
                d1, d2 = kl(models[i], models[j]), kl(models[j], models[i])
                dist_mat[i, j]= dist_mat[j, i] = max(d1, d2)

            elif metric == 'frobenius':
                dist_mat[i, j] = frobenius(models[i], models[j])

            elif metric == 'wasserstein':
                dist_mat[i, j] = wasserstein(models[i], models[j])

            else: raise NotImplementedError(f'Metric {metric} not supported')

    return (dist_mat - dist_mat.min()) / (dist_mat.max() - dist_mat.min()) if norm else dist_mat


def kl(model1:EmpiricalDistribution, model2:EmpiricalDistribution, eps:float=1e-12):
    v1, v2 = set(model1.visited()), set(model2.visited())
    d, union, overlap = 0, v1|v2, v1&v2

    for s in union:
        p1, p2 = model1.probs(s), model2.probs(s)
        assert p1.shape == p2.shape, f'{p1.shape} != {p2.shape}'

        p1 = p1 / (np.sum(p1) + eps)
        p2 = p2 / (np.sum(p2) + eps)

        d += np.sum(p1 * np.log((p1 + eps) / (p2 + eps)))
    return  d / (len(v1)+len(v2))


def frobenius(model1: EmpiricalDistribution, model2: EmpiricalDistribution):
    v1, v2 = set(model1.visited()), set(model2.visited())
    d, union = 0, v1 | v2
    mat1 = np.array([model1.probs(s) for s in union])
    mat2 = np.array([model2.probs(s) for s in union])
    diff = mat1 - mat2
    return np.sqrt(np.sum(diff * diff))


def wasserstein(model1: EmpiricalDistribution, model2: EmpiricalDistribution, eps: float | int = 0.5,
                max_iter: int = 1000, tol: float = 1e-9):
    cost = l2_cost(model1.metadata)

    v1, v2 = set(model1.visited()), set(model2.visited())
    d, union = 0, v1 | v2
    for s in union:
        p1, p2 = model1.probs(s), model2.probs(s)

        kernel = np.exp(-cost / eps)
        u, v = np.ones(len(p1)), np.ones(len(p2))

        for _ in range(max_iter):
            u_prev = u.copy()

            u = p1 / (kernel @ v + 1e-12)
            v = p2 / (kernel.T @ u + 1e-12)

            if np.linalg.norm(u - u_prev, 1) < tol: break
        d += np.sum(np.outer(u, v) * kernel * cost)
    return d / (len(v1) + len(v2))


def l2_cost(c1:list|np.ndarray|dict, c2:list|np.ndarray|None=None):
    if isinstance(c1, dict): c1 = list(c1['action_mapping'].values())
    if c2 is None: c2 = c1
    c1, c2 = np.asarray(c1), np.asarray(c2)

    if c1.ndim == c2.ndim == 1:
        return np.linalg.norm(c1 - c2)
    return np.array([[np.linalg.norm(i - j) for i in c1] for j in c2])