import numpy as np

from trace.behavior.conditioning import EmpiricalDistribution

def distance_matrix(models: list[EmpiricalDistribution], metric :str='kl', smoothing:bool=False, norm:bool=True, lam:float=0.5):
    assert metric in ('frobenius', 'wasserstein' ,'agreement', 'kl'), f'Metric {metric} not supported'
    assert [m.metadata == models[0].metadata for m in models], f'Metadata mismatch'
    dist_mat = np.ones((len(models), len(models)))

    for i in range(len(models)):
        if i % 100 == 0: print(f'{i}/{len(models)}')
        vi = set(models[i].visited())
        dist_mat[i, i] = 0

        for j in range(i+1, len(models)):
            assert models[i].mask == models[j].mask, f'State feature mask mismatch'
            vj = set(models[j].visited())
            if not (overlap := vi & vj): continue

            d, union = 0.0, vi|vj
            if metric == 'frobenius':
                mat_i = [models[i].probs(s) for s in union]
                mat_j = [models[j].probs(s) for s in union]
                d = frobenius(mat_i, mat_j)
            elif metric == 'wasserstein':
                action_mapping = models[0].metadata['action_mapping'].values()
                cost = l2_cost(list(action_mapping))
                for s in union:
                    d += wasserstein(models[i].probs(s), models[j].probs(s), cost)
                d /= len(union)
            elif metric == 'agreement':
                for s in union:
                    ai, aj = models[i].act(s), models[j].act(s)
                    d += int(s in vi and s in vj and ai == aj)
                d /= len(overlap)
            elif metric == 'kl':
                for s in union: d += kl(models[i].probs(s), models[j].probs(s))
                d /= len(vi)+len(vj)

            dist_mat[i, j] = dist_mat[j, i] = d
    if norm: dist_mat = (dist_mat - dist_mat.min()) / (dist_mat.max() - dist_mat.min())
    if smoothing: dist_mat = np.exp(-lam * (1-dist_mat))

    return dist_mat


def wasserstein(dist1:list|np.ndarray, dist2:list|np.ndarray, cost:list|np.ndarray, eps:float=0.5, max_iter:int=1000, tol:float=1e-9):
    dist1, dist2, cost = np.asarray(dist1), np.asarray(dist2), np.asarray(cost)
    kernel = np.exp(-cost / eps)
    u, v = np.ones(len(dist1)), np.ones(len(dist1))

    for _ in range(max_iter):
        u_prev = u.copy()

        u = dist1 / (kernel @ v + 1e-12)
        v = dist2 / (kernel.T @ u + 1e-12)

        if np.linalg.norm(u - u_prev, 1) < tol: break
    return np.sum(np.outer(u, v) * kernel * cost)


def l2_cost(c1:list|np.ndarray, c2:list|np.ndarray|None=None):
    if c2 is None: c2 = c1
    c1, c2 = np.asarray(c1), np.asarray(c2)

    if c1.ndim == c2.ndim == 1:
        return np.linalg.norm(c1 - c2)
    return np.array([[np.linalg.norm(i - j) for i in c1] for j in c2])


def frobenius(mat1:list|np.ndarray, mat2:list|np.ndarray):
    mat1, mat2 = np.asarray(mat1), np.asarray(mat2)
    assert mat1.shape == mat2.shape, f'{mat1.shape} != {mat2.shape}'

    diff = mat1 - mat2
    return np.sqrt(np.sum(diff * diff))


def kl(dist1:list|np.ndarray, dist2:list|np.ndarray, eps:float=1e-12):
    dist1, dist2 = np.asarray(dist1), np.asarray(dist2)
    assert dist1.shape == dist2.shape, f'{dist1.shape} != {dist2.shape}'

    dist1 = dist1 / (np.sum(dist1) + eps)
    dist2 = dist2 / (np.sum(dist2) + eps)

    return np.sum(dist1 * np.log((dist1 + eps) / (dist2 + eps)))
