import numpy as np


def point_dist(a:np.ndarray, b:np.ndarray):
    return np.linalg.norm(np.array(a) - np.array(b))


def seg_angle(a, b):
    a, b = np.array(a), np.array(b)
    dx, dy = b - a
    angle = np.arctan2(dy, dx)
    return (np.degrees(angle) + 360) % 360


def angle_subtr(a:float, b:float):
    return (180 + b - a) % 360 - 180


def seg_proj(point, a, b, eps:float=1e-8):
    point, a, b = np.array(point), np.array(a), np.array(b)
    ab, ap = b - a, point - a

    if (ab_norm := np.linalg.norm(ab)) == 0: return a.copy()

    t = np.dot(ap, ab) / (ab_norm ** 2 + eps)
    t = max(0.0, min(1.0, t))
    return a + t * ab


def counter_clockwise(a:np.ndarray, b:np.ndarray, c:np.ndarray):
    ax, ay = a
    bx, by = b
    cx, cy = c
    return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)


def seg_intercept(a:np.ndarray, b:np.ndarray, c:np.ndarray, d:np.ndarray):
    if same_point(a,c): return same_point(b,d)
    if same_point(a,d): return same_point(b,c)
    if same_point(b,c): return same_point(a,d)
    if same_point(b,d): return same_point(a,c)

    forward = counter_clockwise(a, c, d) != counter_clockwise(b, c, d)
    backward = counter_clockwise(a, b, c) != counter_clockwise(a, b, d)
    return forward and backward


def same_point(p:np.ndarray, q:np.ndarray, tolerance:float=0.01):
    p, q = np.array(p), np.array(q)
    return np.linalg.norm(p - q) < tolerance


def same_direction(ang1:float, ang2:float, tolerance:float=0.01):
    forward = abs(angle_subtr(ang1, ang2)) < tolerance
    backward = abs(angle_subtr(ang1 + 180, ang2)) < tolerance
    return forward or backward


def is_connected(current_trail:list, new_trail:list):
    if same_point(current_trail[1], new_trail[0]):
        return True
    return False


def discount(ar: np.ndarray, gamma: float=0.99):
    if not isinstance(ar, np.ndarray): ar = np.array(ar)
    discounts = gamma ** np.arange(ar.shape[0], dtype=np.float32)
    return np.sum(ar * discounts[:, None], axis=0)


def all_ints(lst:list):
    return all(isinstance(x, int) for x in lst)


def network_edges(start:np.ndarray, nodes:np.ndarray):
    return [[start, node] for node in nodes if not same_point(start, node)]


def pareto_filter(acc_rewards: np.ndarray):
    is_pareto = np.ones(len(acc_rewards), dtype=bool)

    for i in range(acc_rewards.shape[0]):
        if not is_pareto[i]: continue

        for j in range(len(acc_rewards)):
            if i == j: continue
            ri, rj = acc_rewards[i], acc_rewards[j]

            if np.all(rj >= ri) and np.any(rj > ri):
                is_pareto[i] = False
                break

    return np.where(is_pareto)[0]