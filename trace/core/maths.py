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


def same_point(p:np.ndarray, q:np.ndarray, tolerance:float=0.01):
    p, q = np.array(p), np.array(q)
    return np.linalg.norm(p - q) < tolerance




def all_ints(lst:list):
    return all(isinstance(x, int) for x in lst)


def network_edges(start:np.ndarray, nodes:np.ndarray):
    return [[start, node] for node in nodes if not same_point(start, node)]


