from collections import deque
from itertools import combinations, permutations, product
from math import ceil

import numpy as np

from core import same_point
from trace.core import point_dist as point_dist
from trace.policies.minetrain import MinecartTrainWrapper


def reachable_steps(sea_map: np.ndarray, action_mapping:dict, start:tuple=(0,0)):
    h, w = sea_map.shape
    steps = np.full((h, w), np.inf)
    steps[start] = 0
    q = deque([start])

    while q:
        r, c = q.popleft()
        if sea_map[r, c] > 0: continue
        for dr, dc in action_mapping.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if sea_map[nr, nc] != -10 :
                    if steps[nr, nc] > steps[r, c] + 1:
                        steps[nr, nc] = steps[r, c] + 1
                        q.append((nr, nc))
    return steps


def monotonic_paths(sea_map:np.ndarray, steps:np.ndarray, target:tuple, action_mapping:dict, start:tuple=(0,0)):
    h, w = sea_map.shape
    paths = []

    def dfs(pos:tuple, observations, actions):
        if pos == target:
            r, c = pos
            rewards = [[0, -1]]*(steps[r, c] - 1) + [[sea_map[r, c], -1]]
            paths.append((observations.copy(), actions.copy(), rewards))
            return
        elif sea_map[*pos] > 0: return

        r, c = pos
        for a, (dr, dc) in action_mapping.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if sea_map[nr, nc] != -10.:
                    if steps[nr, nc] == steps[r, c] + 1:
                        observations.append((nr, nc))
                        actions.append(a)
                        dfs((nr, nc), observations, actions)
                        observations.pop()
                        actions.pop()

    dfs(start, [start], [])
    return paths


def visit_combinations(mines: list|np.ndarray, base: list|np.ndarray):
    mines, base = [np.array(mine.pos) for mine in mines], tuple(base)

    all_paths = []
    for path_len in range(1, len(mines) + 1):
        for path in combinations(mines, path_len):
            path = best_visit_order(list(path))
            all_paths.append([base, *path, base])
            if len(path)>1: all_paths.append([base, *path[::-1], base])

    return all_paths


def best_visit_order(path:list):
    best_order, best_len = [], float('inf')

    for perm in permutations(path):
        length = sum(point_dist(start, end) for start, end in zip(path, path[1:]))
        if length < best_len:
            best_order, best_len = list(perm), length

    return best_order


def best_acc_steps(trail_length:float, speed_acc:float=0.0075, fuel_acc:float=0.025, fuel_idle:float=0.005):
    best_fuel, best_steps = float('inf'), 0
    max_acc_steps = ceil((-1 + np.sqrt(1 + 8 * trail_length / speed_acc)) / 2)

    for acc_steps in range(1, max_acc_steps + 1):
        vel = acc_steps * speed_acc
        d_acc = speed_acc * acc_steps * (acc_steps + 1) / 2

        if d_acc >= trail_length: # end reached while accelerating
            if acc_steps * (fuel_acc + fuel_idle) < best_fuel:
                return acc_steps

        d_rem = trail_length - d_acc
        idle_steps = ceil(d_rem / vel)
        fuel = acc_steps * (fuel_acc + fuel_idle) + idle_steps * fuel_idle

        if fuel < best_fuel:
            best_fuel, best_steps = fuel, acc_steps

    return best_steps


def ore_limits(ore_totals:list, capacity:float=1.5):
    max_steps = []
    for total in ore_totals:
        if total == 0: max_steps.append(1)
        else:
            max_extra = int(ceil(capacity / total))
            max_steps.append(1 + max_extra)

    return max_steps


def mine_steps(ores: np.ndarray, capacity: float = 1.5):
    ores = np.array(ores)
    ore_totals = ores.sum(axis=1)
    if ore_totals.sum() > capacity: return []

    valid_steps, max_steps = [], ore_limits(ore_totals, capacity)
    for steps in product(*[range(1, m + 1) for m in max_steps]):
        content, action_ind, valid = 0.0, 0, True

        for ore_idx, mine_count in enumerate(steps):
            for _ in range(mine_count):
                action_ind += 1

                last_action = action_ind == sum(steps)
                overflow = content + ore_totals[ore_idx] > capacity
                full = content >= capacity
                if full if last_action else overflow:
                    valid = False
                    break
                content += ore_totals[ore_idx]

            if not valid: break
        if valid: valid_steps.append(list(steps) + [0])

    return valid_steps


def perform_action(env:MinecartTrainWrapper, action:int, trajectory:dict):
    observations, reward, terminated, truncated, _ = env.step(action)
    trajectory['actions'].append(int(action))
    trajectory['observations'].append(observations.tolist())
    trajectory['rewards'].append(np.asarray(reward).tolist())

    return env, terminated or truncated


def available_ores(path:list, mines:list):
    ores = []
    for node in path[1:-1]:
        for mine in mines:
            if same_point(node, mine.pos):
                ores.append([d.mean() for d in mine.distributions])
    return ores
