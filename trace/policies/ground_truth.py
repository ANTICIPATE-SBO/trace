import numpy as np
from random import choices
from mo_gymnasium.envs.minecart import minecart as minecart

from trace.core import seg_angle, angle_subtr, point_dist, same_point
from trace.policies.auxiliary import (reachable_steps, monotonic_paths, perform_action, visit_combinations,
                                      mine_steps, best_acc_steps, available_ores)
from trace.policies.minetrain import MinecartTrainWrapper

def dst_gt(sea_map: np.ndarray, action_mapping: dict, start: tuple = (0,0)):
    steps = reachable_steps(sea_map, action_mapping, start)
    treasure_cells = np.argwhere(sea_map > 0)
    ground_truth = []

    for pos in treasure_cells:
        if np.isinf(steps[*pos]): continue
        ground_truth.extend([
            [{"observations": obs, "actions": acs, "rewards": rews}]
            for obs, acs, rews in monotonic_paths(sea_map, steps, pos, action_mapping, start)
        ])
    return ground_truth


def minetrain_gt(env:MinecartTrainWrapper):
    mines, capacity, base = env.mines, env.capacity, env.base
    return [
        [path_playout(env, path, steps)]
        for path in visit_combinations(mines, base)
        for steps in mine_steps(available_ores(path, mines), capacity)
    ]


def synthetic_stochastic_points(ground_truth: list, num: int=100, length: int=10):
    episodes = [episode for point in ground_truth for episode in point]
    return [choices(episodes, k=length) for _ in range(num)]


def path_playout(env:MinecartTrainWrapper, path:list, steps:list):
    obs, _ = env.reset()
    trajectory, terminated = {'observations': [obs.tolist()], 'actions': [], 'rewards': []}, False

    for e in range(len(path)-1):
        start, end = path[e], path[e+1]
        trail_angle = seg_angle(start, end)
        rot_action = minecart.ACT_LEFT if env.get_angle() - trail_angle > 0 else minecart.ACT_RIGHT
        while not -10 < abs(angle_subtr(env.get_angle(), trail_angle)) < 10:
            env, terminated = perform_action(env, rot_action, trajectory)
            if terminated: break
        if terminated: break

        for _ in range(best_acc_steps(point_dist(start, end))):
            env, terminated = perform_action(env, minecart.ACT_ACCEL, trajectory)
            if terminated or same_point(env.get_pos(), end): break
        if terminated: break

        while not same_point(env.get_pos(), end):
            env, terminated = perform_action(env, minecart.ACT_NONE, trajectory)
            if terminated: break
        if terminated: break
        env, terminated = perform_action(env, minecart.ACT_BRAKE, trajectory)
        if terminated: break

        for i in range(steps[e]):
            env, terminated = perform_action(env, minecart.ACT_MINE, trajectory)
            if terminated: break
    return trajectory
