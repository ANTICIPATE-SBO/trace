import numpy as np

from gymnasium import Env, Wrapper, core
import mo_gymnasium.envs.minecart.minecart as minecart

from trace.core.maths import point_dist, seg_angle, seg_proj, angle_subtr, same_point, network_edges


class MinecartTrainWrapper(Wrapper):
    def __init__(self, env:Env[core.ObsType, core.ActType], tolerance:float=0.01):
        super().__init__(env)
        self.tolerance = tolerance
        self.env.unwrapped.frame_skip = 1

        self.trails = []
        self.trail_angles = []
        self.current_trail = [np.array([0,0]), np.array([0,0])]
        self.current_angle = 0
        self.previous_content = [0., 0.]
        self.on_trail = False

        self.base = np.array(minecart.HOME_POS)
        self.mines = [mine for mine in env.unwrapped.mines]
        self.capacity = self.env.unwrapped.capacity


    def _make_trails(self, start):
        nodes = [mine.pos for mine in self.mines] + [self.base]
        candidate_trails = network_edges(start, nodes)
        candidate_angles = [seg_angle(*trail) for trail in candidate_trails]

        self.trails, self.trail_angles = [], []
        for i in range(len(candidate_trails)):
            dominated = False
            (si, ei), ai = candidate_trails[i], candidate_angles[i]

            for j in range(len(candidate_trails)):
                (sj, ej), aj = candidate_trails[j], candidate_angles[j]
                if same_direction(ai, aj) and point_dist(sj, ej) > point_dist(si, ei):
                    dominated = True
                    break

            if not dominated:
                self.trails.append([si, ei])
                self.trail_angles.append(ai)


    def step(self, action):
        spin_allowed = self.at_node() is not None
        if not spin_allowed: self._lock_orientation()

        if self.on_trail and spin_allowed:
            self.on_trail = False

        elif self.on_trail and not spin_allowed:
            if action in (minecart.ACT_LEFT, minecart.ACT_RIGHT):
                action = minecart.ACT_NONE

        elif not self.on_trail and spin_allowed:
            if action == minecart.ACT_ACCEL:
                self._pick_trail()
                self.on_trail = True

        obs, reward, terminated, truncated, info = self.env.step(action)
        if action == minecart.ACT_MINE: self._deterministic_reward()

        new_pos = seg_proj(self.get_pos(), *self.current_trail)
        self.env.unwrapped.cart.pos, obs[:2]= new_pos, new_pos

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_trail = [np.array([0, 0]), np.array([0, 0])]
        self.on_trail, self.previous_content = False, [0., 0.]
        return obs, info

    def _deterministic_reward(self):
        mine = self.at_node()
        if not hasattr(mine, 'pos'): return

        mined = np.array([d.mean() for d in mine.distributions])
        cart_free = self.capacity - np.sum(self.previous_content)

        if (total_mined := np.sum(mined)) > cart_free:
            mined *= cart_free / total_mined
        self.env.unwrapped.cart.content = self.previous_content + mined
        self.previous_content += mined


    def _pick_trail(self):
        pos, angle, point = self.get_pos(), self.get_angle(), self.at_node()
        assert point is not None, f'Point not found at {pos}'

        if same_point(self.current_trail[0], pos) and angle == seg_angle(*self.current_trail): return
        elif not (same_point(self.current_trail[1], pos) or same_point(self.current_trail[0], pos)):
            self.current_trail[1] = point
        self._make_trails(point.pos if hasattr(point, 'pos') else point)

        next_trail, next_angle, best_angle_dif = None, 0, float('inf')
        for trail, trail_angle in zip(self.trails, self.trail_angles):

            if (angle_diff := abs(angle_subtr(angle, trail_angle))) < best_angle_dif:
                next_trail, next_angle, best_angle_dif = trail, trail_angle, angle_diff

        assert next_trail is not None, f'Trail not found for {self.current_trail}'
        self.current_trail, self.current_angle = next_trail, next_angle

    def _lock_orientation(self):
        self.env.unwrapped.cart.angle = self.current_angle

    def at_node(self):
        if same_point(self.get_pos(), self.base, self.tolerance): return self.base

        for mine in self.mines:
            if same_point(mine.pos, self.get_pos(), self.tolerance):
                return mine
        return None

    def get_angle(self):
        return self.env.unwrapped.cart.angle

    def get_pos(self):
        return self.env.unwrapped.cart.pos

    def get_speed(self):
        return self.env.unwrapped.cart.speed


def same_direction(ang1:float, ang2:float, tolerance:float=0.01):
    forward = abs(angle_subtr(ang1, ang2)) < tolerance
    backward = abs(angle_subtr(ang1 + 180, ang2)) < tolerance
    return forward or backward

