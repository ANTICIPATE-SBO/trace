import os
import sys

from yaml import safe_load
from trace.policies import path_playout, initialize_setting, perform_action
from trace.core import TrajectoryManager
from trace.visuals import trajectory_grid

sys.path.insert(0, os.path.abspath('..'))
os.chdir('../')


def integration(path:list, steps:list):
    _, env = initialize_setting(safe_load(open('trace/configs/minetrain.yaml')), minetrain=True)
    trajectory = path_playout(env, path, steps)

    print(f'path: {path}, mine steps: {steps}')
    for o, a, r in zip(trajectory['observations'], trajectory['actions'], trajectory['rewards']):
        print('-' * 100)
        print('observation: ', o)
        print('action: ', a)
        print('reward: ', r)


def ipro_debug(sample_index:int=25):
    manager = TrajectoryManager('minetrain').load('ipro')
    trajectory_grid(manager.subset([i == sample_index for i in range(len(manager))]), alpha=1, color='red').show()

    _, env = initialize_setting(manager.metadata, minetrain=True)
    obs, _ = env.reset()

    trajectory, terminated = {'observations': [obs.tolist()], 'actions': [], 'rewards': []}, False
    action_sample = manager.sequence('actions', flatten=True)[sample_index]
    for a, action in enumerate(action_sample):
        env, terminated = perform_action(env, action, trajectory)

    manager = TrajectoryManager(manager.metadata).load([[trajectory]])
    trajectory_grid(manager, alpha=1, color='blue').show()

    print('Playout completed\n')


def dataset_overview():
    manager = TrajectoryManager('minetrain').load('ipro', duplicates=True, )
    trajectories = manager.trajectories
    for point in trajectories:
        for t in point:
            print(t.keys())

    print('points:', len(trajectories))
    print('trajectories', len(trajectories[0]))
    print('keys: ',len(trajectories[0][0]))
    new_manager = TrajectoryManager('minetrain').load(trajectories)

    obs, acs, rew = new_manager.conditioning_features()

    for i, r in enumerate(rew):
        print(i+1, r)




if __name__ == '__main__':
    dataset_overview()





