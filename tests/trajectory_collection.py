import os
os.chdir('..')

import numpy as np
np.set_printoptions(suppress=True, precision=2)


from trace.core import TrajectoryManager
from trace.policies import initialize_setting, dst_gt, minetrain_gt, IPRO


def ground_truth(manager:TrajectoryManager, save:bool=True):
    assert len(manager) == 0. , 'Input trajectory manager is not empty'
    env_id, file_prefix = manager.metadata['env_id'], manager.metadata['file_prefix']

    if 'minetrain' in file_prefix:
        _, env = initialize_setting(env_id=env_id, minetrain=True)
        manager.load(minetrain_gt(env))

    elif 'deep-sea-treasure' in env_id:
        _, env = initialize_setting(env_id=env_id, minetrain=True)
        sea_map, action_mapping = env.unwrapped.sea_map, manager.metadata['action_mapping']
        manager.load(dst_gt(sea_map, action_mapping))

    else:
        raise NotImplementedError(f'Ground truth not implemented for {env_id}')

    if save: manager.save(f"data/{file_prefix}_ground_truth.json")
    pareto_points = np.unique(manager.accrue(), axis=0)
    print('Pareto points:')
    for point in pareto_points: print(point)
    return manager


def ipro_samples(manager:TrajectoryManager, iter_steps:int, save:bool=True, verbose:bool=True):
    assert len(manager) == 0. , 'Input trajectory manager is not empty'
    minetrain = 'minetrain' in manager.metadata['file_prefix']
    env, eval_env = initialize_setting(env_id=manager.metadata['env_id'], minetrain=minetrain)

    ipro = IPRO(
        env=env,
        direction="maximize",
        tolerance=float(manager.metadata['tolerance']),
        max_iterations=None,
        iter_total_timesteps=iter_steps,
        num_steps=manager.metadata['num_steps'],
        log=False,
        gamma=manager.metadata['gamma'],
        aug=0.2,
        reset_agent=True
    )
    if verbose: print('Initialized algorithm')

    pareto_set = ipro.train(
        eval_env=eval_env,
        ref_point=manager.metadata['ref_point'],
        deterministic=True,
        eval_episodes=manager.metadata['eval_episodes'],
        extrema=(manager.metadata['nadir'], manager.metadata['ideal']),
    )

    manager.load([traj for _, _, traj in pareto_set], split=False)
    if save: manager.save(f"data/{manager.metadata['file_prefix']}_ipro.json")
    return manager



if __name__ == '__main__':
    #ipro_manager = ipro_samples(TrajectoryManager('dst-conc'), iter_steps=500_000)
    #ipro_manager = TrajectoryManager('dst-conc').load('ipro', pareto=True)
    #obs, acs, rew = ipro_manager.conditioning_features(per_point=True)
    #print('Example point: ', obs[-1])
    #print('Length of points: ', [len(point) for point in obs])
    #print('Pareto points:')
    #for r in rew: print(r)
    gt_manager = ground_truth(TrajectoryManager('minetrain'))
    print('Number of trajectories: ', len(gt_manager))
    #obs, acs, _ = gt_manager.conditioning_features(per_point=False)
