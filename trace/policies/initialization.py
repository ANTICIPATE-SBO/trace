from mo_gymnasium import make
from mo_gymnasium.wrappers.vector import MOSyncVectorEnv

from trace.policies.minetrain import MinecartTrailWrapper

def make_env_factory(env_id: str, minetrain:bool=False):
    def _factory():
        env = make(env_id, max_episode_steps=1000)
        if minetrain: return MinecartTrailWrapper(env)
        else: return env
    return _factory


def initialize_setting(env_id: str|dict, minetrain:bool=False):
    env_id = env_id if isinstance(env_id, str) else env_id['env_id']
    env = MOSyncVectorEnv(iter([make_env_factory(env_id, minetrain=minetrain)]))
    eval_env = MinecartTrailWrapper(make(env_id, max_episode_steps=1000)) if minetrain else make(env_id, max_episode_steps=1000)
    return env, eval_env