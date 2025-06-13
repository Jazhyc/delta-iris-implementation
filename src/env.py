import gymnax
from craftax.craftax_env import make_craftax_env_from_name
from gymnax.wrappers.gym import GymnaxToVectorGymWrapper

def make_env(env_name, num_envs=1, seed=None):
    """Create an environment based on the provided name."""
    if "craftax" in env_name.lower():
        env, env_params = make_craftax_env(env_name)
    else:
        env, env_params = make_gym_env(env_name)
    
    seed = seed if seed is not None else 0
    wrapped_env = GymnaxToVectorGymWrapper(env, num_envs=num_envs, params=env_params)
    return wrapped_env, env_params

def make_gym_env(env_name):
    """Create a Gym environment from a given name."""
    env, env_params = gymnax.make(env_name)
    return env, env_params

def make_craftax_env(env_name):
    """Create a Craftax environment from a given name."""
    env = make_craftax_env_from_name(env_name, auto_reset=True)
    env_params = env.default_params
    return env, env_params