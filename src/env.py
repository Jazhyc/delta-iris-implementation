import gymnax
import jax, torch

from functools import partial
from craftax.craftax_env import make_craftax_env_from_name
from gymnax.wrappers.gym import GymnaxToVectorGymWrapper

from gymnasium.wrappers.array_conversion import (
    array_conversion,
    module_namespace,
)

# Preferably this should be configurable but I'm not certain how to configure jax
device_jax = jax.devices()[0]
device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch_to_jax = partial(array_conversion, xp=module_namespace(jax.numpy), device=device_jax)
jax_to_torch = partial(array_conversion, xp=module_namespace(torch), device=device_torch)

class JaxToTorchWrapper:
    """Converts observations to PyTorch tensors and actions to JAX arrays."""
    def __init__(self, env):
        self.env = env

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        obs = jax_to_torch(obs)
        return obs, info

    def step(self, action):
        action = torch_to_jax(action)
        obs, reward, done, truncated, info = self.env.step(action)
        obs = jax_to_torch(obs)
        return obs, reward, done, truncated, info
    
    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

def make_env(env_name, num_envs=1, seed=None):
    """Create an environment based on the provided name."""
    if "craftax" in env_name.lower():
        env, env_params = make_craftax_env(env_name)
    else:
        env, env_params = make_gym_env(env_name)
    
    seed = seed if seed is not None else 0
    wrapped_vector_env = GymnaxToVectorGymWrapper(env, num_envs=num_envs, params=env_params)
    wrapped_vector_env = JaxToTorchWrapper(wrapped_vector_env)
    
    return wrapped_vector_env, env_params

def make_gym_env(env_name):
    """Create a Gym environment from a given name."""
    env, env_params = gymnax.make(env_name)
    return env, env_params

def make_craftax_env(env_name):
    """Create a Craftax environment from a given name."""
    env = make_craftax_env_from_name(env_name, auto_reset=True)
    env_params = env.default_params
    return env, env_params