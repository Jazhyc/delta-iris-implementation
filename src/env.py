import gymnax, craftax

def make_gym_env(env_name):
    """Create a Gym environment from a given name."""
    try:
        env = gymnax.make(env_name)
        return env
    except Exception as e:
        raise ValueError(f"Failed to create environment '{env_name}': {e}")