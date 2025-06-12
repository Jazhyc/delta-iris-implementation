import jax
import gymnax
import time
from tqdm import tqdm

key = jax.random.key(0)
key, key_reset, key_act, key_step = jax.random.split(key, 4)

# Instantiate the environment & its settings.
env, env_params = gymnax.make("Pendulum-v1")

# Reset the environment.
obs, state = env.reset(key_reset, env_params)

# Measure time for 100k steps
num_steps = 100_000
start_time = time.time()

for i in tqdm(range(num_steps), desc="Running steps"):
    
    # Sample a random action.
    key_act, key_step = jax.random.split(key_step, 2)
    action = env.action_space(env_params).sample(key_act)
    
    # Perform the step transition.
    n_obs, n_state, reward, done, _ = env.step(key_step, state, action, env_params)
    
    # Reset environment if episode is done
    if done:
        obs, state = env.reset(key_step, env_params)
    else:
        obs, state = n_obs, n_state

end_time = time.time()
total_time = end_time - start_time
steps_per_second = num_steps / total_time

print(f"Completed {num_steps:,} steps in {total_time:.2f} seconds")
print(f"Steps per second: {steps_per_second:.2f}")