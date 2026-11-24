import jax
import jax.numpy as jnp

from env.robotaxi import RoboTaxiEnv

key = jax.random.PRNGKey(42)

params, initial_state = RoboTaxiEnv.init_params(
    key=key,
    map_id=0,
    max_steps=1000,
    path_length=100,
)

obs, state = RoboTaxiEnv.reset(
    key, params, initial_state
)
obstacles = jnp.concatenate(
    [
        state.static_obstacles,
        state.kinematic_obstacles,
    ],
    axis=0,
)

distances = jnp.linalg.norm(
    obstacles - state.agent_pos,
    axis=1,
)

print("Nearest obstacle:", float(jnp.min(distances)))
print("Initial position:", state.agent_pos)
print("Observation:", obs.as_vector())
print("Observation shape:", obs.as_vector().shape)

for i in range(100):
    key, step_key = jax.random.split(key)

    action = jnp.array([0.8, 0.1])

    obs, state, reward, done, info = RoboTaxiEnv.step(
        step_key, state, action, params
    )

    print(
        f"Step: {i}, "
        f"Position: {state.agent_pos}, "
        f"Speed: {float(state.speed):.2f}, "
        f"Reward: {float(reward):.3f}"
    )

    if done:
        print("Episode finished:", info)
        break
