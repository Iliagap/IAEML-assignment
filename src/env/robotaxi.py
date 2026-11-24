"""Assignment 1: JAX robo-taxi environment for the InnoRL starter project."""
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp

from env.base import BaseEnv, BaseEnvState, RADIUS

# Proposed simulation parameters (world units, seconds, radians).
WHEELBASE = 0.6
MAX_SPEED = 6.0
MAX_ACCEL = 3.0
MAX_BRAKE = 5.0
MAX_STEER = 0.55
ROLLING = 0.15
DRAG = 0.05


class RoboTaxiState(BaseEnvState):
    speed: jax.Array
    heading: jax.Array


class RoboTaxiObservation(eqx.Module):
    distance_to_path: jax.Array
    direction_of_path: jax.Array
    collision_rays: jax.Array
    goal_relative: jax.Array
    speed: jax.Array
    perception_radius: float = eqx.field(static=True)

    def as_vector(self):
        return jnp.concatenate([
            jnp.atleast_1d(self.distance_to_path / self.perception_radius),
            self.direction_of_path,
            self.collision_rays.reshape(-1) / self.perception_radius,
            self.goal_relative,
            jnp.atleast_1d(self.speed / MAX_SPEED),
        ])


class RoboTaxiEnv(BaseEnv):
    """Action: two normalized values in [-1, 1]: throttle/brake, steering."""

    @staticmethod
    def init_params(key, map_id, max_steps, path_length, **kwargs):
        params, base = BaseEnv.init_params(
            key, map_id, max_steps, path_length, **kwargs
        )
        heading = jnp.arctan2(base.agent_forward_dir[1], base.agent_forward_dir[0])
        state = RoboTaxiState(
            time=base.time, goal_pos=base.goal_pos,
            agent_pos=base.agent_pos,
            agent_forward_dir=jnp.stack([jnp.cos(heading), jnp.sin(heading)]),
            static_obstacles=base.static_obstacles,
            kinematic_obstacles=base.kinematic_obstacles,
            kinematic_obst_velocities=base.kinematic_obst_velocities,
            path_array=base.path_array,
            speed=jnp.zeros((), dtype=jnp.float32), heading=heading,
        )
        return params, state

    @staticmethod
    @partial(jax.jit, static_argnames=('env_params',))
    def reset(key, env_params, init_state):
        return RoboTaxiEnv.get_observation(init_state, env_params), init_state

    @staticmethod
    def get_observation(env_state, env_params):
        s = env_state
        forward = s.agent_forward_dir
        left = jnp.stack([-forward[1], forward[0]])
        def local(vec):
            return jnp.stack([jnp.dot(vec, forward), jnp.dot(vec, left)])

        # Closest point on any path segment, including repeated endpoints.
        starts = s.path_array[:-1]
        segments = s.path_array[1:] - starts
        lengths_sq = jnp.sum(segments ** 2, axis=1)
        if starts.shape[0] > 0:
            t = jnp.clip(jnp.sum((s.agent_pos - starts) * segments, axis=1)
                         / jnp.maximum(lengths_sq, 1e-8), 0., 1.)
            closest = starts + t[:, None] * segments
            distances = jnp.linalg.norm(closest - s.agent_pos, axis=1)
            idx = jnp.argmin(jnp.where(lengths_sq > 1e-8, distances, jnp.inf))
            valid = jnp.any(lengths_sq > 1e-8)
            path_vec = jnp.where(valid, segments[idx], s.goal_pos - s.agent_pos)
            path_distance = jnp.where(valid, distances[idx],
                                       jnp.linalg.norm(s.goal_pos - s.agent_pos))
        else:
            path_vec = s.goal_pos - s.agent_pos
            path_distance = jnp.linalg.norm(path_vec)
        path_dir = path_vec / jnp.maximum(jnp.linalg.norm(path_vec), 1e-8)
        obstacles = jnp.concatenate([s.static_obstacles, s.kinematic_obstacles], axis=0)
        rays = BaseEnv._generate_rays(forward, env_params.num_ray_sensors, env_params.agent_FOV)
        if obstacles.shape[0] > 0:
            distances = BaseEnv._ray_circle_intersections(s.agent_pos, rays, obstacles, RADIUS)
            distances = jnp.clip(distances, 0., env_params.perception_radius)
        else:
            distances = jnp.full((rays.shape[0],), env_params.perception_radius)
        ray_vectors = rays * distances[:, None]
        h, w = env_params.map_height_width
        goal_relative = local(s.goal_pos - s.agent_pos) / jnp.hypot(float(h), float(w))
        return RoboTaxiObservation(path_distance, local(path_dir), ray_vectors,
                                   goal_relative, s.speed, env_params.perception_radius)

    @staticmethod
    @partial(jax.jit, static_argnames=('env_params',))
    def step(key, env_state, action, env_params):
        s = env_state
        dt = env_params.step_size
        action = jnp.clip(jnp.asarray(action, dtype=jnp.float32), -1., 1.)
        throttle, steer = action[0], action[1] * MAX_STEER
        acceleration = jnp.where(throttle >= 0., throttle * MAX_ACCEL, throttle * MAX_BRAKE)
        resistance = ROLLING * jnp.tanh(s.speed / 0.1) + DRAG * s.speed * jnp.abs(s.speed)
        speed = jnp.clip(s.speed + dt * (acceleration - resistance), 0., MAX_SPEED)
        mean_speed = (s.speed + speed) / 2.
        yaw_change = dt * mean_speed * jnp.tan(steer) / WHEELBASE
        mid_heading = s.heading + yaw_change / 2.
        pos = s.agent_pos + dt * mean_speed * jnp.stack([jnp.cos(mid_heading), jnp.sin(mid_heading)])
        heading = (s.heading + yaw_change + jnp.pi) % (2 * jnp.pi) - jnp.pi
        forward = jnp.stack([jnp.cos(heading), jnp.sin(heading)])
        moving = BaseEnv._move_kinematic_obstacles(s, env_params)
        obstacles = jnp.concatenate([s.static_obstacles, moving], axis=0)
        time = s.time + 1

        # Match the starter environment collision convention (center distance < RADIUS).
        # A 2R threshold would make the original map spawn overlap the walls.
        collision = BaseEnv._check_collisions(pos, obstacles)
        h, w = env_params.map_height_width
        outside = jnp.any(pos < 0.) | (pos[0] > w - 1) | (pos[1] > h - 1)
        failed = collision | outside
        reached = BaseEnv._check_goal(pos, s.goal_pos) & ~failed
        timeout = time >= env_params.max_steps_in_episode
        done = failed | reached | timeout

        # Reward progress and penalize time, steering effort and unsafe endings.
        progress = jnp.linalg.norm(s.goal_pos - s.agent_pos) - jnp.linalg.norm(s.goal_pos - pos)
        reward = 2. * progress - 0.01 - 0.005 * action[1] ** 2
        reward = reward + jnp.where(reached, 20., 0.) - jnp.where(failed, 20., 0.)

        # Avoid planning from invalid positions; replan once per second.
        replan = (time % env_params.fps == 0) & ~done
        path = jax.lax.cond(replan,
            lambda _: BaseEnv._find_path(pos, s.goal_pos, obstacles, env_params),
            lambda _: s.path_array, operand=None)
        new_state = RoboTaxiState(time=time, goal_pos=s.goal_pos, agent_pos=pos,
            agent_forward_dir=forward, static_obstacles=s.static_obstacles,
            kinematic_obstacles=moving, kinematic_obst_velocities=s.kinematic_obst_velocities,
            path_array=path, speed=speed, heading=heading)
        obs = RoboTaxiEnv.get_observation(new_state, env_params)
        info = {'time': time, 'distance_to_path': obs.distance_to_path,
                'direction_of_path': obs.direction_of_path, 'collision': collision,
                'out_of_bounds': outside, 'goal_reached': reached,
                'timeout': timeout, 'speed': speed}
        return obs, new_state, reward, done, info


RobotaxiEnv = RoboTaxiEnv
