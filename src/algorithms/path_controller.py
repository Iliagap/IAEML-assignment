"""Pure-pursuit baseline for starter map 0 (not a learned policy).
Use WHEELBASE = 0.6 in robotaxi.py and the starter collision convention.
The reference route is hand-designed for map 0, not the base DP output.
"""
import numpy as np
import jax.numpy as jnp
from env.robotaxi import WHEELBASE, MAX_STEER, MAX_ACCEL, MAX_BRAKE, ROLLING, DRAG


class PathController:
    def __init__(self):
        waypoints = np.array([
            [2, 1], [2, 4.8], [2.25, 5.65], [3, 6.4],
            [4, 6.5], [7, 6.5], [11, 6.5], [13, 5.5],
            [15.5, 2.5], [16, 1],
        ], dtype=float)
        self.path = np.concatenate([
            np.linspace(a, b, max(2, int(np.linalg.norm(b-a)/0.05)), endpoint=False)
            for a, b in zip(waypoints[:-1], waypoints[1:])
        ] + [waypoints[-1:]])
        self.index = 0

    def action(self, state, params):
        pos = np.asarray(state.agent_pos, dtype=float)
        heading = float(state.heading)
        speed = float(state.speed)
        # Track the nearest forward reference point, never jump backwards.
        nearby = self.path[self.index:self.index+60]
        self.index += int(np.argmin(np.linalg.norm(nearby-pos, axis=1)))
        target = self.path[min(self.index+14, len(self.path)-1)]
        delta = target-pos
        angle = np.arctan2(delta[1], delta[0])-heading
        alpha = np.arctan2(np.sin(angle), np.cos(angle))
        steer = np.clip(np.arctan2(2*WHEELBASE*np.sin(alpha),
                                   max(np.linalg.norm(delta), 0.15)),
                        -MAX_STEER, MAX_STEER)
        desired = min(1.6, max(0.5, 1.6*(1-abs(steer)/0.8)))

        # Predict approaching moving obstacles over the next second.
        moving = np.asarray(state.kinematic_obstacles, dtype=float)
        velocities = np.asarray(state.kinematic_obst_velocities, dtype=float)
        if len(moving):
            t = np.linspace(0, 1, 11)
            forward = np.array([np.cos(heading), np.sin(heading)])
            future = pos + t[:, None]*max(speed, 0.6)*forward
            h, w = params.map_height_width
            obstacles = (moving[None, :, :] + t[:, None, None]*velocities[None, :, :]) % [w, h]
            clearance = np.min(np.linalg.norm(future[:, None, :]-obstacles, axis=-1))
            if clearance < 1.35:
                desired = 0.0

        # Proportional speed control with drag compensation.
        acceleration = np.clip(4*(desired-speed) + ROLLING*np.tanh(speed/0.1)
                               + DRAG*speed*abs(speed), -MAX_BRAKE, MAX_ACCEL)
        throttle = acceleration / (MAX_ACCEL if acceleration >= 0 else MAX_BRAKE)
        return jnp.asarray([throttle, steer/MAX_STEER], dtype=jnp.float32)
