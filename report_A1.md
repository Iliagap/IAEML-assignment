# Assignment 1 — RoboTaxi Environment

**Course:** Introduction to Adaptive and Efficient Machine Learning (IAEML)  
**Author:** [Full name]  
**Group:** [Group]  
**Signature:** [Sign before submission]

## 1. Vehicle dynamics model

The objective is to implement a two-dimensional vehicle environment in JAX that can later be used for reinforcement learning. The `RoboTaxiEnv` class inherits from the provided `BaseEnv`. The original environment supplies the map, moving obstacles, ray sensors and dynamic programming path planner. The new environment replaces direct position control with acceleration and steering, without modifying `base.py`.

### State and action spaces

The vehicle state is represented by $(x,y,\theta,v)$, where $(x,y)$ is the position, $\theta$ is the heading and $v$ is the forward speed. The environment also retains the goal, obstacle positions and velocities, planned path and simulation step counter.

The action is a two-dimensional continuous vector $(u_a,u_s)\in[-1,1]^2$. Positive acceleration commands increase speed, negative commands apply braking, and steering commands determine the front-wheel angle. Reverse motion is not supported.

### Dynamics

A kinematic bicycle model with simplified longitudinal resistance is used:

$$
\dot{x}=v\cos\theta,\qquad \dot{y}=v\sin\theta,
$$

$$
\dot{\theta}=\frac{v}{L}\tan\delta,\qquad
\dot{v}=a-c_r\tanh(v/0.1)-c_d v|v|.
$$

Here $L$ is the wheelbase, $\delta=0.55u_s$ is the steering angle, and $a=3u_a$ for nonnegative commands or $a=5u_a$ for negative commands. The resistance terms approximate rolling resistance and quadratic drag. The hyperbolic tangent makes rolling resistance vanish smoothly near zero speed.

| Parameter | Value |
|---|---:|
| Simulation frequency | 60 Hz |
| Wheelbase | 0.6 world units |
| Maximum speed | 6 units/s |
| Maximum acceleration | 3 units/s² |
| Maximum braking magnitude | 5 units/s² |
| Maximum steering angle | 0.55 rad |
| Rolling resistance coefficient | 0.15 units/s² |
| Quadratic drag coefficient | 0.05 inverse world units |

At each time step, velocity is updated using an explicit Euler step and clipped to the interval $[0,6]$. The position and heading are updated using the average of the old and new speeds, with the position evaluated at the midpoint heading. Heading is wrapped to $[-\pi,\pi)$. Moving obstacles are advanced using the original environment's movement function.

The kinematic model is a compromise between realistic steering constraints and computational simplicity. Unlike direct position control, it cannot move sideways or rotate in place. A full dynamic bicycle model with tire slip and lateral forces would introduce additional parameters that are unnecessary for this small simulation. The wheelbase is deliberately short to accommodate tight turns in the provided map; these parameters are simulation values rather than calibrated real-world vehicle measurements.

### Observation space

The observation is flattened into a 38-element vector with the default 16 ray sensors:

| Feature | Dimensions | Representation |
|---|---:|---|
| Distance to planned path | 1 | Nearest valid path-segment distance divided by perception radius |
| Path direction | 2 | Unit direction in the vehicle's local coordinate frame |
| Obstacle rays | 32 | 16 two-dimensional world-frame ray vectors divided by perception radius |
| Relative goal position | 2 | Vehicle-local displacement divided by map diagonal |
| Vehicle speed | 1 | Speed divided by maximum speed |
| **Total** | **38** | |

The rays provide information about nearby obstacles, while path and goal features indicate the direction of travel. The speed feature helps distinguish stationary and moving states. The original navigation planner is updated once per simulated second during nonterminal steps. The observation does not explicitly include obstacle velocities, so it is not a complete representation of the simulator state.

The `reset` and `step` methods use `jax.jit`. Fixed-size arrays and `jax.lax.cond` for conditional replanning make the environment suitable for compiled execution. No performance benchmark was conducted for this assignment.

## 2. Reward function

Let $d_t=\|p_t-g\|_2$ be the Euclidean distance from the vehicle to the goal. The implemented reward is

$$
r_t=2(d_t-d_{t+1})-0.01-0.005u_s^2
+20\,\mathbf{1}_{\text{goal}}-20\,\mathbf{1}_{\text{failure}}.
$$

| Component | Purpose |
|---|---|
| $2(d_t-d_{t+1})$ | Encourage progress toward the destination |
| $-0.01$ | Discourage unnecessarily long episodes |
| $-0.005u_s^2$ | Discourage excessive steering commands |
| $+20$ | Reward reaching the destination safely |
| $-20$ | Penalize collisions or leaving map boundaries |

The progress term provides feedback before the goal is reached, unlike a purely sparse reward. One limitation is that temporarily moving away from the goal to navigate around an obstacle may produce negative progress reward. There is no separate timeout penalty or explicit acceleration-effort penalty.

An episode terminates upon reaching the goal, colliding with an obstacle, leaving the map or exceeding the maximum number of steps. These conditions are evaluated after the movement update. Goal success is suppressed if a collision or boundary failure occurs on the same step.

Collision detection follows the original `BaseEnv` convention: an obstacle is considered hit when its center is closer than approximately one world unit to the agent center. This differs from exact overlap between two radius-one circles, which would require a threshold of two units. The original map places the agent one unit from a wall, making that stricter model incompatible with the initial placement. Collisions are checked at discrete time steps rather than continuously along the trajectory.

## 3. Difficulties and reflection

The following issues were observed during implementation and testing:

| Issue | Resolution and observed result |
|---|---|
| Immediate collision on the first step | The initial nearest obstacle distance was 1.0. Reusing the starter collision convention allowed the vehicle to complete a 100-step smoke test without collision. |
| Constant acceleration and steering did not navigate to the goal | The initial demonstration ended in a collision after 148 steps. A classical path-following controller was introduced and the wheelbase was reduced from 2.0 to 0.6. |
| Moving obstacles could cross the reference route | The controller predicts relative positions over the next second and requests braking when predicted clearance is below 1.35 units. |
| Matplotlib failed to initialize Tkinter on Windows | Switching to the noninteractive `Agg` backend allowed MP4 rendering without opening a graphical window. |

## 4. Results and reproduction

### Classical controller

To demonstrate the environment independently of reinforcement learning, `src/path_controller.py` implements a Pure Pursuit-style controller. It follows a manually specified reference route for map 0, interpolated into closely spaced points. It selects a forward lookahead point and computes the steering angle using

$$
\delta=\operatorname{clip}\left(\arctan2(2L\sin\alpha,\max(d_{\mathrm{lookahead}},0.15)), -0.55,0.55\right),
$$

where $\alpha$ is the heading error to the lookahead point. The controller reduces target speed when steering increases, uses proportional speed control with resistance compensation, and brakes for predicted moving obstacles. Its maximum target speed is 1.6 units/s. It uses privileged obstacle positions and velocities from the simulator state, not solely the observation vector.

The reference route is separate from the dynamic programming path generated inside the environment. It is specific to map 0 and does not establish generalization to other maps. This controller is a demonstration baseline, not a trained reinforcement learning policy.

### Experimental results

The final experiment used map 0, PRNG seed 42, a path array length of 100 and a maximum episode length of 1,800 steps.

| Metric | Initial fixed-action run | Final controller run |
|---|---:|---:|
| Goal reached | No | Yes |
| Collision | Yes | No |
| Episode length | 148 steps | 1,118 steps |
| Final speed | 4.565 units/s | 1.318 units/s |
| Total reward | Not recorded | 34.46 |
| Out of bounds | No | No |
| Timeout | No | No |

The final episode lasted approximately 18.63 simulated seconds. The final distance to the environment planner's path was 0.305 units. The vehicle reached the goal region without triggering a collision, boundary failure or timeout. The comparison is between two implementation configurations (including a wheelbase change), rather than a controlled ablation of the controller alone. A single successful episode demonstrates functionality but does not establish a general success rate. Dedicated braking, boundary and timeout tests remain unverified.

**Demonstration video:** [assignment1_demo.mp4](assignment1_demo.mp4)

The video displays static and moving obstacles, the vehicle and its heading, sensor rays, the reference route, actual trajectory and reward. It records every second simulation step at 30 frames per second, preserving approximately real-time playback.

### Reflection

The first difficulty was implementing vehicle dynamics while maintaining compatibility with the existing JAX environment. Initially, the vehicle collided with an obstacle immediately after the first simulation step. After checking the distances between the vehicle and nearby obstacles, the problem was identified as a difference between our collision detection logic and the convention used in the original environment. This was resolved by using the original collision detection method.

Another challenge was controlling the vehicle. The initial simulation used constant acceleration and steering, which eventually caused a collision. To address this, a Pure Pursuit controller was introduced, together with adaptive speed control and braking when approaching moving obstacles. Reducing the wheelbase also allowed the vehicle to make tighter turns within the limited map space.

There was also an issue with Matplotlib on Windows, which was resolved by switching to the Agg rendering backend.

The final simulation successfully reached the destination in 1,118 steps without collisions. This assignment demonstrated the importance of testing individual environment components before combining them into a complete system. One remaining limitation is that the controller uses a manually defined reference path specific to the selected map.

### Reproduction instructions

Use the [starter repository](https://github.com/InnoRL/IAEML-assignment) with the following added files:

```text
src/
├── env/
│   └── robotaxi.py
├── path_controller.py
├── test_robotaxi.py
└── visualize.py
```

From the repository root, activate the Python virtual environment and install the starter project's dependencies. The additional scripts require JAX, Equinox, Chex, NumPy, Matplotlib and FFmpeg (available on PATH). The following commands install the Python libraries and run the demonstration:

```powershell
python -m pip install jax equinox chex numpy matplotlib
ffmpeg -version
python src/test_robotaxi.py
python src/visualize.py
```

The visualization script writes `assignment1_demo.mp4` to the repository root. The recorded successful run printed:

```text
Goal reached | Steps: 1118 | Total reward: 34.46
Video saved: assignment1_demo.mp4
```

Exact dependency versions were not recorded in the supplied experiment output. They can be captured from the working environment using `python -m pip freeze` before submission.

## References

- InnoRL, [IAEML assignment starter repository](https://github.com/InnoRL/IAEML-assignment), original environment, map definitions and navigation functions.
- IAEML — Assignments, course handout, Assignment 1: Environment.
