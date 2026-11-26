# Assignment 2 — Reinforcement Learning for RoboTaxi Navigation

**Course:** Introduction to Adaptive and Efficient Machine Learning (IAEML)  
**Author:** [Ilia Salmin]  
**Group:** [AI-03]  

## 1. Experimental setup and RL model

### 1.1. Objective and environment

The objective is to train a policy that controls the RoboTaxi environment developed in Assignment 1. The vehicle must reach its destination while avoiding static and moving obstacles. The experiments use map 0, the original vehicle dynamics, a simulation frequency of 60 Hz and a maximum episode length of 1,800 physics steps. The two continuous actions are normalized acceleration/braking and steering, each in the interval $[-1,1]$.

The environment provides observations, applies the selected action, and returns a reward and a termination flag. An episode ends when the vehicle reaches the goal, collides, leaves the map or reaches the time limit. The collision convention is inherited from the starter repository; it is a simplified center-distance test rather than exact overlap of two radius-one circles.

Two PPO experiments were performed. Experiment A uses the original 38-dimensional observation and reward. Experiment B changes the observation representation, reward, action frequency and starting-position distribution while preserving the underlying physics. The manually designed reference route from Assignment 1 is used as a navigation hint in B, but the handcrafted controller does not select any training or evaluation actions.

### 1.2. Actor-Critic architecture

The implementation in `src/ppo.py` uses JAX arrays to represent a shared multilayer perceptron with two hidden layers of 64 neurons and tanh activations. The actor head predicts two Gaussian means and uses two learned log-standard-deviation parameters. The critic head predicts a scalar state value $V_\theta(s)$, an estimate of expected discounted future rewards.

The stochastic policy samples an unconstrained action $z$ and maps it to the valid control interval:

$$z=\mu_\theta(s)+\sigma_\theta\odot\epsilon,\qquad \epsilon\sim\mathcal N(0,I),\qquad a=\tanh(z).$$

The action log-probability includes the Jacobian correction for the tanh transformation. During deterministic evaluation, the action is $\tanh(\mu_\theta(s))$; stochastic evaluation samples from the learned distribution. Orthogonal weight initialization is used. Experiment B additionally initializes the actor acceleration bias to 0.35 and the log standard deviations to -0.2 instead of -0.5, providing an initial forward tendency and greater exploration without expert actions.

### 1.3. Proximal Policy Optimization (PPO)

PPO is an on-policy reinforcement learning algorithm. The current policy collects transitions from the environment, and the actor and critic are then updated using these transitions. The clipped policy objective limits the incentive for excessively large changes relative to the policy that generated the rollout.

Let $q_t(\theta)=\exp(\log\pi_\theta(a_t|s_t)-\log\pi_{old}(a_t|s_t))$. The actor loss is

$$L_{actor}=-\mathbb E_t[\min(q_t\hat A_t,\operatorname{clip}(q_t,1-\varepsilon,1+\varepsilon)\hat A_t)],\qquad \varepsilon=0.2.$$

The advantages are calculated with Generalized Advantage Estimation (GAE). With terminal indicator $d_t$ and discount factor $\gamma$,

$$\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t),$$

$$\hat A_t=\delta_t+\gamma\lambda(1-d_t)\hat A_{t+1}.$$

The implementation uses $\lambda=0.95$, normalizes advantages before optimization, and constructs value targets from the unnormalized advantages plus the rollout values. The critic loss is half the mean squared error between predicted values and these targets. The combined loss is $L_{actor}+0.5L_{critic}-0.01H$, where $H$ is the entropy of the underlying Gaussian (an exploration regularizer, not the exact entropy of the squashed distribution). All terminal transitions, including timeouts, mask the bootstrap value in this implementation.

Each update collects 128 policy decisions from each of 16 environments, giving 2,048 transitions. The batch is shuffled and divided into four minibatches for each of four optimization epochs. Adam updates the parameters after clipping the global gradient norm to 0.5.

### 1.4. Experiment configuration

| Parameter | A: baseline | B: improved |
|---|---:|---:|
| Training seed | 42 | 42 |
| Parallel environments | 16 | 16 |
| Rollout length (decisions/environment) | 128 | 128 |
| Total policy decisions | 1,048,576 | 262,144 |
| PPO updates | 512 | 128 |
| Observation dimension | 38 | 25 |
| Physics steps per decision | 1 | Up to 6 |
| Discount factor per decision | 0.99 | 0.995 |
| GAE lambda | 0.95 | 0.95 |
| Adam learning rate | 0.0003 | 0.0003 |
| Adam epsilon | 0.00001 | 0.00001 |
| Optimization epochs / minibatches | 4 / 4 | 4 / 4 |
| Curriculum | None | Three stages |

### 1.5. Reward design and curriculum

Experiment A uses the original environment reward:

$$r_t=2(D_t-D_{t+1})-0.01-0.005u_t^2+20I_{goal}-20I_{failure},$$

where $D_t$ is the distance to the destination, $u_t$ is the normalized steering command and failure means collision or leaving the map. The baseline policy can avoid the large collision penalty by remaining stationary. The direct-distance progress term also does not explicitly account for detours around walls.

Experiment B uses the wrapper `src/rl_env.py`. A piecewise-linear reference route is defined by ten waypoints. The vehicle position is projected onto the closest route segment to calculate arc-length progress $p_t$ and cross-track distance $c_t$. The new reward per physics step is

$$r_t^{B}=4(p_{t+1}-p_t)-0.01-0.003c_{t+1}^2-0.002u_t^2+50I_{goal}-20I_{failure}-10I_{timeout}.$$

Timeout is penalized only when neither success nor failure occurs. The original reward is accumulated separately for comparison. The new reward provides a positive signal for following a detour, even when that detour increases the direct distance to the destination.

The 25-dimensional observation contains 16 normalized ray distances, two local coordinates of a look-ahead point, two local components of the route tangent, two local coordinates of the goal, normalized speed, cross-track distance and remaining route fraction. Local coordinates are relative to the vehicle heading. This representation supplies navigation information without requiring the policy to output waypoints.

The curriculum changes the initial position along the same route while retaining the original obstacles:

| Stage | Initial route progress | Approximate share of updates |
|---|---|---:|
| 1 | Random 72–82% | 15% |
| 2 | Random 32–50% | 30% |
| 3 | Exact original spawn and heading | 55% |

Partial episodes are reset at stage transitions. The schedule is fixed rather than dependent on performance. Evaluation always uses stage 3, so the final model must navigate from the original starting position.

## 2. Computational optimizations

The implementation uses the JAX transformations studied in the course:

- **`jax.jit`:** compiles rollout collection and optimization, avoiding repeated Python dispatch for the inner training computations.
- **`jax.vmap`:** batches environment transitions across 16 independent simulations. The actor and critic also process batched observations using matrix operations.
- **`jax.lax.scan`:** collects sequential rollouts, computes GAE in reverse order, and executes optimization epochs and minibatches. Experiment B also uses a scan for repeated physics steps.
- **`jax.value_and_grad`:** evaluates the PPO loss and its gradients for each minibatch.
- **Action repetition:** B holds the same control for up to six physics steps, stopping early on terminal transitions. Physics remains at 60 Hz while the nominal policy decision frequency becomes 10 Hz. This reduces neural-network calls and the number of decisions needed to complete a route. The wrapper sums substep rewards without within-repeat discounting; the configured discount is applied at the policy-decision level.
- **Fixed-shape batches and automatic resets:** terminal environments are reset within the batched transition pipeline, allowing rollouts to continue without variable-length Python episode loops.

Measured throughput is cumulative physics steps divided by elapsed training-loop wall time, including initial compilation and logging overhead. At the end of A it was 1,593.55 physics steps/s; at the end of B it was 2,608.26 physics steps/s. This is approximately 1.64 times the baseline throughput (63.7% higher). B also completed 438.56 policy decisions/s. The two units should not be confused.

The measurement is an end-to-end comparison of the two configurations, not an isolated benchmark of JIT, vectorization or action repetition. Observation size, episode behavior, training duration and reward computation also differ. No separate GPU/CPU benchmark, multi-device experiment or hardware utilization measurement was performed. Exact device and dependency versions should be recorded from the local machine if available.

## 3. Difficulties, resolutions and reflection

### 3.1. Observed technical difficulties

The first experiment completed its full training budget, but none of the logged completed episodes reached the goal. The mean return eventually approached -18.24, close to the -18 time penalty for a stationary 1,800-step episode. The baseline evaluation video shows the vehicle stopping rather than navigating. The reward therefore supported a low-risk but unsuccessful strategy. Experiment B addressed this with route-progress rewards, a timeout penalty, curriculum learning and modified exploration.

Another issue was interpreting `NaN` values in training statistics. In Experiment A these appeared in logging windows with zero completed episodes; the episode averages were undefined, not evidence of nonfinite model weights. Both uploaded training CSVs contain finite logged PPO losses.

The initial success-video export also selected the wrong frames for the reported successful episode. The evaluation script was corrected to keep episode frames separate, select the first successful episode (or a requested episode), and verify that the first and last frame times and frame count match the selected episode. Repeating the stochastic evaluation produced identical CSV contents.

### 3.2. Reflection

The biggest challenge was getting the agent to move instead of staying still to avoid collisions. After watching the baseline evaluation, I changed the reward, introduced curriculum learning and used action repetition. The improved policy reached the goal in 1 out of 10 stochastic evaluation episodes, and training throughput increased from 1,594 to 2,608 physics steps per second.

I learned that a better reward does not always mean better behavior. Watching the agent and checking its actual success rate are just as important. The policy still collides too often and needs further improvement.

## 4. Results and local reproduction

### 4.1. Training results

The following values are taken from the two training CSV files. Rates and mean returns in the final nonempty logging windows describe training episodes, not independent evaluation runs.

| Metric | A | B |
|---|---:|---:|
| Actual physics steps | 1,048,576 | 1,559,065 |
| Final nonempty-window success | 0% | 8.5% |
| Final nonempty-window collision | 0% | 91.5% |
| Final nonempty-window original return | -18.24 | -9.82 |
| Final nonempty-window shaped return | Not applicable | 27.53 |
| Peak reported full-route success | 0% | 12.9% |
| Final cumulative physics steps/s | 1,593.55 | 2,608.26 |

Experiment A finished with a logging window containing no completed episodes. Its last nonempty window was at 1,044,480 steps. The recorded episode lengths were 1,800 steps, and no successes or collisions were logged.

Experiment B's episode-weighted training success rates across stages were 54.4%, 10.9% and 7.2%, respectively. The increase in difficulty caused substantial drops in success at the stage transitions. In stage 3, the best reported window reached 12.9% success at 245,760 policy decisions, but the final window fell to 8.5%. The final checkpoint is therefore not necessarily the best checkpoint encountered during training.

### 4.2. Evaluation of the final checkpoint

The improved model was evaluated from the original starting position using seed 123. The deterministic and stochastic evaluations use the same trained parameters but different action selection methods.

| Metric | Deterministic | Stochastic |
|---|---:|---:|
| Episodes | 1 | 10 |
| Successful episodes | 0 | 1 |
| Collisions | 1 | 9 |
| Timeouts / outside map | 0 / 0 | 0 / 0 |
| Success rate | 0% | 10% |
| Mean original return | -7.20 | -8.38 |
| Mean shaped return | 37.57 | 30.04 |
| Mean physics steps | 259 | 272.7 |

The successful stochastic episode was episode 5. It reached the goal in 442 physics steps and 74 policy decisions, with original return 41.09 and shaped return 129.19. It had no collision, timeout or out-of-bounds event. The same results were reproduced in repeated evaluation CSVs.

For comparison, the successful classical controller demonstration in Assignment 1 required 1,118 physics steps and obtained original return 34.46. The selected successful PPO run required 7.37 seconds of simulated time versus 18.63 seconds for that classical run. This is a comparison of individual successful trajectories, not evidence that PPO is faster or safer on average.

The results show that the improved policy can complete the task, whereas the original policy did not. However, nine of ten stochastic evaluations collided and the single deterministic episode also collided. The policy is not reliable. The sample is small, only one training seed and one map were studied, and the reference route was manually designed. Generalization and the individual contribution of each modification remain untested. The successful video is deliberately selected from the evaluation batch and must not be interpreted as typical performance.

### 4.3. Demonstration and checkpoint

The following paths are relative to the repository root. Keep the generated files in these locations (or update the links if the repository layout differs).

| Artifact | Path |
|---|---|
| Baseline training metrics | [`results/ppo/metrics.csv`](results/ppo/metrics.csv) |
| Baseline checkpoint | [`results/ppo/model.eqx`](results/ppo/model.eqx) |
| Improved training configuration | [`results/ppo_improved/config.json`](results/ppo_improved/config.json) |
| Improved training metrics | [`results/ppo_improved/metrics.csv`](results/ppo_improved/metrics.csv) |
| Improved checkpoint | [`results/ppo_improved/model.eqx`](results/ppo_improved/model.eqx) |
| Deterministic evaluation | [`results/ppo_improved/evaluation_deterministic.csv`](results/ppo_improved/evaluation_deterministic.csv) |
| Stochastic evaluation | [`results/ppo_improved/evaluation_stochastic.csv`](results/ppo_improved/evaluation_stochastic.csv) |
| Successful episode video | [`results/ppo_improved/assignment2_stochastic_success.mp4`](results/ppo_improved/assignment2_stochastic_success.mp4) |

The corrected video is 1200 × 700 pixels at 10 FPS and lasts 8.5 seconds, including the initial frame and a one-second final-frame hold. Model checkpoints contain the parameter leaves serialized using Equinox; they are intended for inference rather than resuming the complete optimizer and random-number state.

### 4.4. Reproduction instructions

Run the following commands from the repository root in the existing Python virtual environment. The Assignment 1 source files and starter repository must be present. FFmpeg must be installed and accessible on PATH for MP4 export.

```powershell
python -m pip install jax equinox chex optax numpy matplotlib
python -c "import jax; print(jax.devices())"
```

To evaluate the existing improved checkpoint without retraining:

```powershell
python src/evaluate_ppo_b.py --episodes 1 --seed 123 --deterministic
python src/evaluate_ppo_b.py --episodes 10 --seed 123 --video --video-episode 5
```

The second command reproduces the selected episode and writes the stochastic evaluation CSV and video beside the checkpoint. The corrected evaluation script must be used. Different library versions or devices can affect exact numerical reproducibility.

To retrain without overwriting existing results:

```powershell
python src/train_ppo.py --steps 1048576 --envs 16 --rollout 128 --seed 42 --out results/ppo_reproduction
python src/train_ppo_b.py --steps 262144 --envs 16 --rollout 128 --seed 42 --out results/ppo_improved_reproduction
python src/evaluate_ppo_b.py --checkpoint results/ppo_improved_reproduction/model.eqx --episodes 10 --seed 123 --video
```

The improved trainer requires a new output directory if a metrics file already exists. It logs actual physics steps separately from policy decisions and saves the latest model periodically. The last evaluation command automatically selects the first successful episode if one occurs; otherwise it records a failure. Exact reproduction of the original success is not guaranteed for newly trained checkpoints on different software or hardware.

## References

- IAEML Assignments, Assignment 2: ML, course handout, p. 3.
- IAEML Lecture 4: Learning Frameworks, curriculum learning.
- IAEML Lecture 8: Foundations of High-Performance Computing in Machine Learning.
- Starter environment: [InnoRL/IAEML-assignment](https://github.com/InnoRL/IAEML-assignment).
