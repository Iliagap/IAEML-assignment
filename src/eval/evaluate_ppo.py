"""Evaluate a trained PPO policy and optionally render its first episode."""
import argparse
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from env.robotaxi import RoboTaxiEnv
from ppo import init_model, forward, sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='results/ppo/model.eqx')
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--map', type=int, default=0)
    ap.add_argument('--deterministic', action='store_true')
    ap.add_argument('--video', action='store_true')
    args = ap.parse_args()
    if args.episodes < 1:
        ap.error('episodes must be positive')
    key = jax.random.PRNGKey(args.seed)
    params, initial = RoboTaxiEnv.init_params(key, args.map, 1800, 100)
    obs, _ = RoboTaxiEnv.reset(key, params, initial)
    p = eqx.tree_deserialise_leaves(args.checkpoint, init_model(key, obs.as_vector().shape[0]))
    frames, results = [], []
    @jax.jit
    def action(o, k):
        if args.deterministic:
            return jnp.tanh(forward(p, o)[0])
        return sample(p, o, k)[0]
    for episode in range(args.episodes):
        obs, state = RoboTaxiEnv.reset(key, params, initial)
        total = 0.
        def record():
            frames.append(dict(pos=np.asarray(state.agent_pos), direction=np.asarray(state.agent_forward_dir),
                static=np.asarray(state.static_obstacles), moving=np.asarray(state.kinematic_obstacles),
                goal=np.asarray(state.goal_pos), path=np.asarray(state.path_array),
                rays=np.asarray(obs.collision_rays), speed=float(state.speed), time=int(state.time), total=total))
        if args.video and episode == 0: record()
        for step in range(params.max_steps_in_episode):
            key, ak, ek = jax.random.split(key, 3)
            obs, state, reward, done, info = RoboTaxiEnv.step(ek, state, action(obs.as_vector(), ak), params)
            total += float(reward)
            if args.video and episode == 0 and ((step+1)%2 == 0 or bool(done)): record()
            if bool(done): break
        result = (bool(info['goal_reached']), bool(info['collision']), total, step+1)
        results.append(result)
        print(f'Episode {episode+1}: success={result[0]} collision={result[1]} return={total:.2f} steps={step+1}')
    averages = np.mean(results, axis=0)
    print(f'Success={averages[0]:.1%} | Collision={averages[1]:.1%} | Mean return={averages[2]:.2f} | Mean steps={averages[3]:.1f}')
    if args.video:
        render(frames, params, Path(args.checkpoint).parent/'assignment2_demo.mp4', results[0][0])


def render(frames, params, output, success):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(12, 7))
    h,w = params.map_height_width
    trajectory = np.array([f['pos'] for f in frames])
    def update(i):
        ax.clear(); f = frames[i]; p = f['pos']
        ax.set(xlim=(-1,w), ylim=(-1,h), xlabel='X', ylabel='Y', aspect='equal',
               title=f'PPO | Step {f["time"]} | Speed {f["speed"]:.2f} | Return {f["total"]:.2f}')
        for obstacle in f['static']: ax.add_patch(Circle(obstacle,1,color='gray',alpha=.35))
        for obstacle in f['moving']: ax.add_patch(Circle(obstacle,1,color='red',alpha=.5))
        ax.add_patch(Circle(f['goal'],1,color='green',alpha=.5))
        ax.add_patch(Circle(p,.35,color='royalblue',zorder=5))
        ax.arrow(*p,*(f['direction']*.8),color='royalblue',head_width=.18)
        ax.plot(*f['path'].T,'g--',alpha=.5,label='Environment planner')
        ax.plot(*trajectory[:i+1].T,color='blue',label='Learned trajectory')
        for ray in f['rays']:
            end=p+ray; ax.plot([p[0],end[0]],[p[1],end[1]],color='orange',alpha=.4)
        if i == len(frames)-1: ax.set_title(ax.get_title()+(' | Goal reached' if success else ' | Goal not reached'))
        ax.legend(loc='upper right'); ax.grid(alpha=.15)
    ani=FuncAnimation(fig,update,frames=len(frames),interval=2000/params.fps,repeat=False)
    ani.save(str(output),writer=FFMpegWriter(fps=params.fps/2,bitrate=1800),dpi=100)
    plt.close(fig); print('Video:',output)


if __name__ == '__main__': main()
