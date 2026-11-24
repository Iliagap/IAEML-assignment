"""Record the map-0 classical path-following baseline as an MP4."""
import jax
import numpy as np
import matplotlib
matplotlib.use('Agg')  # No Tkinter or GUI required.
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, writers
from matplotlib.patches import Circle
from env.robotaxi import RoboTaxiEnv
from path_controller import PathController

MAP_ID = 0
MAX_STEPS = 1800
OUTPUT = 'assignment1_demo.mp4'
FRAME_STRIDE = 2


def main():
    if not writers.is_available('ffmpeg'):
        raise RuntimeError('FFmpeg not found. Install FFmpeg and add it to PATH.')
    key = jax.random.PRNGKey(42)
    params, initial_state = RoboTaxiEnv.init_params(
        key=key, map_id=MAP_ID, max_steps=MAX_STEPS, path_length=100)
    obs, state = RoboTaxiEnv.reset(key, params, initial_state)
    controller = PathController()
    frames = []
    total_reward = 0.0
    info = {}

    def record(reward):
        frames.append(dict(position=np.asarray(state.agent_pos),
                           direction=np.asarray(state.agent_forward_dir),
                           static=np.asarray(state.static_obstacles),
                           moving=np.asarray(state.kinematic_obstacles),
                           goal=np.asarray(state.goal_pos),
                           rays=np.asarray(obs.collision_rays),
                           speed=float(state.speed), time=int(state.time),
                           reward=float(reward), total=total_reward))

    record(0.0)
    for step in range(MAX_STEPS):
        key, step_key = jax.random.split(key)
        action = controller.action(state, params)
        obs, state, reward, done, info = RoboTaxiEnv.step(step_key, state, action, params)
        total_reward += float(reward)
        if (step + 1) % FRAME_STRIDE == 0 or bool(done) or step + 1 == MAX_STEPS:
            record(reward)
        if bool(done):
            break
    status = ('Goal reached' if bool(info.get('goal_reached', False)) else
              'Collision' if bool(info.get('collision', False)) else
              'Out of bounds' if bool(info.get('out_of_bounds', False)) else 'Time limit')
    print(f'{status} | Steps: {int(state.time)} | Total reward: {total_reward:.2f}')
    print('Episode info:', info)
    print('Rendering video...')
    fig, ax = plt.subplots(figsize=(12, 7))
    height, width = params.map_height_width
    path = controller.path
    trajectory = np.array([f['position'] for f in frames])

    def update(i):
        ax.clear()
        f = frames[i]
        ax.set(xlim=(-1, width), ylim=(-1, height), xlabel='X position', ylabel='Y position',
               title=f'RoboTaxi | Step {f["time"]} | Speed {f["speed"]:.2f} u/s')
        ax.set_aspect('equal')
        ax.grid(alpha=0.15)
        for j, obstacle in enumerate(f['static']):
            ax.add_patch(Circle(obstacle, 1.0, color='gray', alpha=0.35,
                                label='Static obstacle' if j == 0 else None))
        for j, obstacle in enumerate(f['moving']):
            ax.add_patch(Circle(obstacle, 1.0, color='red', alpha=0.5,
                                label='Moving obstacle' if j == 0 else None))
        ax.add_patch(Circle(f['goal'], 1.0, color='green', alpha=0.4, label='Goal'))
        ax.plot(path[:, 0], path[:, 1], '--', color='green', label='Reference route')
        p, direction = f['position'], f['direction']
        # A small vehicle marker reflects the starter point-agent collision convention.
        ax.add_patch(Circle(p, 0.35, color='royalblue', label='Vehicle', zorder=5))
        ax.arrow(*p, *(direction * 0.8), color='royalblue', head_width=0.18,
                 length_includes_head=True, zorder=6)
        for ray in f['rays']:
            end = p + ray
            ax.plot([p[0], end[0]], [p[1], end[1]], color='orange', alpha=0.45, lw=0.8)
        ax.plot(trajectory[:i+1, 0], trajectory[:i+1, 1], color='blue', lw=1.5, label='Trajectory')
        text = f'Reward: {f["reward"]:.3f}\nTotal reward: {f["total"]:.2f}'
        if i == len(frames)-1:
            text += '\nResult: ' + status
        ax.text(0.02, 0.98, text, transform=ax.transAxes, va='top',
                bbox=dict(facecolor='white', alpha=0.9))
        ax.legend(loc='upper right', fontsize=8)

    fps = float(params.fps) / FRAME_STRIDE
    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000/fps, repeat=False)
    animation.save(OUTPUT, writer=FFMpegWriter(fps=fps, bitrate=1800), dpi=100)
    plt.close(fig)
    print(f'Video saved: {OUTPUT}')


if __name__ == '__main__':
    main()
