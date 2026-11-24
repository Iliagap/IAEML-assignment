"""Evaluate Experiment B on the unchanged original start; no expert actions."""
import argparse, csv
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from env.robotaxi import RoboTaxiEnv
from ppo import init_model, forward, sample
import rl_env as env


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',default='results/ppo_improved/model.eqx')
    ap.add_argument('--episodes',type=int,default=10)
    ap.add_argument('--seed',type=int,default=123)
    ap.add_argument('--deterministic',action='store_true')
    ap.add_argument('--video',action='store_true')
    ap.add_argument('--video-episode',type=int,default=None,help='1-based episode to record; default: first success, or first episode if none succeeds')
    args=ap.parse_args()
    if args.episodes<1: ap.error('episodes must be positive')
    if args.video_episode is not None and not 1 <= args.video_episode <= args.episodes:
        ap.error('video-episode must be between 1 and episodes')
    key=jax.random.PRNGKey(args.seed)
    params,initial=RoboTaxiEnv.init_params(key,0,1800,100)
    vec,_=env.reset(key,params,initial,jnp.array(2))
    p=eqx.tree_deserialise_leaves(args.checkpoint,init_model(key,vec.shape[0]))
    @jax.jit
    def act(o,k):
        return jnp.tanh(forward(p,o)[0]) if args.deterministic else sample(p,o,k)[0]
    @jax.jit
    def step(k,s,a): return env.step(k,s,a,params)
    selected_frames=None; selected_row=None; selected_episode=None; results=[]
    for episode in range(args.episodes):
        key,rk=jax.random.split(key)
        o,s=env.reset(rk,params,initial,jnp.array(2))
        total=base=0.; decisions=0
        frames=[]
        capture = args.video and (
            episode + 1 == args.video_episode if args.video_episode is not None
            else selected_row is None or not selected_row[0]
        )
        def record():
            obs=RoboTaxiEnv.get_observation(s,params)
            frames.append(dict(pos=np.asarray(s.agent_pos),direction=np.asarray(s.agent_forward_dir),
                static=np.asarray(s.static_obstacles),moving=np.asarray(s.kinematic_obstacles),
                goal=np.asarray(s.goal_pos),path=np.asarray(env.ROUTE),rays=np.asarray(obs.collision_rays),
                speed=float(s.speed),time=int(s.time),total=total))
        if capture: record()
        while True:
            key,ak,sk=jax.random.split(key,3)
            o,s,r,d,info=step(sk,s,act(o,ak)); total+=float(r); base+=float(info['original_reward']); decisions+=1
            if capture: record()
            if bool(d): break
        row=[bool(info[k]) for k in ('goal_reached','collision','out_of_bounds','timeout')]+[total,base,int(s.time),decisions]
        results.append(row)
        if capture and (args.video_episode is not None or selected_row is None or row[0]):
            selected_frames, selected_row, selected_episode = frames, row, episode + 1
        print(f'Episode {episode+1}: success={row[0]} collision={row[1]} outside={row[2]} timeout={row[3]} '
              f'return={total:.2f} original_return={base:.2f} physics_steps={row[6]} decisions={decisions}')
    avg=np.mean(results,axis=0)
    print(f'Success={avg[0]:.1%} | Collision={avg[1]:.1%} | Outside={avg[2]:.1%} | Timeout={avg[3]:.1%} | '
          f'Mean shaped return={avg[4]:.2f} | Mean original return={avg[5]:.2f}')
    out=Path(args.checkpoint).parent
    mode='deterministic' if args.deterministic else 'stochastic'
    with (out/f'evaluation_{mode}.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['success','collision','outside','timeout','shaped_return','original_return','physics_steps','decisions']); w.writerows(results)
    if args.video:
        assert selected_frames is not None and selected_row is not None
        assert len(selected_frames) == selected_row[7] + 1
        assert selected_frames[0]['time'] == 0
        assert selected_frames[-1]['time'] == selected_row[6]
        status = 'success' if selected_row[0] else 'failure'
        print(f'Rendering episode {selected_episode}: {status}, {selected_row[6]} physics steps, {len(selected_frames)} frames')
        render(selected_frames,params,out/f'assignment2_{mode}_{status}.mp4',selected_row[0])

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
        ax.plot(*f['path'].T,'g--',alpha=.5,label='Reference route')
        ax.plot(*trajectory[:i+1].T,color='blue',label='Learned trajectory')
        for ray in f['rays']:
            end=p+ray; ax.plot([p[0],end[0]],[p[1],end[1]],color='orange',alpha=.4)
        if i == len(frames)-1: ax.set_title(ax.get_title()+(' | Goal reached' if success else ' | Goal not reached'))
        ax.legend(loc='upper right'); ax.grid(alpha=.15)
    ani=FuncAnimation(fig,update,frames=list(range(len(frames)))+[len(frames)-1]*10,interval=1000*env.ACTION_REPEAT/params.fps,repeat=False)
    ani.save(str(output),writer=FFMpegWriter(fps=params.fps/env.ACTION_REPEAT,bitrate=1800),dpi=100)
    plt.close(fig); print('Video:',output)



if __name__=='__main__':main()
