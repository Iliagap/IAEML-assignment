"""PPO Experiment B. --steps counts policy decisions across all environments."""
import argparse, csv, json, time
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from env.robotaxi import RoboTaxiEnv
from ppo import init_model, forward, sample, gae, loss
import rl_env as env


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--steps',type=int,default=262144)
    ap.add_argument('--envs',type=int,default=16)
    ap.add_argument('--rollout',type=int,default=128)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--out',default='results/ppo_improved')
    args=ap.parse_args()
    n,t=args.envs,args.rollout
    batch=n*t
    if min(n,t,args.steps)<1 or batch%4: ap.error('positive sizes; envs * rollout divisible by 4')
    updates=(args.steps+batch-1)//batch
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.csv').exists(): ap.error('Output already exists; choose a new --out directory')
    key=jax.random.PRNGKey(args.seed)
    key,k,mk=jax.random.split(key,3)
    params,initial=RoboTaxiEnv.init_params(k,0,1800,100)
    vec,_=env.reset(k,params,initial,jnp.array(0))
    p=init_model(mk,vec.shape[0])
    # Mild forward bias and broader initial exploration, not expert actions.
    p['actor']=(p['actor'][0],jnp.array([.35,0.],dtype=jnp.float32))
    p['log_std']=jnp.full((2,),-.2)
    optimizer=optax.chain(optax.clip_by_global_norm(.5),optax.adam(3e-4,eps=1e-5))
    os=optimizer.init(p)
    (out/'config.json').write_text(json.dumps({**vars(args),'action_repeat':env.ACTION_REPEAT,
        'gamma':env.GAMMA,'obs_dim':int(vec.shape[0]),'map':0,'max_physics_steps':1800,
        'curriculum':'first 15% near goal, next 30% middle, remaining original spawn'},indent=2))

    @jax.jit
    def reset_batch(key,stage):
        key,rk=jax.random.split(key)
        obs,states=jax.vmap(lambda k:env.reset(k,params,initial,stage))(jax.random.split(rk,n))
        return key,states,obs,jnp.zeros(n),jnp.zeros(n),jnp.zeros(n,jnp.int32)

    def step_one(k,s,a,stage):
        sk,rk=jax.random.split(k)
        o,ns,r,d,info=env.step(sk,s,a,params)
        ro,rs=env.reset(rk,params,initial,stage)
        ns=jax.tree.map(lambda x,y:jnp.where(d,y,x),ns,rs)
        return jnp.where(d,ro,o),ns,r,d,info

    @jax.jit
    def collect(p,carry,stage):
        def body(c,_):
            key,s,o,er,br,el=c
            key,ak,sk=jax.random.split(key,3)
            a,z,lp,v=sample(p,o,ak)
            no,ns,r,d,info=jax.vmap(step_one,in_axes=(0,0,0,None))(jax.random.split(sk,n),s,a,stage)
            er=er+r; br=br+info['original_reward']; el=el+info['physics_steps']
            data=(o,z,lp,v,r,d,info['goal_reached'],info['collision'],info['out_of_bounds'],info['timeout'],
                  jnp.where(d,er,0.),jnp.where(d,br,0.),jnp.where(d,el,0),info['physics_steps'])
            return (key,ns,no,jnp.where(d,0.,er),jnp.where(d,0.,br),jnp.where(d,0,el)),data
        return jax.lax.scan(body,carry,None,length=t)

    @jax.jit
    def optimize(p,os,key,data,last_obs):
        o,z,lp,v,r,d,*_=data
        _,_,fv=forward(p,last_obs)
        adv,ret=gae(r,v,d,fv,gamma=env.GAMMA)
        adv=(adv-jnp.mean(adv))/(jnp.std(adv)+1e-8)
        flat=tuple(x.reshape((batch,)+x.shape[2:]) for x in (o,z,lp,adv,ret))
        def epoch(c,k):
            ids=jax.random.permutation(k,batch).reshape(4,-1)
            def mb(c,idx):
                p,os=c
                l,g=jax.value_and_grad(loss)(p,tuple(x[idx] for x in flat))
                delta,os=optimizer.update(g,os,p)
                return (optax.apply_updates(p,delta),os),l
            return jax.lax.scan(mb,c,ids)
        (p,os),ls=jax.lax.scan(epoch,(p,os),jax.random.split(key,4))
        return p,os,jnp.mean(ls)

    start=time.perf_counter(); physics=0; stage=-1; window=np.zeros(8)
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['decisions','physics_steps','stage','episodes','success_rate','collision_rate',
            'out_of_bounds_rate','timeout_rate','mean_return','mean_original_return','mean_physics_length',
            'loss','decisions_per_second','physics_steps_per_second'])
        for i in range(updates):
            # At least one update at original spawn, even for smoke tests.
            new_stage=2 if i==updates-1 else (0 if i/updates<.15 else 1 if i/updates<.45 else 2)
            if stage!=new_stage:
                if stage!=-1: key=carry[0]
                stage=new_stage; carry=reset_batch(key,jnp.array(stage)); window[:]=0
                print(f'Curriculum stage {stage+1}/3 (reset partial episodes)',flush=True)
            carry,data=collect(p,carry,jnp.array(stage))
            key,uk=jax.random.split(carry[0]); carry=(key,*carry[1:])
            p,os,l=optimize(p,os,uk,data,carry[2]); lv=float(l)
            if not np.isfinite(lv): raise RuntimeError('Nonfinite PPO loss; inspect inputs and checkpoint')
            stats=np.asarray(jnp.stack([jnp.sum(x).astype(jnp.float32) for x in data[5:]]))
            window+=stats[:-1]; physics+=int(stats[-1])
            next_stage=2 if i+1==updates-1 else (0 if (i+1)/updates<.15 else 1 if (i+1)/updates<.45 else 2)
            if i==0 or (i+1)%10==0 or i+1==updates or next_stage!=stage:
                count=window[0]; rates=window[1:]/count if count else np.full(7,np.nan)
                decisions=(i+1)*batch; elapsed=time.perf_counter()-start
                writer.writerow([decisions,physics,stage+1,int(count),*rates,lv,decisions/elapsed,physics/elapsed]); f.flush()
                print(f'{decisions:,} decisions | {physics:,} physics | stage={stage+1} | episodes={int(count)} | '
                    f'success={rates[0]:.1%} | collision={rates[1]:.1%} | timeout={rates[3]:.1%} | '
                    f'return={rates[4]:.2f} | original={rates[5]:.2f} | loss={lv:.4f} | {physics/elapsed:.0f} physics/s',flush=True)
                eqx.tree_serialise_leaves(out/'model.eqx',p); window[:]=0
    print('Checkpoint:',out/'model.eqx')

if __name__=='__main__':main()
