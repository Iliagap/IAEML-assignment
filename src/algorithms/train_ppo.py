"""Train PPO on map 0. Run from repository root: python src/train_ppo.py."""
import argparse
import csv
from pathlib import Path
import time
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from env.robotaxi import RoboTaxiEnv
from ppo import init_model, forward, sample, gae, loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=1048576)
    ap.add_argument('--envs', type=int, default=16)
    ap.add_argument('--rollout', type=int, default=128)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='results/ppo')
    args = ap.parse_args()
    n, t = args.envs, args.rollout
    batch_size = n*t
    if batch_size < 4 or batch_size % 4:
        ap.error('envs * rollout must be divisible by 4')
    updates = max(1, (args.steps + batch_size-1)//batch_size)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(args.seed)
    key, init_key, model_key = jax.random.split(key, 3)
    params, initial = RoboTaxiEnv.init_params(init_key, 0, 1800, 100)
    initial_obs, _ = RoboTaxiEnv.reset(init_key, params, initial)
    initial_vec = initial_obs.as_vector()
    p = init_model(model_key, initial_vec.shape[0])
    optimizer = optax.chain(optax.clip_by_global_norm(.5), optax.adam(3e-4, eps=1e-5))
    opt_state = optimizer.init(p)
    states = jax.tree.map(lambda x: jnp.broadcast_to(x, (n,)+x.shape), initial)
    obs = jnp.broadcast_to(initial_vec, (n, initial_vec.shape[0]))
    ep_returns = jnp.zeros(n)
    ep_lengths = jnp.zeros(n, dtype=jnp.int32)

    def step_one(k, s, a):
        o, ns, r, done, info = RoboTaxiEnv.step(k, s, a, params)
        # Reset after recording the terminal transition.
        ns = jax.tree.map(lambda x, y: jnp.where(done, y, x), ns, initial)
        return jnp.where(done, initial_vec, o.as_vector()), ns, r, done, info['goal_reached'], info['collision']

    @jax.jit
    def collect(p, carry):
        def body(carry, _):
            key, states, obs, er, el = carry
            key, action_key, step_key = jax.random.split(key, 3)
            actions, z, lp, value = sample(p, obs, action_key)
            no, ns, r, done, success, collision = jax.vmap(step_one)(jax.random.split(step_key, n), states, actions)
            er, el = er+r, el+1
            data = (obs, z, lp, value, r, done, success, collision,
                    jnp.where(done, er, 0.), jnp.where(done, el, 0))
            return (key, ns, no, jnp.where(done, 0., er), jnp.where(done, 0, el)), data
        return jax.lax.scan(body, carry, None, length=t)

    @jax.jit
    def optimize(p, opt_state, key, data, last_obs):
        o,z,lp,v,r,d,*_ = data
        _, _, final_value = forward(p, last_obs)
        adv, ret = gae(r,v,d,final_value)
        adv = (adv-jnp.mean(adv))/(jnp.std(adv)+1e-8)
        flat = tuple(x.reshape((batch_size,)+x.shape[2:]) for x in (o,z,lp,adv,ret))
        def epoch(carry, k):
            p, os = carry
            ids = jax.random.permutation(k, batch_size).reshape(4, -1)
            def minibatch(carry, idx):
                p, os = carry
                l, grads = jax.value_and_grad(loss)(p, tuple(x[idx] for x in flat))
                delta, os = optimizer.update(grads, os, p)
                return (optax.apply_updates(p, delta), os), l
            return jax.lax.scan(minibatch, (p,os), ids)
        (p, os), losses = jax.lax.scan(epoch, (p,opt_state), jax.random.split(key, 4))
        return p, os, jnp.mean(losses)

    carry = (key, states, obs, ep_returns, ep_lengths)
    start = time.perf_counter()
    window = np.zeros(5)
    with (out/'metrics.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['steps','episodes','success_rate','collision_rate','mean_return','mean_length','loss','steps_per_second'])
        for i in range(updates):
            carry, data = collect(p, carry)
            key, update_key = jax.random.split(carry[0]); carry = (key, *carry[1:])
            p, opt_state, l = optimize(p,opt_state,update_key,data,carry[2])
            done, success, collision, returns, lengths = data[5:]
            window += np.array([float(jnp.sum(x)) for x in (done,success,collision,returns,lengths)])
            if (i+1)%10 == 0 or i == 0 or i+1 == updates:
                count = window[0]; rates = window[1:]/count if count else [float('nan')]*4
                steps = (i+1)*batch_size; sps = steps/(time.perf_counter()-start)
                writer.writerow([steps,int(count),*rates,float(l),sps]); f.flush()
                print(f'{steps:,} steps | episodes={int(count)} | success={rates[0]:.1%} | collision={rates[1]:.1%} | return={rates[2]:.2f} | {sps:.0f} steps/s', flush=True)
                window[:] = 0
                eqx.tree_serialise_leaves(out/'model.eqx', p)
    print('Checkpoint:', out/'model.eqx')


if __name__ == '__main__':
    main()
