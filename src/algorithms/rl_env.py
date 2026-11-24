"""Experiment B: route-conditioned RL wrapper for the original map 0.
No PathController actions are used; the route is a navigation hint only.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from env.robotaxi import RoboTaxiEnv, MAX_SPEED

ACTION_REPEAT = 6
GAMMA = 0.995  # per policy decision, not per physics frame
ROUTE = jnp.array([[2,1],[2,4.8],[2.25,5.65],[3,6.4],[4,6.5],
                   [7,6.5],[11,6.5],[13,5.5],[15.5,2.5],[16,1]], dtype=jnp.float32)
SEG = ROUTE[1:] - ROUTE[:-1]
LENGTH = jnp.linalg.norm(SEG, axis=1)
ARC = jnp.concatenate([jnp.zeros(1), jnp.cumsum(LENGTH)])


def point_at(distance):
    distance = jnp.clip(distance, 0., ARC[-1])
    i = jnp.clip(jnp.searchsorted(ARC, distance, side='right')-1, 0, len(SEG)-1)
    return ROUTE[i] + SEG[i] * ((distance-ARC[i])/LENGTH[i]), SEG[i]/LENGTH[i]


def project(pos):
    t = jnp.clip(jnp.sum((pos-ROUTE[:-1])*SEG, axis=1)/LENGTH**2, 0., 1.)
    dist2 = jnp.sum((pos-ROUTE[:-1]-t[:,None]*SEG)**2, axis=1)
    i = jnp.argmin(dist2)
    return ARC[i]+t[i]*LENGTH[i], jnp.sqrt(dist2[i])


def observe(s, obs, params):
    progress, cross = project(s.agent_pos)
    target, tangent = point_at(progress + 1.0)
    forward = s.agent_forward_dir
    left = jnp.array([-forward[1], forward[0]])
    def local(v):
        return jnp.stack([jnp.dot(v, forward), jnp.dot(v, left)])
    # Rays are in a fixed order relative to heading; magnitudes are sufficient.
    rays = jnp.linalg.norm(obs.collision_rays, axis=-1)/params.perception_radius
    return jnp.concatenate([rays, local(target-s.agent_pos)/3., local(tangent),
        local(s.goal_pos-s.agent_pos)/25., jnp.array([s.speed/MAX_SPEED,
        cross/3., (ARC[-1]-progress)/ARC[-1]])])


def reset(key, params, initial, stage):
    """Stages 0/1 shorten the remaining route; stage 2 is the original start."""
    fraction = jax.random.uniform(key, (), minval=0., maxval=1.)
    start = jnp.where(stage == 0, .72+.10*fraction,
                      jnp.where(stage == 1, .32+.18*fraction, 0.))
    pos, direction = point_at(start*ARC[-1])
    heading = jnp.arctan2(direction[1], direction[0])
    state = eqx.tree_at(lambda s: (s.agent_pos,s.agent_forward_dir,s.heading),
                        initial,(pos,direction,heading))
    # Restore the exact original start in the final stage.
    state = jax.tree.map(lambda a,b: jnp.where(stage == 2,b,a), state,initial)
    obs = RoboTaxiEnv.get_observation(state, params)
    return observe(state, obs, params), state


def step(key, state, action, params):
    """Hold a policy action for at most six physics steps, stopping on done."""
    def body(carry, k):
        s, total, original, done, success, collision, outside, timeout, count = carry
        def active(_):
            _, ns, base_r, terminal, info = RoboTaxiEnv.step(k,s,action,params)
            old_progress,_ = project(s.agent_pos)
            progress,cross = project(ns.agent_pos)
            fail = info['collision'] | info['out_of_bounds']
            reached = info['goal_reached']
            timed = info['timeout'] & ~fail & ~reached
            reward = 4.*(progress-old_progress) - .01 - .003*cross**2 - .002*action[1]**2
            reward += 50.*reached - 20.*fail - 10.*timed
            return (ns,total+reward,original+base_r,terminal,reached,
                    info['collision'],info['out_of_bounds'],timed,count+1)
        return jax.lax.cond(done,lambda _:carry,active,operand=None), None
    zero = jnp.array(0.,dtype=jnp.float32)
    false = jnp.array(False)
    carry = (state,zero,zero,false,false,false,false,false,jnp.array(0,jnp.int32))
    (ns,r,base_r,done,success,collision,outside,timeout,count),_ = jax.lax.scan(
        body,carry,jax.random.split(key,ACTION_REPEAT))
    obs = RoboTaxiEnv.get_observation(ns,params)
    return observe(ns,obs,params),ns,r,done,dict(goal_reached=success,collision=collision,
        out_of_bounds=outside,timeout=timeout,physics_steps=count,original_reward=base_r)
