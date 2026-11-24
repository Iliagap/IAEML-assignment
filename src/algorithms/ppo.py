"""Continuous-action PPO, with a tanh-squashed Gaussian policy."""
import jax
import jax.numpy as jnp
import optax


def init_model(key, obs_dim):
    keys = jax.random.split(key, 4)
    def layer(k, n, m, scale):
        return (jax.nn.initializers.orthogonal(scale)(k, (n, m)), jnp.zeros(m))
    return dict(l1=layer(keys[0], obs_dim, 64, 2**0.5),
                l2=layer(keys[1], 64, 64, 2**0.5),
                actor=layer(keys[2], 64, 2, 0.01),
                critic=layer(keys[3], 64, 1, 1.0), log_std=jnp.full((2,), -0.5))


def forward(p, obs):
    def linear(x, layer):
        return x @ layer[0] + layer[1]
    x = jnp.tanh(linear(obs, p['l1']))
    x = jnp.tanh(linear(x, p['l2']))
    return linear(x, p['actor']), jnp.clip(p['log_std'], -5., 1.), linear(x, p['critic'])[..., 0]


def log_prob(z, mean, log_std):
    normal = -0.5 * (((z-mean)*jnp.exp(-log_std))**2 + 2*log_std + jnp.log(2*jnp.pi))
    correction = 2*(jnp.log(2.) - z - jax.nn.softplus(-2*z))
    return jnp.sum(normal - correction, axis=-1)


def sample(p, obs, key):
    mean, log_std, value = forward(p, obs)
    z = mean + jnp.exp(log_std)*jax.random.normal(key, mean.shape)
    return jnp.tanh(z), z, log_prob(z, mean, log_std), value


def gae(rewards, values, dones, final_value, gamma=0.99, lam=0.95):
    next_values = jnp.concatenate([values[1:], final_value[None]], axis=0)
    def body(carry, x):
        r, v, nv, done = x
        mask = 1. - done.astype(jnp.float32)
        advantage = r + gamma*mask*nv - v + gamma*lam*mask*carry
        return advantage, advantage
    _, advantages = jax.lax.scan(body, jnp.zeros_like(final_value),
                                    (rewards, values, next_values, dones), reverse=True)
    return advantages, advantages + values


def loss(p, batch):
    obs, z, old_logp, advantages, returns = batch
    mean, log_std, value = forward(p, obs)
    ratio = jnp.exp(log_prob(z, mean, log_std) - old_logp)
    policy = -jnp.mean(jnp.minimum(ratio*advantages, jnp.clip(ratio, .8, 1.2)*advantages))
    critic = .5*jnp.mean((value-returns)**2)
    # Entropy of the underlying Gaussian, a simple exploration regularizer.
    entropy = jnp.sum(log_std + .5*jnp.log(2*jnp.pi*jnp.e))
    return policy + .5*critic - .01*entropy
