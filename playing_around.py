import jax.numpy as jnp
import jax

jax.config.update("jax_default_device", jax.devices("gpu")[0])

@jax.jit
def slow_fn(x):
    return jnp.sum(jnp.sin(x) ** 2)

# fast_fn = jax.jit(slow_fn) -> extra step that can be removed using a decorator
fr = slow_fn(jnp.ones(1000))
sr = slow_fn(jnp.ones(1000))
print(fr, sr)

x = jnp.arange(10)
print(x.devices())