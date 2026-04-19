## Linear noise scheduler

import jax
import jax.numpy as jnp

jax.config.update("jax_default_device", jax.devices("gpu")[0])

class LinearNoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = jnp.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumulative_prod = jnp.cumulative_prod(self.alphas, axis=0)
        self.sqrt_alpha_cumulative_prod = jnp.sqrt(self.alpha_cumulative_prod)
        self.sqrt_one_minus_alpha_cumulative_prod = jnp.sqrt(1 - self.alpha_cumulative_prod)

    def add_noise(self, original, noise, t):
        original_shape = original.shape
        batch_size = original_shape[0]

        sqrt_alpha_cumulative_prod = self.sqrt_alpha_cumulative_prod[t].reshape(batch_size) # Reshaped to bx1x1x1
        sqrt_one_minus_alpha_cumulative_prod = self.sqrt_one_minus_alpha_cumulative_prod.reshape(batch_size) # Reshaped to bx1x1x1
        
        # Reshape till (B, ) becomes (B, 1, 1, 1) if image is (B, C, H, W)
        for _ in range(len(original_shape) - 1)