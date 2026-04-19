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
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cumulative_prod = sqrt_alpha_cumulative_prod.unsqueeze(-1)
        for _ in range(len(original_shape) - 1):
            sqrt_one_minus_alpha_cumulative_prod = sqrt_one_minus_alpha_cumulative_prod.unsqueeze(-1)
        
        # Apply and return forward process equation
        return (sqrt_alpha_cumulative_prod * original + sqrt_one_minus_alpha_cumulative_prod * noise)

    # takes in xₜ (a noisy image at step t) and the UNet's noise_pred (its guess of what noise is in the image), and outputs xₜ₋₁ (a slightly less noisy image)
    def sample_prev_timestep(self, xt, noise_pred, t, key):
        x0 = ((xt - (self.sqrt_one_minus_alpha_cumulative_prod[t] * noise_pred)) / jnp.sqrt(self.alpha_cumulative_prod[t]))
        x0 = jnp.clip(x0, -1.0, 1.0)
        mean = xt - ((self.betas[t]) * noise_pred) / (self.sqrt_one_minus_alpha_cumulative_prod[t])
        mean = mean / jnp.sqrt(self.alphas[t])
        if t == 0:
            return mean, x0
        else:
            variance = (1 - self.alpha_cumulative_prod[t-1]) / (1.0 - self.alpha_cumulative_prod[t])
            variance = variance * self.betas[t]
            sigma = variance ** 0.5
            key, subkey = jax.random.split(key)
            z = jax.random.normal(subkey, shape=xt.shape)
            return mean + sigma * z, x0, key
        
    
