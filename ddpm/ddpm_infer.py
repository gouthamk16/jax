import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import yaml
import numpy as np
import orbax.checkpoint as ocp
from PIL import Image
from tqdm import tqdm
from ddpm import LinearNoiseScheduler, Unet, get_latest_checkpoint

# Set GPU
jax.config.update("jax_default_device", jax.devices("gpu")[0])

# Load model ckpts and configs
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)
dataset_config = config["dataset"]
model_config = config["model"]
diffusion_config = config["diffusion"]
train_config = config["training"]

scheduler = LinearNoiseScheduler(
    num_timesteps=diffusion_config["num_timesteps"],
    beta_start=diffusion_config["beta_start"],
    beta_end=diffusion_config["beta_end"]
)

model = Unet(
    img_channels=model_config["img_channels"],
    down_channels=model_config["down_channels"],
    mid_channels=model_config["mid_channels"],
    t_emb_dim=model_config["time_emb_dim"],
    down_sample=model_config["down_sample"],
    num_down_layers=model_config["num_down_layers"],
    num_mid_layers=model_config["num_mid_layers"],
    num_up_layers=model_config["num_up_layers"],
)

checkpointer = ocp.PyTreeCheckpointer()
ckpt_dir = os.path.abspath(train_config["ckpt_dir"])

key = jax.random.PRNGKey(0)
im_size = dataset_config["im_size"]
params = model.init(key, jnp.ones((1, im_size, im_size, model_config["img_channels"])), jnp.ones((1,), dtype=jnp.int32))["params"]
latest_ckpt_path, _ = get_latest_checkpoint(ckpt_dir)
if latest_ckpt_path is not None:
    params = checkpointer.restore(latest_ckpt_path, item=params)
    print(f"Checkpoint Loaded from: {latest_ckpt_path}")
else:
    print(f"No saved ddpm model checkpoints found in {ckpt_dir}, please train the model before inference.")

# Denoising loop
num_samples = 8
os.makedirs(train_config["sample_dir"], exist_ok=True)

noise_pred_model = jax.jit(lambda p, x, t: model.apply({"params": p}, x, t))
key, subkey = jax.random.split(key)
xt = jax.random.normal(subkey, (num_samples, im_size, im_size, model_config["img_channels"]))

for t in tqdm(reversed(range(diffusion_config["num_timesteps"])), total=diffusion_config["num_timesteps"]):
    t_batch = jnp.full((num_samples,), t, dtype=jnp.int32)
    noise_pred = noise_pred_model(params, xt, t_batch)
    if t==0:
        xt, _ = scheduler.sample_prev_timestep(xt, noise_pred, t, key)
    else:
        key, subkey = jax.random.split(key)
        xt, _, key = scheduler.sample_prev_timestep(xt, noise_pred, t, subkey)

images = np.array(jnp.clip(xt, -1.0, 1.0))
images = ((images + 1) / 2 * 255).astype(np.uint8)
for i, img in enumerate(images):
    Image.fromarray(img).save(os.path.join(train_config["sample_dir"], f"sample_{i}.png"))
print(f"Saved {num_samples} images to {train_config["sample_dir"]}")