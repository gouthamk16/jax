## WIP
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = (
    "false"  # don't allocate 90% of GPU upfront
)

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import tqdm
import torch
from PIL import Image
from pathlib import Path

jax.config.update("jax_default_device", jax.devices("gpu")[0])

## Linear noise scheduler
class LinearNoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = jnp.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumulative_prod = jnp.cumprod(self.alphas, axis=0)
        self.sqrt_alpha_cumulative_prod = jnp.sqrt(self.alpha_cumulative_prod)
        self.sqrt_one_minus_alpha_cumulative_prod = jnp.sqrt(
            1 - self.alpha_cumulative_prod
        )

    def add_noise(self, original, noise, t):
        original_shape = original.shape
        batch_size = original_shape[0]

        sqrt_alpha_cumulative_prod = self.sqrt_alpha_cumulative_prod[t].reshape(
            batch_size
        )  # Reshaped to bx1x1x1
        sqrt_one_minus_alpha_cumulative_prod = (
            self.sqrt_one_minus_alpha_cumulative_prod[t].reshape(batch_size)
        )  # Reshaped to bx1x1x1

        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cumulative_prod = sqrt_alpha_cumulative_prod[..., None]
            sqrt_one_minus_alpha_cumulative_prod = sqrt_one_minus_alpha_cumulative_prod[
                ..., None
            ]

        # Apply and return forward process equation
        return (
            sqrt_alpha_cumulative_prod * original
            + sqrt_one_minus_alpha_cumulative_prod * noise
        )

    # takes in xt (a noisy image at step t) and the UNet's noise_pred (its guess of what noise is in the image), and outputs x(t-1) (a slightly less noisy image)
    def sample_prev_timestep(self, xt, noise_pred, t, key):
        x0 = (
            xt - (self.sqrt_one_minus_alpha_cumulative_prod[t] * noise_pred)
        ) / jnp.sqrt(self.alpha_cumulative_prod[t])
        x0 = jnp.clip(x0, -1.0, 1.0)
        mean = (
            xt
            - ((self.betas[t]) * noise_pred)
            / (self.sqrt_one_minus_alpha_cumulative_prod[t])
        )
        mean = mean / jnp.sqrt(self.alphas[t])
        if t == 0:
            return mean, x0
        else:
            variance = (1 - self.alpha_cumulative_prod[t - 1]) / (
                1.0 - self.alpha_cumulative_prod[t]
            )
            variance = variance * self.betas[t]
            sigma = variance**0.5
            key, subkey = jax.random.split(key)
            z = jax.random.normal(subkey, shape=xt.shape)
            return mean + sigma * z, x0, key


# Time embedding
def get_time_embedding(time_steps, temb_dim):
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"

    # factor = 10000^(2i/d_model)
    factor = 10000 ** (
        jnp.arange(start=0, stop=temb_dim // 2, dtype=jnp.float32) / (temb_dim // 2)
    )
    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = (
        time_steps[:, None] / factor
    )  # (B,1) / (temb_dim//2,) broadcasts to (B, temb_dim//2)
    t_emb = jnp.concatenate([jnp.sin(t_emb), jnp.cos(t_emb)], axis=-1)
    return t_emb


## The downsampling block
class DownBlock(nn.Module):
    num_heads: Any
    num_layers: Any
    down_sample: Any
    in_channels: Any
    out_channels: Any
    t_embed_dim: Any

    def setup(self):
        self.resnet_conv_first = [
            nn.Sequential(
                [
                    nn.GroupNorm(num_groups=8),
                    nn.activation.silu,
                    nn.Conv(
                        features=self.out_channels,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding=1,
                    ),
                ]
            )
            for _ in range(self.num_layers)
        ]
        self.resnet_conv_second = [
            nn.Sequential(
                [
                    nn.GroupNorm(num_groups=8),
                    nn.activation.silu,
                    nn.Conv(
                        features=self.out_channels,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding=1,
                    ),
                ]
            )
            for _ in range(self.num_layers)
        ]
        self.t_emb_layers = [
            nn.Sequential([nn.activation.silu, nn.Dense(self.out_channels)])
            for _ in range(self.num_layers)
        ]
        self.attention_norms = [
            nn.GroupNorm(num_groups=8) for _ in range(self.num_layers)
        ]
        self.attentions = [
            nn.MultiHeadDotProductAttention(num_heads=self.num_heads)
            for _ in range(self.num_layers)
        ]
        self.residual_input_conv = [
            nn.Conv(features=self.out_channels, kernel_size=(1, 1))
            for _ in range(self.num_layers)
        ]
        self.down_sample_conv = (
            nn.Conv(
                features=self.out_channels,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding=1,
            )
            if self.down_sample
            else (lambda x: x)
        )

    def __call__(self, x, t_emb, train: bool = False):
        out = x
        for i in range(self.num_layers):
            # ResNet block
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = out + self.t_emb_layers[i](t_emb)[:, None, None, :]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

            # Attention block
            B, H, W, C = out.shape
            in_attn = self.attention_norms[i](out.reshape(B, H * W, C))
            out_attn = self.attentions[i](in_attn, in_attn)
            out = out + out_attn.reshape(B, H, W, C)

        return self.down_sample_conv(out)


class MidBlock(nn.Module):
    num_heads: Any
    num_layers: Any
    in_channels: Any
    out_channels: Any
    t_emb_dim: Any

    def setup(self):
        # num_layers+1 resnet blocks, num_layers attention blocks
        self.resnet_conv_first = [
            nn.Sequential(
                [
                    nn.GroupNorm(num_groups=8),
                    nn.activation.silu,
                    nn.Conv(
                        features=self.out_channels,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding=1,
                    ),
                ]
            )
            for _ in range(self.num_layers + 1)
        ]
        self.resnet_conv_second = [
            nn.Sequential(
                [
                    nn.GroupNorm(num_groups=8),
                    nn.activation.silu,
                    nn.Conv(
                        features=self.out_channels,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding=1,
                    ),
                ]
            )
            for _ in range(self.num_layers + 1)
        ]
        self.t_emb_layers = [
            nn.Sequential([nn.activation.silu, nn.Dense(self.out_channels)])
            for _ in range(self.num_layers + 1)
        ]
        self.attention_norms = [
            nn.GroupNorm(num_groups=8) for _ in range(self.num_layers)
        ]
        self.attentions = [
            nn.MultiHeadDotProductAttention(num_heads=self.num_heads)
            for _ in range(self.num_layers)
        ]
        self.residual_input_conv = [
            nn.Conv(features=self.out_channels, kernel_size=(1, 1))
            for _ in range(self.num_layers + 1)
        ]

    def __call__(self, x, t_emb, train: bool = False):
        out = x

        # First resnet block
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        out = out + self.t_emb_layers[0](t_emb)[:, None, None, :]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)

        for i in range(self.num_layers):
            # Attention block
            B, H, W, C = out.shape
            in_attn = self.attention_norms[i](out.reshape(B, H * W, C))
            out_attn = self.attentions[i](in_attn, in_attn)
            out = out + out_attn.reshape(B, H, W, C)

            # Resnet block
            resnet_input = out
            out = self.resnet_conv_first[i + 1](out)
            out = out + self.t_emb_layers[i + 1](t_emb)[:, None, None, :]
            out = self.resnet_conv_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](resnet_input)

        return out


class UpBlock(nn.Module):
    num_heads: Any
    num_layers: Any
    in_channels: Any
    out_channels: Any
    t_emb_dim: Any
    up_sample: bool = True

    def setup(self):
        self.resnet_conv_first = [
            nn.Sequential(
                [
                    nn.GroupNorm(num_groups=8),
                    nn.activation.silu,
                    nn.Conv(
                        features=self.out_channels,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding=1,
                    ),
                ]
            )
            for _ in range(self.num_layers)
        ]
        self.resnet_conv_second = [
            nn.Sequential(
                [
                    nn.GroupNorm(num_groups=8),
                    nn.activation.silu,
                    nn.Conv(
                        features=self.out_channels,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding=1,
                    ),
                ]
            )
            for _ in range(self.num_layers)
        ]
        self.t_emb_layers = [
            nn.Sequential([nn.activation.silu, nn.Dense(self.out_channels)])
            for _ in range(self.num_layers)
        ]
        self.attention_norms = [
            nn.GroupNorm(num_groups=8) for _ in range(self.num_layers)
        ]
        self.attentions = [
            nn.MultiHeadDotProductAttention(num_heads=self.num_heads)
            for _ in range(self.num_layers)
        ]
        self.residual_input_conv = [
            nn.Conv(features=self.out_channels, kernel_size=(1, 1))
            for _ in range(self.num_layers)
        ]
        self.up_sample_conv = (
            nn.ConvTranspose(
                features=self.in_channels // 2,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding="SAME",
            )
            if self.up_sample
            else (lambda x: x)
        )  ## In Flax, padding=1 on a ConvTranspose pads the dilated input (not the output), which gives 14×14 instead of 16×16. The skip connection from the down path is correctly 16×16 (regular Conv padding works fine).

    def __call__(self, x, out_down, t_emb, train: bool = False):
        x = self.up_sample_conv(x)
        out = jnp.concatenate([x, out_down], axis=-1)

        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = out + self.t_emb_layers[i](t_emb)[:, None, None, :]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

            B, H, W, C = out.shape
            in_attn = self.attention_norms[i](out.reshape(B, H * W, C))
            out_attn = self.attentions[i](in_attn, in_attn)
            out = out + out_attn.reshape(B, H, W, C)

        return out


class Unet(nn.Module):
    img_channels: Any
    down_channels: Any
    mid_channels: Any
    t_emb_dim: Any
    down_sample: Any
    num_down_layers: Any
    num_mid_layers: Any
    num_up_layers: Any

    def setup(self):
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1

        self.t_proj = nn.Sequential(
            [
                nn.Dense(self.t_emb_dim),
                nn.activation.silu,
                nn.Dense(self.t_emb_dim),
            ]
        )
        self.conv_in = nn.Conv(
            features=self.down_channels[0], kernel_size=(3, 3), padding=1
        )

        self.downs = [
            DownBlock(
                num_heads=4,
                num_layers=self.num_down_layers,
                in_channels=self.down_channels[i],
                out_channels=self.down_channels[i + 1],
                t_embed_dim=self.t_emb_dim,
                down_sample=self.down_sample[i],
            )
            for i in range(len(self.down_channels) - 1)
        ]
        self.mids = [
            MidBlock(
                num_heads=4,
                num_layers=self.num_mid_layers,
                in_channels=self.mid_channels[i],
                out_channels=self.mid_channels[i + 1],
                t_emb_dim=self.t_emb_dim,
            )
            for i in range(len(self.mid_channels) - 1)
        ]
        self.ups = [
            UpBlock(
                num_heads=4,
                num_layers=self.num_up_layers,
                in_channels=self.down_channels[i] * 2,
                out_channels=self.down_channels[i - 1] if i != 0 else 16,
                t_emb_dim=self.t_emb_dim,
                up_sample=self.down_sample[i],
            )
            for i in reversed(range(len(self.down_channels) - 1))
        ]
        self.norm_out = nn.GroupNorm(num_groups=8)
        self.conv_out = nn.Conv(
            features=self.img_channels, kernel_size=(3, 3), padding=1
        )

    def __call__(self, x, t, train: bool = False):
        out = self.conv_in(x)

        t_emb = get_time_embedding(jnp.asarray(t, dtype=jnp.int32), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        down_outs = []
        for down in self.downs:
            down_outs.append(out)
            out = down(out, t_emb)

        for mid in self.mids:
            out = mid(out, t_emb)

        for up in self.ups:
            out = up(out, down_outs.pop(), t_emb)

        out = self.norm_out(out)
        out = nn.activation.silu(out)
        return self.conv_out(out)

# Some helper functions for training and data loading
def get_latest_checkpoint(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return None, 0

    latest_epoch = 0
    latest_path = None
    for name in os.listdir(ckpt_dir):
        if not name.startswith("epoch_"):
            continue
        try:
            epoch = int(name.removeprefix("epoch_"))
        except ValueError:
            continue
        path = os.path.join(ckpt_dir, name)
        if os.path.isdir(path) and epoch > latest_epoch:
            latest_epoch = epoch
            latest_path = path

    return latest_path, latest_epoch

class ImageDataset(torch.utils.data.Dataset):
    EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
    def __init__(self, folder, transform=None):
        self.paths = [p for p in Path(folder).rglob('*') if p.suffix.lower() in self.EXTS]
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img) if self.transform else img, 0


def train():
    # Training loop
    import numpy as np
    import optax
    import orbax.checkpoint as ocp
    import torchvision
    import torchvision.transforms as transforms
    import yaml

    # One problem - jax does not have a built in dataloader, have to still use torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    # Load training and model configs in config.yaml
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    diffusion_config = config["diffusion"]
    dataset_config = config["dataset"]
    model_config = config["model"]
    train_config = config["training"]
    checkpointer = ocp.PyTreeCheckpointer()
    ckpt_dir = os.path.abspath(train_config["ckpt_dir"])
    im_size = dataset_config["im_size"]

    # CIFAR100
    transform = transforms.Compose(
        [
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    if dataset_config["name"].lower() == "cifar100":
        dataset = torchvision.datasets.CIFAR100(
            root=dataset_config["path"], train=True, download=True, transform=transform
        )
    elif dataset_config["name"].lower() == "custom":
        dataset = ImageDataset(dataset_config["path"], transform=transform)
    else:
        print(f"No dataset loader implemented yet for dataset: {dataset_config["name"]}. Available options: 'custom','cifar100'")
        return
    loader = DataLoader(
        dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # Initialize scheduler
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config["num_timesteps"],
        beta_start=diffusion_config["beta_start"],
        beta_end=diffusion_config["beta_end"],
    )
    # Initialize the model
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

    key = jax.random.PRNGKey(train_config["seed"])
    key, init_key = jax.random.split(key)

    dummy_x = jnp.ones((1, im_size, im_size, model_config["img_channels"]))
    dummy_t = jnp.ones((1,), dtype=jnp.int32)
    params = model.init(init_key, dummy_x, dummy_t)["params"]

    latest_ckpt_path, start_epoch = get_latest_checkpoint(ckpt_dir)
    if latest_ckpt_path is not None:
        print(f"Resuming from checkpoint: {latest_ckpt_path}", flush=True)
        params = checkpointer.restore(latest_ckpt_path, item=params)
    else:
        print(f"No saved checkpoint found in {ckpt_dir}, starting training from 0")

    # Optimizer setup
    learning_rate = float(train_config["lr"])
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    # Loss Function
    def loss_fn(params, batch, noise, t):
        noisy_img = scheduler.add_noise(batch, noise, t)
        noise_pred = model.apply({"params": params}, noisy_img, t)
        return jnp.mean((noise_pred - noise) ** 2)

    # Train Step and loop
    @jax.jit
    def train_step(params, opt_state, batch, noise, t):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch, noise, t)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    best_mean_loss = float("inf")  # To check whether the running loss is improving

    for epoch in range(start_epoch, train_config["num_epochs"]):
        losses = []
        progress_bar = tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{train_config['num_epochs']}",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch, _ in progress_bar:
            batch = jnp.array(batch)  # Converting torch tensor -> jnp array
            batch = jnp.transpose(batch, (0, 2, 3, 1))  # NCHW -> NHWC
            key, noise_key, t_key = jax.random.split(key, 3)
            noise = jax.random.normal(noise_key, batch.shape)
            t = jax.random.randint(
                t_key, (batch.shape[0],), 0, diffusion_config["num_timesteps"]
            )
            params, opt_state, loss = train_step(params, opt_state, batch, noise, t)
            loss_value = float(loss)
            losses.append(loss_value)
            progress_bar.set_postfix(
                loss=f"{loss_value:.4f}",
                avg_loss=f"{np.mean(losses):.4f}",
            )
        epoch_mean_loss = np.mean(losses)
        tqdm.write(f"Epoch {epoch + 1} | Loss: {epoch_mean_loss:.4f}")
        if epoch_mean_loss < best_mean_loss:
            best_mean_loss = epoch_mean_loss
            checkpointer.save(
                os.path.join(ckpt_dir, f"epoch_{epoch + 1}"), params, force=True
            )


import argparse
parser = argparse.ArgumentParser(description="DDPM training script.")
parser.add_argument("train", help="Initialize the training loop.", type=bool, default=True)
args = parser.parse_args()
if args.train:
    train()
