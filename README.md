# JAX Experiments

Exploring JAX and XLA through DL models: DDPM, ViT, etc. 

**Note: All models run on the GPU using Cuda, so make sure you have cuda-toolkit installed and install [jax](https://docs.jax.dev/en/latest/installation.html) matching your cuda version (check using `nvcc --version` or `nvidia-smi`).**

## DDPM

Denoising Diffusion Probabilistic Models using Jax.

Currently trains on CIFAR100 by default (to run the training loop on a custom folder of images, set name to `custom` and the corresponding path to the folder in [config.yaml](ddpm/config.yaml)).
```bash
# Activate your virtual environment
pip install -r req.txt # Make sure you set the jax cuda version based on your cuda version
cd ddpm
python3 ddpm.py
```

To run the inference loop:
```bash
cd ddpm
python3 ddpm_infer.py
```

## todo
1. Benchmark the DDPM inference results
