# JAX Experiments

Exploring JAX and XLA through DL models: DDPM, ViT, etc. 

**Note: All models run on the GPU using Cuda, so make sure you have cuda-toolkit installed and install [jax](https://docs.jax.dev/en/latest/installation.html) matching your cuda version (check using `nvcc --version` or `nvidia-smi`).**

## DDPM

Denoising Diffusion Probabilistic Models using Jax.

Currently trains on CIFAR100, to run the training loop modify config.yaml as required and:
```bash
# Activate your virtual environment
pip install -r req.txt # Make sure you set the jax cuda version based on your cuda version
cd ddpm
python3 ddpm.py
```

To run the inference loop (make sure to comment out line 478 in [ddpm.py](ddpm/ddpm.py) to disable the training loop before running inference):
```bash
cd ddpm
python3 ddpm_infer.py
```

## todo
1. Implement DDPM training loop on custom datasets
2. Benchmarking for the DDPM inference results
