# MrCogito




## Installing dependencies

Most of the dependencies are installed via poetry. 

For pytorch I have added a ["explicit" Poetry source for CUDA 12.4](pyproject.toml)

Installing unsloth with CUDA 12.4 is done as follows for GTX 3090 was done accoridng to the [unsloth documentation](https://github.com/unslothai/unsloth/tree/main#-installation-instructions)


Installation procedure for unsloth is as follows (date 31.12.2024):

1. install packages from poetry

```bash
poetry install
```
1. spawn a shell with "poetry shell" and install unsloth

```bash
pip install --upgrade pip
pip install "unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git"
```

**Windows installation**

To run Unsloth directly on Windows:

Install Triton from this Windows fork and follow the instructions: https://github.com/woct0rdho/triton-windows



And then I have added the `torch` dependency as follows:

```bash
poetry add torch==2.2.1+cu121 --source torch
poetry add triton --source torch
```