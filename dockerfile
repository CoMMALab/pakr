# Use the devel image so we have NVCC for JAX/XLA compilation
FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

# 1. System-level dependencies for MuJoCo and Python
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    wget \
    git \
    libgl1-mesa-dev \
    libosmesa6-dev \
    libglew-dev \
    libglfw3 \
    libglfw3-dev \
    xorg-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Set up environment variables
ENV MUJOCO_GL=egl
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# 3. Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# 4. Install JAX with CUDA 12 support
RUN pip3 install --no-cache-dir "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

CMD ["/bin/bash"]
# 5. Install MuJoCo + MJX explicitly
# mujoco-mjx is a SEPARATE package that provides the `mujoco.mjx` namespace
# Pin both to the same version to avoid conflicts
RUN pip3 install --no-cache-dir \
    "mujoco==3.3.5" \
    "mujoco-mjx==3.3.5" \
    numpy \
    scipy \
    matplotlib \
    pandas \
    flax \
    optax \
    chex \
    PyYAML \
    jupyterlab

