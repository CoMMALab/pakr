# =========================
# Dockerfile for Python + MuJoCo + JAX CUDA 13
# =========================

FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

# -------------------------
# Install basic utilities
# -------------------------
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    git \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    xorg-dev \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Install micromamba
# -------------------------
ENV MAMBA_ROOT_PREFIX=/opt/conda
RUN wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /usr/local/bin/ --strip-components=1 bin/micromamba

# -------------------------
# Copy environment file
# -------------------------
COPY env.yml /tmp/environment.yml

# -------------------------
# Create micromamba environment
# -------------------------
RUN micromamba create -y -f /tmp/environment.yml && \
    micromamba clean --all --yes

# -------------------------
# Activate environment automatically
# -------------------------
ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV CONDA_DEFAULT_ENV=mjx
ENV PATH=$MAMBA_ROOT_PREFIX/envs/mjx/bin:$PATH

WORKDIR /workspace

# -------------------------
# Install GPU-enabled JAX
# -------------------------
RUN pip install --upgrade pip && \
    pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
CMD ["/bin/bash"]