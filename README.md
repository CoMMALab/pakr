# Batched SST for icra submission

Batched sst with jax for gpu acceleration. Not all parts can be jaxed, details later.

Compare with baselines: 

    kinoPAX: arxiv.org/abs/2409.06807
    ompl's non-batched sst: github.com/ompl/ompl

Motion Primitives:

    6d Double Integrator
    6d Dubin's Airplane
    12d Quadcopter

Some parts can connect to mujoco/mjx but mujoco is not used for the planner.
All forward propagation is done with hard-coded physics equations

## Setup and dependencies

python 3.10.18
jax jaxlib jax-cuda 0.6.2
flax 0.10.7
mujoco mjx 3.3.5
cuda 12.5

to install:

```bash
micromamba create -n batch_sst python=3.10 jax flax "jaxlib=*=*cuda*" -c conda-forge
micromamba activate batch_sst
pip install mujoco mujoco-mjx
```

## Run examples

## Jax and jitted processes

All forward propagations are jax compatible. Batched rollouts are vmapped. Within each rollout, jax.lax.scan per timestep

All helpers are jax compatible (samplers, validators, distance, collistion check) excluding get_obs which uses pandas

All params are jax immutable dataclasses and set as static for jit. Most are constants and set in the very beginning. 
Exceptions are δBN and δs which are updated between sst* iterations, so we store the initial values

SSTree is jax compatible. In-place np mutations are replaces with batched jnp.at[].set(). The class itself cannot be passed into jax functions but its
member functions are jit compiled

best_nearest, check_dominating_nodes are jax compatible. Advanced indexing (array[indices] with a batch of indices) replaced with jnp.take_along_axis()

sst and sst* are not jax compatible. The outer loop includes witness pruning and storing solutions using a heapq. The inner loop has to pass the SSTree object.
It is still fast because it is manually vectorized
