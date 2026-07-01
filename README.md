# PAKR: Fast Asymptotically Optimal Kinodynamic Planning via Vectorization

**IROS 2026** | [Paper](https://arxiv.org/abs/2604.13323) | [Project Page](https://commalab.github.io/pakr)

PAKR is a massively parallel kinodynamic planner that uses JAX and the XLA compiler to achieve GPU-accelerated sampling-based planning entirely through standard Python tooling. Combined with the AO-x meta-algorithm, it achieves asymptotic optimality via fast iterative replanning.

---

## Setup

Build and run the Docker container with GPU access:

```bash
docker build -t pakr .
docker run --gpus all -it pakr
```

---

## Usage

### Basic Run

This is for the 9 kinopax problems. Run rrt.py. Inside it you can change batch size, branching factor, env, and dynamic model. Can also use the flags 
--env and --motion. Envs for tree, house, and narrow. For quadrotor, use quadtree, quadhouse, and quadnarrow envs since everything is scaled 100x.

### AO Runs (Asymptotically Optimal)

To run the AO-x version of the kinopax problems, run ao/aorrt.py with the same parameters that can be modified. There is an additional parameter for the cost threshold. 
The reason for this is because JAX does not allow us to interrupt a program, so where normally if the best cost is 
too low and the resource allotment is exceeded without finding a solution and the program exits, here we cannot interrupt and the entire program hangs.
So instead, we have to set a cost threshold that can find a solution in a reasonable time, so if we get "lucky" with initial runs and get a very low best cost,
it exits earlier.



### DynoBench Runs

Can be found in ./benchmarks. Working experiments include acrobot.py, cartpole.py, di2d.py (2d double integrator), qc.py, unicycle_acc.py, and unicycle_vel.py. 
Their corresponding ao scripts are also in the same folder. There are visualization scripts for unicycle_vel and di2d to show the solution trajectories.

### MJX Runs

MJX models are found in ./models. There are only 2 mjx experiments: eeonly, which is the block push, and cartpole. Acrobot does not use MJX but a simpler 
custom propagator, but can use the mjx models to visualize. To visualize mjx solutions, first run parse_solutions_mjx.py. This reruns the mjx propagator with 
the solution trajectory and saves each state. This is necessary because the mjx simulator has significantly different outputs compared to the mujoco sim, so 
we can't just run the solution in mujoco, but we want to use the mujoco viewer (mjx does not have one since everything is batched)
---

## Custom Environments and Dynamics

PAKR is designed to be modular. To define your own system, you replace three core functions:

- **Propagation** — forward-simulates the system given a state and action over a time step
- **Collision checking** — determines whether a state is valid (obstacle-free)
- **Distance** — defines the metric used for nearest-neighbor selection in the tree

_Placeholder: detailed instructions and example templates for swapping in custom propagation, checking, and distance functions._

---

## Citation

```bibtex
@misc{gao2026pakr,
  title={PAKR: Fast Asymptotically Optimal Kinodynamic Planning via Vectorization},
  author={Yitian Gao and Andrew Lu and Zachary Kingston},
  year={2026},
  eprint={2604.13323},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2604.13323},
}
```
