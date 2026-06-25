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

_Placeholder: instructions for running a single planning query._

### AO Runs (Asymptotically Optimal)

_Placeholder: instructions for running PAKR under the AO-x meta-algorithm with iterative replanning and cost thresholds._

### DynoBench Runs

_Placeholder: instructions for reproducing DynoBench experiments (unicycle, acrobot, quadrotor) and comparisons against iDb-A\* and SST\*._

### MJX Runs

_Placeholder: instructions for running MuJoCo-XLA experiments (cartpole, block push) with the MJX physics backend._

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
