import matplotlib.pyplot as plt
from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
import rrtree
import sys
import os
import argparse
import propagate
import params
import helper
import time
import gc
import numpy as np
from params import Bounds, Position, MotionConstraints, PhysicsConstants, MJXparams, SSTparams, Callables
from benchmarks.di2d import valid_DI2, sample_DI2, dist_DI2, sample_actions_DI2, reached_goal_DI2

def rollout_full_trajectory(start_state, actions, sst_params, sim_params, prop_fn):
    """Reconstructs every intermediate state for a sequence of actions."""
    steps_per_action = sst_params.time_to_evolve
    dt = sim_params.dt
    
    # Ensure start_state is [1, Dims] for the prop_fn
    current_state = start_state[None, :] 
    trajectory = [start_state]

    for action in actions:
        # Action also needs a batch dim: [1, Action_Dims]
        batched_action = action[None, :]
        
        for _ in range(steps_per_action):
            # Propagate returns [1, Dims]
            current_state = prop_fn(current_state, batched_action, dt, sim_params.physics_constants)
            
            # Extract the 1D state to store it in our list
            trajectory.append(current_state.reshape(-1)) 
            
    return jnp.stack(trajectory)

import matplotlib.patches as patches

def plot_all_results(data_path, obstacles):
    data = np.load(data_path, allow_pickle=True)
    solutions = data['solutions']
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 1. Plot Obstacles as Rectangles
    # obs is [x1, y1, x2, y2]
    for obs in obstacles:
        width = obs[2] - obs[0]
        height = obs[3] - obs[1]
        rect = patches.Rectangle(
            (obs[0], obs[1]), width, height, 
            linewidth=1, edgecolor='red', facecolor='gray', alpha=0.5
        )
        ax.add_patch(rect)

    # 2. Reconstruct and Plot each solution path
    for sol in solutions:
        full_traj = rollout_full_trajectory(
            sol['path'][0], 
            sol['actions'], 
            sst_params, 
            sim_params, 
            callables.prop_fn
        )
        
        # Plot (x, y) coordinates
        ax.plot(full_traj[:, 0], full_traj[:, 1], color='blue', alpha=0.15, linewidth=1)
        
    # 3. Plot Start and Goal for context
    ax.scatter(sst_params.start.x, sst_params.start.y, color='green', s=100, label='Start', zorder=5)
    ax.scatter(sst_params.goal.x, sst_params.goal.y, color='gold', s=100, label='Goal', zorder=5)

    # 4. Format Plot
    ax.set_xlim(sim_params.bounds.min_x, sim_params.bounds.max_x)
    ax.set_ylim(sim_params.bounds.min_y, sim_params.bounds.max_y)
    ax.set_aspect('equal')
    ax.set_title(f"2D Double Integrator: {len(solutions)} Successful Paths")
    ax.legend()
    
    plt.savefig('benchmarks/di2d_solutions.png')
    print(f"Saved visualization to benchmarks/di2d_solutions.png")

# Run plotting

batch_size = 4096
time_to_evolve = 30

motion_constraints = MotionConstraints(
    max_vel = 0.5,
    min_vel = -0.5,
    max_accel = 1.0,
    min_accel = -1.0)

bounds = Bounds(
    min_x = 0.0, max_x = 1.0,
    min_y = 0.0, max_y = 1.0,
    min_z = 0.0, max_z = 0.0 # Unused in 2D
)

start_pos = Position(x = 0.05, y = 0.05, z = 0.0)
goal_pos = Position(x = 0.95, y = 0.95, z = 0.0)

sim_params = MJXparams(
    motion_constraints=motion_constraints,
    physics_constants=PhysicsConstants(),
    batch_size=batch_size,
    bounds = bounds,
    dims=4,        # [x, y, vx, vy]
    action_dims=2, # [ax, ay]
    dt = 0.02,
    seed = 42
)

sst_params = SSTparams(
    batch_size=batch_size,
    δBN=0.04,
    δs=0.02,
    decay=0.8,
    start=start_pos,
    goal=goal_pos,
    goal_radius=0.05,
    geo_cost_to_go_weight=0.2,
    do_cost_to_go=True,
    do_maximal= True,
    do_set_cover= True,
    time_to_evolve= time_to_evolve,
    sparsity = 0,
)

callables = Callables(
    prop_fn=propagate.propagate_2d_integrator,
    valid_fn=valid_DI2,
    sample_fn=sample_DI2,
    dist_fn=dist_DI2,
    sampact_fn=sample_actions_DI2,
    goal_fn=reached_goal_DI2
)
obstacles = helper.get_obs('envs/tree2d.csv')

plot_all_results('benchmarks/di2d_results.npz', obstacles)