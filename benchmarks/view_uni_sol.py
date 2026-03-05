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
from benchmarks.unicycle_vel import valid_unicycle, sample_unicycle, dist_unicycle, sample_actions_unicycle, reached_goal_unicycle

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


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def plot_all_results(data_path, obstacles):
    data = np.load(data_path, allow_pickle=True)
    solutions = data['solutions']
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Obstacle Gradient Setup
    obs_cmap = LinearSegmentedColormap.from_list("grey_grad", ["#d3d3d3", "#a9a9a9"])
    
    # Plot Obstacles
    for obs in obstacles:
        x1, y1, x2, y2 = obs[0], obs[1], obs[2], obs[3]
        gradient = np.linspace(0, 1, 100).reshape(-1, 1)
        ax.imshow(gradient, extent=[x1, x2, y1, y2], cmap=obs_cmap, aspect='auto', zorder=1)
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1.5, edgecolor='#505050', facecolor='none', zorder=2)
        ax.add_patch(rect)

    # 1. Goal Gradient Setup (Green to White radial gradient)
    radius = 0.05
    res = 50
    x_grid = np.linspace(-radius, radius, res)
    y_grid = np.linspace(-radius, radius, res)
    X, Y = np.meshgrid(x_grid, y_grid)
    R = np.sqrt(X**2 + Y**2)
    
    # Create radial alpha/color mask: 1.0 at center, 0.0 at edge
    goal_gradient = np.clip(1.0 - (R / radius), 0, 1)
    
    # 1. Goal Gradient Setup
    radius = 0.05
    res = 50
    x_grid = np.linspace(-radius, radius, res)
    y_grid = np.linspace(-radius, radius, res)
    X, Y = np.meshgrid(x_grid, y_grid)
    R = np.sqrt(X**2 + Y**2)
    
    # Create the gradient (1.0 at center, 0.0 at edge)
    goal_gradient = np.clip(1.0 - (R / radius), 0, 1)
    
    # Create a mask: set everything outside the radius to transparent (NaN or alpha 0)
    # Using np.ma (masked array) is the cleanest way to hide pixels outside the circle
    mask = R <= radius
    masked_gradient = np.ma.masked_where(~mask, goal_gradient)
    
    # 2. Plot the Masked Gradient
    ax.imshow(masked_gradient, extent=[sst_params.goal.x-radius, sst_params.goal.x+radius, 
                                       sst_params.goal.y-radius, sst_params.goal.y+radius], 
              cmap='Greens', alpha=0.9, zorder=4)
    
    # 3. Add the Outline
    circle_outline = plt.Circle((sst_params.goal.x, sst_params.goal.y), radius, 
                                color='green', fill=False, linewidth=1.5, zorder=5)
    ax.add_patch(circle_outline)

    # 3. Plot Paths
    for sol in solutions:
        full_traj = rollout_full_trajectory(sol['path'][0], sol['actions'], sst_params, sim_params, callables.prop_fn)
        ax.plot(full_traj[:, 0], full_traj[:, 1], color='blue', alpha=0.15, linewidth=1, zorder=3)
        
    # Start Marker
    ax.scatter(sst_params.start.x, sst_params.start.y, color='blue', s=20, label='Start', zorder=5)

    # Clean axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    
    ax.set_xlim(sim_params.bounds.min_x, sim_params.bounds.max_x)
    ax.set_ylim(sim_params.bounds.min_y, sim_params.bounds.max_y)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('benchmarks/uni_solutions.png', bbox_inches='tight', pad_inches=0)
    print("Saved visualization.")

# Run plotting

batch_size = 2048
time_to_evolve = 10


bounds = Bounds(
    min_x = 0.0, max_x = 1.0,
    min_y = 0.0, max_y = 1.0,
    min_z = 0.0, max_z = 0.0 # Unused in 2D
)

start_pos = Position(x = 0.05, y = 0.05, z = 0.0)
goal_pos = Position(x = 0.95, y = 0.95, z = 0.0)

motion_constraints = MotionConstraints(
    max_vel = 0.4,       # Max linear velocity
    min_vel = 0.0,       # Unicycle usually moves forward
    max_accel = 1.5,     # Used here as Max Angular Velocity (omega)
    min_accel = -1.5)

sim_params = MJXparams(
    motion_constraints=motion_constraints,
    physics_constants=PhysicsConstants(),
    batch_size=batch_size,
    bounds = Bounds(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0, min_z=0.0, max_z=0.0),
    dims=3,              # [x, y, theta]
    action_dims=2,       # [v, omega]
    dt = 0.1,
    seed = 42
)

sst_params = SSTparams(
    batch_size=batch_size,
    δBN=0.04, δs=0.02, decay=0.8,
    start=Position(x=0.05, y=0.05, z=0.0),
    goal=Position(x=0.95, y=0.95, z=0.0),
    goal_radius=0.05,
    time_to_evolve=time_to_evolve,
)

callables = Callables(
    prop_fn=propagate.propagate_unicycle,
    valid_fn=valid_unicycle,
    sample_fn=sample_unicycle,
    dist_fn=dist_unicycle,
    sampact_fn=sample_actions_unicycle,
    goal_fn=reached_goal_unicycle
)
obstacles = helper.get_obs('envs/tree2d.csv')

plot_all_results('benchmarks/uni_results.npz', obstacles)