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
import plotly.graph_objects as go
# ------------------------------------------------------------------
# 1. Tiered Nearest Neighbor Kernels
# ------------------------------------------------------------------

# Global references to allow the switch branches to access static objects via closure
SIM_PARAMS_RESERVED = None
CALLABLES_RESERVED = None

def nn_tier_factory(size):
    """Generates a function for lax.switch that scans a fixed slice of the tree."""
    def nn_fn(operands):
        states, tree_size, query = operands
        sliced_states = states[:size]
        
        # Access static objects from outer scope
        parents, _ = helper.nearest_neighbor_masked(
            SIM_PARAMS_RESERVED, 
            CALLABLES_RESERVED.dist_fn, 
            sliced_states, 
            tree_size, 
            query
        )
        return parents
    return nn_fn

# Define memory buckets
TIERS = [512, 1024, 4096, 16384, 32768, 64536, 131_072, 262_144, 500_000]
NN_BRANCHES = [nn_tier_factory(t) for t in TIERS]



print("JAX version:", jax.__version__)
print("Devices:", jax.devices())

@partial(jax.jit, static_argnums=(3, 4, 5))
def rrt_iteration(tree, rng_key, obstacles, sst_params, sim_params, callables):
    """One batched kinodynamic RRT iteration with reduced NN calls (B/64 parents × 64 actions)."""

    B = sim_params.batch_size
    K = B // A  # number of NN queries

    rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)

    # -------------------------------------------------
    # 1. Sample only K target points
    # -------------------------------------------------
    seed_pts = callables.sample_fn(sim_params, subkey1)
    seed_pts = seed_pts[:K]

    # -------------------------------------------------
    # 2. Tiered Nearest Neighbor Lookup (only K queries)
    # -------------------------------------------------
    branch_idx = jnp.digitize(tree.tree_size, jnp.array(TIERS))
    branch_idx = jnp.minimum(branch_idx, len(TIERS) - 1)

    operands = (tree.states, tree.tree_size, seed_pts)
    parents_small = jax.lax.switch(branch_idx, NN_BRANCHES, operands)  # shape (K,)

    start_states_small = tree.states[parents_small]  # (K, dims)

    # -------------------------------------------------
    # 3. Repeat each parent 64 times
    # -------------------------------------------------
    start_states = jnp.repeat(start_states_small, A, axis=0)  # (B, dims)
    parents = jnp.repeat(parents_small, A, axis=0)            # (B,)

    # -------------------------------------------------
    # 4. Sample B actions (64 per parent implicitly)
    # -------------------------------------------------
    actions = callables.sampact_fn(sim_params, subkey2)  # (B, action_dims)

    # -------------------------------------------------
    # 5. Rollout
    # -------------------------------------------------
    states_end, valid_mask, dist_traveled = propagate.rollout_final(
        start_states, actions, obstacles, sst_params, sim_params, callables
    )

    # -------------------------------------------------
    # 6. Static padding for JIT shape safety
    # -------------------------------------------------
    valid_mask = valid_mask.at[-1].set(False)
    states_end = states_end.at[-1].set(jnp.zeros(sim_params.dims))
    actions = actions.at[-1].set(jnp.zeros(sim_params.action_dims))
    parents = parents.at[-1].set(-1)
    dist_traveled = dist_traveled.at[-1].set(0.0)

    # -------------------------------------------------
    # 7. Filter valid insertions
    # -------------------------------------------------
    num_new = jnp.sum(valid_mask)
    valid_idx = jnp.nonzero(valid_mask, size=B, fill_value=-1)[0]

    new_states = states_end[valid_idx]
    new_actions = actions[valid_idx]
    new_parents = parents[valid_idx]
    new_costs = tree.costs[new_parents] + dist_traveled[valid_idx]

    # -------------------------------------------------
    # 8. Insert nodes
    # -------------------------------------------------
    tree, start_idx = rrtree.add_nodes(
        tree, new_states, new_actions, new_parents, new_costs, num_new
    )

    # -------------------------------------------------
    # 9. Goal check
    # -------------------------------------------------
    goal_mask = helper.reached_goal(
        new_states, sst_params.goal, sst_params.goal_radius
    )

    return tree, rng_key, goal_mask, jnp.sum(goal_mask), new_states, start_idx

@partial(jax.jit, static_argnums=(1, 2, 3))
def jit_while(tree, sst_params, sim_params, callables, obstacles, i):
    def body_fn(carry):
        tree, key, goal_mask, goal, states, start_idx, iter = carry
        key, subkey = jax.random.split(key)
        tree, subkey, goal_mask, goal, states, start_idx = rrt_iteration(
            tree, subkey, obstacles, sst_params, sim_params, callables
        )
        return (tree, key, goal_mask, goal, states, start_idx, iter + 1)

    def cond_fn(carry):
        tree, key, goal_mask, goal, states, start_idx, iter = carry
        return (goal == 0) & (tree.tree_size < MAX_TREE_SIZE - sim_params.batch_size)

    init_carry = (tree, 
                  jax.random.PRNGKey(i),
                  jnp.zeros(sim_params.batch_size, dtype=bool),
                  jnp.array(0, dtype=jnp.int32),
                  jnp.zeros([sim_params.batch_size, sim_params.dims], dtype=jnp.float32),
                  jnp.array(0, dtype=jnp.int32),
                  jnp.array(0, dtype=jnp.int32))

    tree, key, goal_mask, goal, states, start_idx, iter = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    return tree, key, goal_mask, goal, states, start_idx, iter, tree.tree_size

def extract_sol(tree, goal_mask, start_idx):
    if jnp.sum(goal_mask) == 0:
        print(tree.tree_size)
        print("error: no goal reached")
        return None, None
    
    #print(jnp.sum(goal_mask))

    goal_idxs = jnp.argmax(goal_mask)
    goal_idx = goal_idxs + start_idx
    path = []
    actions = []
    while goal_idx != -1:
        path.append(tree.states[goal_idx])
        actions.append(tree.actions[goal_idx])
        goal_idx = tree.parents[goal_idx]
    return jnp.array(path[::-1]), jnp.array(actions[::-1])

def verify_sol(path, actions, obstacles, sst_params, sim_params, callables):
    for i in range(len(actions)-1):
        start = path[i][None, :]    # shape: (1, state_dim)
        action = actions[i+1][None, :]
        states_end, valid_mask, _ = propagate.rollout_final(
            start, action, obstacles, sst_params, sim_params, callables
        )
        #print(states_end)
        if not valid_mask:
            return False
    return True


    # ... (Keep all your existing imports and JAX kernels above) ...

import plotly.express as px
def create_box_mesh(x1, y1, z1, x2, y2, z2):
    """
    Returns vertices and triangle indices for a box defined
    by min corner (x1,y1,z1) and max corner (x2,y2,z2)
    """

    # 8 vertices of box
    x = [x1, x1, x2, x2, x1, x1, x2, x2]
    y = [y1, y2, y2, y1, y1, y2, y2, y1]
    z = [z1, z1, z1, z1, z2, z2, z2, z2]

    # 12 triangles (2 per face)
    i = [0, 0, 4, 4, 0, 1, 2, 3, 6, 6, 5, 4]
    j = [1, 2, 5, 6, 4, 5, 3, 7, 2, 7, 1, 0]
    k = [2, 3, 6, 7, 5, 6, 7, 6, 7, 4, 0, 5]

    return x, y, z, i, j, k


def add_box_edges(fig, x, y, z):
    """
    Adds explicit black edges to the figure for visual clarity.
    """

    edges = [
        (0,1),(1,2),(2,3),(3,0),      # bottom
        (4,5),(5,6),(6,7),(7,4),      # top
        (0,4),(1,5),(2,6),(3,7)       # vertical
    ]

    for e in edges:
        fig.add_trace(go.Scatter3d(
            x=[x[e[0]], x[e[1]]],
            y=[y[e[0]], y[e[1]]],
            z=[z[e[0]], z[e[1]]],
            mode='lines',
            line=dict(color='grey', width=3),
            opacity=0.5,
            showlegend=False
        ))


def rollout_full_trajectory(start_state, actions, sst_params, sim_params, callables):
    """Reconstructs every intermediate state for a sequence of actions using physics."""
    steps_per_action = sst_params.time_to_evolve
    dt = sim_params.dt
    
    current_state = start_state[None, :] 
    trajectory = [start_state]

    for action in actions:
        batched_action = action[None, :]
        for _ in range(steps_per_action):
            current_state = callables.prop_fn(
                current_state, batched_action, dt, sim_params.physics_constants
            )
            trajectory.append(current_state.reshape(-1)) 
            
    return jnp.stack(trajectory)

def get_sphere_mesh(x0, y0, z0, radius=0.05, resolution=20):
    """Generates vertices and triangles for a sphere at (x0, y0, z0)."""
    phi = np.linspace(0, np.pi, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi, theta = np.meshgrid(phi, theta)
    
    x = x0 + radius * np.sin(phi) * np.cos(theta)
    y = y0 + radius * np.sin(phi) * np.sin(theta)
    z = z0 + radius * np.cos(phi)
    
    # Flatten arrays for Mesh3d
    x, y, z = x.flatten(), y.flatten(), z.flatten()
    
    # Generate triangle indices
    triangles = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            v0 = i * resolution + j
            v1 = v0 + 1
            v2 = (i + 1) * resolution + j
            v3 = v2 + 1
            triangles.append([v0, v1, v2])
            triangles.append([v1, v3, v2])
            
    triangles = np.array(triangles)
    return x, y, z, triangles[:, 0], triangles[:, 1], triangles[:, 2]

def visualize_multi_trajectories(env_path, trajectories, sst_params, idx, output_name="./solution.html"):
    """Generates the 3D HTML visualization with obstacles and multiple planned paths."""
    
    # Load Environment Obstacles
    data = np.loadtxt(env_path, delimiter=',', skiprows=1)
    if data.ndim == 1: 
        data = data.reshape(1, -1)
    
    fig = go.Figure()

    # 1. Add Obstacles (Matching your Env Visualizer Style)
    for box in data:
        x1, y1, z1, x2, y2, z2 = box
        x, y, z, i, j, k = create_box_mesh(x1, y1, z1, x2, y2, z2)
        
        if idx == 0:
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z, i=i, j=j, k=k,
                color='grey',
                opacity=0.3,
                flatshading=False,
                # Adjusted lighting for uniform face color
                lighting=dict(
                    ambient=0.9, 
                    diffuse=0.0, 
                    specular=0.0, 
                    fresnel=0.0,
                    roughness=0.5
                ),
                showlegend=False
            ))
        else:
            fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k,
            color='lightgrey',
            opacity=0.3,
            flatshading=True,
            lighting=dict(
                ambient=0.05,
                diffuse=1.0,
                roughness=1.0,
                specular=0.0,
                fresnel=0.0
            ),
            lightposition=dict(x=5, y=5, z=10),
            showlegend=False
        ))
        add_box_edges(fig, x, y, z)

    # 2. Add Solution Paths (Uniform Green Color)
    for idx, traj in enumerate(trajectories):
        fig.add_trace(go.Scatter3d(
            x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
            mode='lines', 
            line=dict(color='limegreen', width=7.5), 
            name=f'Solution {idx+1}',
            legendgroup="Solutions",
            showlegend=(idx == 0) # Only show one legend entry for all green lines
        ))

    # 3. Add Start and Goal Markers
    fig.add_trace(go.Scatter3d(
        x=[sst_params.start.x], y=[sst_params.start.y], z=[sst_params.start.z],
        mode='markers', 
        marker=dict(size=6, color='blue'), 
        name='Start'
    ))
    
    x, y, z, i, j, k = get_sphere_mesh(sst_params.goal.x, sst_params.goal.y, sst_params.goal.z, radius=0.05)

    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        color='limegreen', 
        opacity=0.6,
        showlegend=True,
        name='Goal Area'
    ))

    # 4. Scene Configuration (Matching your Env Visualizer)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', showbackground=True, backgroundcolor="rgb(240,240,240)"),
            yaxis=dict(title='Y', showbackground=True, backgroundcolor="rgb(240,240,240)"),
            zaxis=dict(title='Z', showbackground=True, backgroundcolor="rgb(240,240,240)"),
            aspectmode='data',
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.3))
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='white',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_name)), exist_ok=True)
    fig.write_html(output_name)
    print(f"Visualization with {len(trajectories)} trajectories saved to: {output_name}")


# 8192, 16384, 32768, 65536, 131072
MAX_TREE_SIZE = 400000
A = 128
batch_size = 32768
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the SST planner.')
    parser.add_argument('--motion', type=str, default='di', help='di, da, qc')
    args = parser.parse_args()

    # Define your environments
    environments = {
        "narrow": "envs/narrow.csv",
        "tree": "envs/tree.csv",
        "house": "envs/house.csv"
    }

    # Setup parameters based on motion type
    # (Assuming params is imported)
    motion_map = {
        'di': (params.sst_params_DI, params.sim_params_DI, params.Callables()),
        'da': (params.sst_params_DA, params.sim_params_DA, params.callables_DA),
        'qc': (params.sst_params_QC, params.sim_params_QC, params.callables_QC)
    }

    if args.motion not in motion_map:
        print("Invalid motion type")
        sys.exit()

    sst_base, sim_base, callables = motion_map[args.motion]
    sst_params = sst_base.replace(batch_size=batch_size)
    sim_params = sim_base.replace(batch_size=batch_size)

    # Update global references for the JIT closure
    SIM_PARAMS_RESERVED = sim_params
    CALLABLES_RESERVED = callables
    # Pre-compilation setup
    init = jnp.concatenate([jnp.asarray([sst_params.start.x, sst_params.start.y, sst_params.start.z]), 
                            jnp.zeros(sim_params.dims - 3, dtype=jnp.float32)], axis=0)
    controls = jnp.zeros(sim_params.action_dims)

    # ------------------------------------------------------------------
    # Execution Loop over Environments
    # ------------------------------------------------------------------
    for idx, (env_name, env_path) in enumerate(environments.items()):
        print(f"\n--- Processing Environment: {env_name} ({env_path}) ---")
        obstacles = helper.get_obs(env_path)
        all_trajectories = []

        # Run 10 iterations per environment
        for i in range(10):
            gc.collect()
            tree = rrtree.KinoTree.init(max_size=MAX_TREE_SIZE, state_dim=sim_params.dims, action_dim=sim_params.action_dims)
            tree = jax.device_put(tree)
            tree, _ = rrtree.add_nodes(tree, init, controls, -1, 0.0, 1)

            result = jit_while(tree, sst_params, sim_params, callables, obstacles, i)
            tree, key, goal_mask, goal, states, start_idx, iter_val, size = jax.block_until_ready(result)

            path_nodes, actions = extract_sol(tree, goal_mask, start_idx)
            
            if actions is not None:
                traj = rollout_full_trajectory(path_nodes[0], actions, sst_params, sim_params, callables)
                all_trajectories.append(traj)
                print(f"Run {i}: Path found.")
            else:
                print(f"Run {i}: No path.")

        # Save specific to the index
        if all_trajectories:
            output_file = f"visuals/solutiona{idx}.html"
            visualize_multi_trajectories(env_path, all_trajectories, sst_params, idx, output_file)
        else:
            print(f"No successful runs for {env_name}.")