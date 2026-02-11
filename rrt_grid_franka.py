from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
from dataclasses import dataclass
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


@dataclass
class HierarchicalGrid:
    """
    Hierarchical spatial grid for 27-DOF manipulation:
    - State structure: [q1..q7 (arm joints), v1..v7 (arm velocities), 
                        obj_xyz, obj_quat(4), obj_vel(3), obj_angvel(3)]
    
    Uses two grids:
    1. Object position grid (3D, fine resolution) - primary filter
    2. Arm configuration grid (7D, coarse resolution) - secondary filter
    
    Plus overflow buffer and recent nodes fallback.
    """
    # Object position grid (3D, fine)
    obj_bins_per_dim: int = 10  # 10^3 = 1000 cells
    obj_num_cells: int = 1000
    obj_max_nodes_per_cell: int = 150
    obj_grid_min: jnp.ndarray  # [3]
    obj_grid_max: jnp.ndarray  # [3]
    obj_cell_contents: jnp.ndarray  # [obj_num_cells, max_nodes_per_cell]
    obj_cell_counts: jnp.ndarray    # [obj_num_cells]
    
    # Arm configuration grid (7D, coarse - 3 bins/dim = 3^7 = 2187 cells)
    arm_bins_per_dim: int = 3
    arm_num_cells: int = 2187
    arm_max_nodes_per_cell: int = 100
    arm_grid_min: jnp.ndarray  # [7] - joint limits min
    arm_grid_max: jnp.ndarray  # [7] - joint limits max
    arm_cell_contents: jnp.ndarray  # [arm_num_cells, max_nodes_per_cell]
    arm_cell_counts: jnp.ndarray    # [arm_num_cells]
    
    # Shared metadata
    node_to_obj_cell: jnp.ndarray   # [max_tree_size]
    node_to_arm_cell: jnp.ndarray   # [max_tree_size]
    
    # Overflow handling
    max_overflow: int = 5000
    overflow_nodes: jnp.ndarray  # [max_overflow]
    overflow_count: jnp.ndarray  # scalar
    
    @staticmethod
    def init(max_tree_size, obj_bounds=None, arm_bounds=None):
        """
        Initialize hierarchical grid for 27-DOF manipulation.
        
        Args:
            max_tree_size: Maximum number of nodes
            obj_bounds: [[x_min, y_min, z_min], [x_max, y_max, z_max]] for object
            arm_bounds: [[q1_min, ..., q7_min], [q1_max, ..., q7_max]] for arm
        """
        # Object grid defaults
        if obj_bounds is None:
            obj_grid_min = jnp.array([-2.0, -2.0, -2.0])  # Typical tabletop workspace
            obj_grid_max = jnp.array([2.0, 2.0, 2.0])
        else:
            obj_grid_min = jnp.array(obj_bounds[0])
            obj_grid_max = jnp.array(obj_bounds[1])
        
        # Arm grid defaults (typical 7-DOF arm has ~[-π, π] joint limits)
        if arm_bounds is None:
            arm_grid_min = jnp.array([-jnp.pi, -jnp.pi, -jnp.pi, -jnp.pi, -jnp.pi, -jnp.pi, -jnp.pi])
            arm_grid_max = jnp.array([jnp.pi, jnp.pi, jnp.pi, jnp.pi, jnp.pi, jnp.pi, jnp.pi])
        else:
            arm_grid_min = jnp.array(arm_bounds[0])
            arm_grid_max = jnp.array(arm_bounds[1])
        
        obj_bins = 10
        obj_num_cells = obj_bins ** 3
        obj_max_nodes = max(150, max_tree_size // obj_num_cells * 3)
        
        arm_bins = 3
        arm_num_cells = arm_bins ** 7
        arm_max_nodes = max(100, max_tree_size // arm_num_cells * 2)
        
        max_overflow = max(5000, max_tree_size // 10)
        
        return HierarchicalGrid(
            # Object grid
            obj_bins_per_dim=obj_bins,
            obj_num_cells=obj_num_cells,
            obj_max_nodes_per_cell=obj_max_nodes,
            obj_grid_min=obj_grid_min,
            obj_grid_max=obj_grid_max,
            obj_cell_contents=jnp.full((obj_num_cells, obj_max_nodes), -1, dtype=jnp.int32),
            obj_cell_counts=jnp.zeros(obj_num_cells, dtype=jnp.int32),
            
            # Arm grid
            arm_bins_per_dim=arm_bins,
            arm_num_cells=arm_num_cells,
            arm_max_nodes_per_cell=arm_max_nodes,
            arm_grid_min=arm_grid_min,
            arm_grid_max=arm_grid_max,
            arm_cell_contents=jnp.full((arm_num_cells, arm_max_nodes), -1, dtype=jnp.int32),
            arm_cell_counts=jnp.zeros(arm_num_cells, dtype=jnp.int32),
            
            # Shared
            node_to_obj_cell=jnp.full(max_tree_size, -1, dtype=jnp.int32),
            node_to_arm_cell=jnp.full(max_tree_size, -1, dtype=jnp.int32),
            
            # Overflow
            max_overflow=max_overflow,
            overflow_nodes=jnp.full(max_overflow, -1, dtype=jnp.int32),
            overflow_count=jnp.array(0, dtype=jnp.int32)
        )


def compute_obj_cell_idx(state, grid_min, grid_max, bins_per_dim):
    """
    Compute object position cell index.
    State structure: [q(7), v(7), obj_xyz(3), obj_quat(4), obj_vel(3), obj_angvel(3)]
    Object xyz is at indices 14:17
    """
    obj_pos = state[14:17]
    
    # Clamp to bounds
    obj_pos = jnp.clip(obj_pos, grid_min, grid_max)
    
    # Normalize to [0, bins_per_dim)
    normalized = (obj_pos - grid_min) / (grid_max - grid_min) * bins_per_dim
    bin_coords = jnp.floor(normalized).astype(jnp.int32)
    bin_coords = jnp.clip(bin_coords, 0, bins_per_dim - 1)
    
    # Convert 3D to 1D
    cell_idx = bin_coords[0] * bins_per_dim**2 + bin_coords[1] * bins_per_dim + bin_coords[2]
    
    return cell_idx


def compute_arm_cell_idx(state, grid_min, grid_max, bins_per_dim):
    """
    Compute arm configuration cell index.
    Uses joint positions q1..q7 (first 7 dimensions).
    """
    q = state[:7]
    
    # Clamp to joint limits
    q = jnp.clip(q, grid_min, grid_max)
    
    # Normalize to [0, bins_per_dim)
    normalized = (q - grid_min) / (grid_max - grid_min) * bins_per_dim
    bin_coords = jnp.floor(normalized).astype(jnp.int32)
    bin_coords = jnp.clip(bin_coords, 0, bins_per_dim - 1)
    
    # Convert 7D to 1D using base-bins_per_dim encoding
    cell_idx = jnp.array(0, dtype=jnp.int32)
    for i in range(7):
        cell_idx = cell_idx * bins_per_dim + bin_coords[i]
    
    return cell_idx


def get_neighbor_cells_3d(cell_idx, bins_per_dim):
    """
    Get 27 neighboring cells for 3D grid.
    """
    z = cell_idx % bins_per_dim
    y = (cell_idx // bins_per_dim) % bins_per_dim
    x = cell_idx // (bins_per_dim ** 2)
    
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                nx, ny, nz = x + dx, y + dy, z + dz
                
                if 0 <= nx < bins_per_dim and 0 <= ny < bins_per_dim and 0 <= nz < bins_per_dim:
                    neighbor_idx = nx * bins_per_dim**2 + ny * bins_per_dim + nz
                    neighbors.append(neighbor_idx)
                else:
                    neighbors.append(-1)
    
    return jnp.array(neighbors, dtype=jnp.int32)


def get_neighbor_cells_7d(cell_idx, bins_per_dim):
    """
    Get neighboring cells for 7D grid.
    In 7D with 3 bins/dim, we have 3^7 = 2187 total cells.
    Neighbors: all cells within Hamming distance 1 (adjacent in any dimension).
    That's 1 + 2*7 = 15 cells (self + 2 neighbors per dim).
    
    For efficiency, we only check immediate neighbors in each dimension.
    """
    # Decode 7D coordinates from 1D index
    coords = jnp.zeros(7, dtype=jnp.int32)
    temp_idx = cell_idx
    for i in range(6, -1, -1):
        coords[i] = temp_idx % bins_per_dim
        temp_idx = temp_idx // bins_per_dim
    
    neighbors = [cell_idx]  # Include self
    
    # Add neighbors in each dimension
    for dim in range(7):
        for delta in [-1, 1]:
            new_coord = coords[dim] + delta
            if 0 <= new_coord < bins_per_dim:
                # Create new coordinates
                new_coords = coords.at[dim].set(new_coord)
                
                # Encode back to 1D
                new_idx = 0
                for i in range(7):
                    new_idx = new_idx * bins_per_dim + new_coords[i]
                
                neighbors.append(int(new_idx))
    
    # Pad to fixed size (15 neighbors max)
    while len(neighbors) < 15:
        neighbors.append(-1)
    
    return jnp.array(neighbors[:15], dtype=jnp.int32)


def add_node_to_hierarchical_grid(grid, node_idx, state):
    """
    Add a node to both object and arm grids.
    """
    # Compute cell indices
    obj_cell = compute_obj_cell_idx(state, grid.obj_grid_min, grid.obj_grid_max, grid.obj_bins_per_dim)
    arm_cell = compute_arm_cell_idx(state, grid.arm_grid_min, grid.arm_grid_max, grid.arm_bins_per_dim)
    
    # Try to add to object grid
    obj_count = grid.obj_cell_counts[obj_cell]
    obj_has_space = obj_count < grid.obj_max_nodes_per_cell
    
    # Try to add to arm grid
    arm_count = grid.arm_cell_counts[arm_cell]
    arm_has_space = arm_count < grid.arm_max_nodes_per_cell
    
    # If both have space, add to both
    # If either is full, add to overflow
    both_have_space = obj_has_space & arm_has_space
    
    def add_to_grids():
        # Add to object grid
        new_obj_contents = grid.obj_cell_contents.at[obj_cell, obj_count].set(node_idx)
        new_obj_counts = grid.obj_cell_counts.at[obj_cell].add(1)
        
        # Add to arm grid
        new_arm_contents = grid.arm_cell_contents.at[arm_cell, arm_count].set(node_idx)
        new_arm_counts = grid.arm_cell_counts.at[arm_cell].add(1)
        
        # Update node mappings
        new_node_to_obj = grid.node_to_obj_cell.at[node_idx].set(obj_cell)
        new_node_to_arm = grid.node_to_arm_cell.at[node_idx].set(arm_cell)
        
        return grid._replace(
            obj_cell_contents=new_obj_contents,
            obj_cell_counts=new_obj_counts,
            arm_cell_contents=new_arm_contents,
            arm_cell_counts=new_arm_counts,
            node_to_obj_cell=new_node_to_obj,
            node_to_arm_cell=new_node_to_arm
        )
    
    def add_to_overflow():
        overflow_pos = grid.overflow_count
        new_overflow_nodes = grid.overflow_nodes.at[overflow_pos].set(node_idx)
        new_overflow_count = grid.overflow_count + 1
        new_node_to_obj = grid.node_to_obj_cell.at[node_idx].set(-2)  # -2 = overflow
        new_node_to_arm = grid.node_to_arm_cell.at[node_idx].set(-2)
        return grid._replace(
            overflow_nodes=new_overflow_nodes,
            overflow_count=new_overflow_count,
            node_to_obj_cell=new_node_to_obj,
            node_to_arm_cell=new_node_to_arm
        )
    
    grid = lax.cond(both_have_space, add_to_grids, add_to_overflow)
    
    return grid


def add_nodes_to_grid_batched(grid, node_indices, states, num_new):
    """
    Add multiple nodes to the hierarchical grid.
    """
    def add_single(i, grid):
        node_idx = node_indices[i]
        state = states[i]
        
        grid = lax.cond(
            i < num_new,
            lambda g: add_node_to_hierarchical_grid(g, node_idx, state),
            lambda g: g,
            grid
        )
        return grid
    
    grid = lax.fori_loop(0, node_indices.shape[0], add_single, grid)
    
    return grid


def get_candidate_mask_hierarchical(grid, query_state, tree_size, max_tree_size, recent_window=1000):
    """
    Get candidate mask using both object and arm grids.
    
    Returns union of:
    - Object grid neighbors (27 cells)
    - Arm grid neighbors (15 cells)
    - Overflow nodes
    - Recent nodes (last 1000)
    """
    candidate_mask = jnp.zeros(max_tree_size, dtype=bool)
    
    # === Object grid neighbors ===
    obj_cell = compute_obj_cell_idx(query_state, grid.obj_grid_min, grid.obj_grid_max, grid.obj_bins_per_dim)
    obj_neighbors = get_neighbor_cells_3d(obj_cell, grid.obj_bins_per_dim)
    
    def add_obj_cell_nodes(i, mask):
        neighbor_idx = obj_neighbors[i]
        
        def process_cell(m):
            count = grid.obj_cell_counts[neighbor_idx]
            indices = grid.obj_cell_contents[neighbor_idx, :grid.obj_max_nodes_per_cell]
            
            valid_indices = jnp.where(
                jnp.arange(grid.obj_max_nodes_per_cell) < count,
                indices,
                -1
            )
            
            def set_mask(j, m):
                idx = valid_indices[j]
                m = jnp.where(
                    (idx >= 0) & (idx < max_tree_size),
                    m.at[idx].set(True),
                    m
                )
                return m
            
            m = lax.fori_loop(0, grid.obj_max_nodes_per_cell, set_mask, m)
            return m
        
        mask = lax.cond(neighbor_idx >= 0, process_cell, lambda m: m, mask)
        return mask
    
    candidate_mask = lax.fori_loop(0, 27, add_obj_cell_nodes, candidate_mask)
    
    # === Arm grid neighbors ===
    arm_cell = compute_arm_cell_idx(query_state, grid.arm_grid_min, grid.arm_grid_max, grid.arm_bins_per_dim)
    arm_neighbors = get_neighbor_cells_7d(arm_cell, grid.arm_bins_per_dim)
    
    def add_arm_cell_nodes(i, mask):
        neighbor_idx = arm_neighbors[i]
        
        def process_cell(m):
            count = grid.arm_cell_counts[neighbor_idx]
            indices = grid.arm_cell_contents[neighbor_idx, :grid.arm_max_nodes_per_cell]
            
            valid_indices = jnp.where(
                jnp.arange(grid.arm_max_nodes_per_cell) < count,
                indices,
                -1
            )
            
            def set_mask(j, m):
                idx = valid_indices[j]
                m = jnp.where(
                    (idx >= 0) & (idx < max_tree_size),
                    m.at[idx].set(True),
                    m
                )
                return m
            
            m = lax.fori_loop(0, grid.arm_max_nodes_per_cell, set_mask, m)
            return m
        
        mask = lax.cond(neighbor_idx >= 0, process_cell, lambda m: m, mask)
        return mask
    
    candidate_mask = lax.fori_loop(0, 15, add_arm_cell_nodes, candidate_mask)
    
    # === Overflow nodes ===
    def add_overflow(i, mask):
        idx = grid.overflow_nodes[i]
        mask = jnp.where(
            (i < grid.overflow_count) & (idx >= 0) & (idx < max_tree_size),
            mask.at[idx].set(True),
            mask
        )
        return mask
    
    candidate_mask = lax.fori_loop(0, grid.max_overflow, add_overflow, candidate_mask)
    
    # === Recent nodes ===
    recent_start = jnp.maximum(0, tree_size - recent_window)
    candidate_mask = candidate_mask.at[recent_start:tree_size].set(True)
    
    return candidate_mask


def nearest_neighbor_hierarchical(grid, tree_states, tree_size, query_states, dist_fn, sim_params):
    """
    Find nearest neighbors using hierarchical grid.
    """
    batch_size = query_states.shape[0]
    
    def find_nn_single(i):
        query = query_states[i]
        
        # Get candidate mask from hierarchical grid
        candidate_mask = get_candidate_mask_hierarchical(
            grid, query, tree_size, tree_states.shape[0], recent_window=1000
        )
        
        # Compute distances only for candidates
        dists = jax.vmap(lambda s, m: jnp.where(m, dist_fn(query, s), jnp.inf))(
            tree_states, candidate_mask
        )
        
        nn_idx = jnp.argmin(dists)
        
        return nn_idx
    
    nn_indices = jax.vmap(find_nn_single)(jnp.arange(batch_size))
    
    return nn_indices


def update_valid_mask_for_overflow(grid, valid_mask, new_states, tree_size):
    """
    Check if new states would cause overflow and update valid mask.
    """
    batch_size = new_states.shape[0]
    
    def check_overflow(i):
        state = new_states[i]
        
        obj_cell = compute_obj_cell_idx(state, grid.obj_grid_min, grid.obj_grid_max, grid.obj_bins_per_dim)
        arm_cell = compute_arm_cell_idx(state, grid.arm_grid_min, grid.arm_grid_max, grid.arm_bins_per_dim)
        
        obj_overflowed = grid.obj_cell_counts[obj_cell] >= grid.obj_max_nodes_per_cell
        arm_overflowed = grid.arm_cell_counts[arm_cell] >= grid.arm_max_nodes_per_cell
        
        # Reject if overflow buffer is nearly full
        overflow_full = grid.overflow_count >= (grid.max_overflow * 0.9)
        should_reject = (obj_overflowed | arm_overflowed) & overflow_full
        
        return should_reject
    
    reject_mask = jax.vmap(check_overflow)(jnp.arange(batch_size))
    updated_valid_mask = valid_mask & ~reject_mask
    
    return updated_valid_mask


@partial(jax.jit, static_argnums=(3, 4, 5))
def rrt_iteration(tree, grid, rng_key, obstacles, sst_params, sim_params, callables):
    """
    One batched kinodynamic RRT iteration with hierarchical grid acceleration.
    """
    rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)

    # Sample and nearest neighbor
    seed_pts = callables.sample_fn(sim_params, subkey1)
    
    parents = nearest_neighbor_hierarchical(
        grid, tree.states, tree.tree_size, seed_pts, callables.dist_fn, sim_params
    )
    
    start_states = tree.states[parents]

    # Sample actions and rollout
    actions = callables.sampact_fn(sim_params, subkey2)

    states_end, valid_mask, dist_traveled = propagate.rollout_final(
        start_states, actions, obstacles, sst_params, sim_params, callables
    )

    # Check overflow
    valid_mask = update_valid_mask_for_overflow(grid, valid_mask, states_end, tree.tree_size)

    # Padding
    valid_mask = valid_mask.at[-1].set(False)
    states_end = states_end.at[-1].set(jnp.zeros(sim_params.dims))
    actions = actions.at[-1].set(jnp.zeros(sim_params.action_dims))
    parents = parents.at[-1].set(-1)
    dist_traveled = dist_traveled.at[-1].set(0.0)

    # Filter valid
    num_new = jnp.sum(valid_mask)

    valid_idx = jnp.nonzero(
        valid_mask,
        size=sim_params.batch_size,
        fill_value=-1
    )[0]

    new_states = states_end[valid_idx]
    new_actions = actions[valid_idx]
    new_parents = parents[valid_idx]
    new_costs = tree.costs[new_parents] + dist_traveled[valid_idx]

    # Insert into tree
    tree, start_idx = rrtree.add_nodes(
        tree,
        new_states,
        new_actions,
        new_parents,
        new_costs,
        num_new
    )

    # Insert into grid
    node_indices = jnp.arange(start_idx, start_idx + sim_params.batch_size, dtype=jnp.int32)
    grid = add_nodes_to_grid_batched(grid, node_indices, new_states, num_new)

    # Goal check
    goal_mask = helper.reached_goal(
        new_states, sst_params.goal, sst_params.goal_radius
    )

    return tree, grid, rng_key, goal_mask, jnp.sum(goal_mask), new_states, start_idx


@partial(jax.jit, static_argnums=(1,2,3))
def jit_while(tree, grid, sst_params, sim_params, callables, obstacles, i):
    def body_fn(carry):
        tree, grid, key, goal_mask, goal, states, start_idx, iter = carry

        key, subkey = jax.random.split(key)

        tree, grid, subkey, goal_mask, goal, states, start_idx = rrt_iteration(
            tree, grid, subkey, obstacles, sst_params, sim_params, callables
        )

        return (tree, grid, key, goal_mask, goal, states, start_idx, iter + 1)

    def cond_fn(carry):
        tree, grid, key, goal_mask, goal, states, start_idx, iter = carry
        return goal == 0
    
    init_carry = (tree,
                  grid,
                  jax.random.PRNGKey(i),
                  jnp.zeros(sim_params.batch_size, dtype=bool),
                  jnp.array(0, dtype=jnp.int32),
                  jnp.zeros([sim_params.batch_size, sim_params.dims], dtype=jnp.float32),
                  jnp.array(0, dtype=jnp.int32),
                  jnp.array(0, dtype=jnp.int32))

    tree, grid, key, goal_mask, goal, states, start_idx, iter = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    return tree, grid, key, goal_mask, goal, states, start_idx, iter, tree.tree_size


MAX_TREE_SIZE = 70000


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SST planner with hierarchical grid for 27-DOF manipulation.')
    parser.add_argument('--env', type=str, default='envs/tree.csv', help='Path to the environment config file.')
    parser.add_argument('--motion', type=str, default='arm', help='Motion type (should be arm/manipulation system)')
    parser.add_argument('--obj_bins', type=int, default=10, help='Bins per dim for object position grid')
    parser.add_argument('--arm_bins', type=int, default=3, help='Bins per dim for arm config grid')
    parser.add_argument('--obj_bounds', type=str, default=None, help='Object bounds: "xmin,ymin,zmin,xmax,ymax,zmax"')
    parser.add_argument('--arm_bounds', type=str, default=None, help='Arm joint limits: "q1min,...,q7min,q1max,...,q7max"')
    args = parser.parse_args()

    # Load parameters for your 27-DOF system
    # You'll need to define these in params.py
    match args.motion:
        case 'arm' | 'manipulation':
            sst_params = params.sst_params_ARM27  # You need to define this
            sim_params = params.sim_params_ARM27  # You need to define this
            callables = params.callables_ARM27    # You need to define this
        case _:
            print(f"Motion type '{args.motion}' not supported for hierarchical grid.")
            print("This version is specifically for 27-DOF arm manipulation.")
            exit()

    # Parse bounds
    obj_bounds = None
    if args.obj_bounds:
        bounds = [float(x) for x in args.obj_bounds.split(',')]
        obj_bounds = [bounds[:3], bounds[3:]]
    
    arm_bounds = None
    if args.arm_bounds:
        bounds = [float(x) for x in args.arm_bounds.split(',')]
        arm_bounds = [bounds[:7], bounds[7:]]

    obstacles = helper.get_obs(args.env)
    
    # Initialize tree
    tree = rrtree.KinoTree.init(max_size=MAX_TREE_SIZE, state_dim=sim_params.dims, action_dim=sim_params.action_dims)
    tree = jax.device_put(tree)
    
    # Initialize hierarchical grid
    grid = HierarchicalGrid.init(
        max_tree_size=MAX_TREE_SIZE,
        obj_bounds=obj_bounds,
        arm_bounds=arm_bounds
    )
    grid = jax.device_put(grid)
    
    # Add root node
    # State structure: [q(7), v(7), obj_xyz(3), obj_quat(4), obj_vel(3), obj_angvel(3)] = 27D
    init = jnp.concatenate([
        jnp.asarray([sst_params.start.x, sst_params.start.y, sst_params.start.z]),  # Initial joint angles (first 3)
        jnp.zeros(sim_params.dims - 3, dtype=jnp.float32)  # Rest of state
    ], axis=0)
    controls = jnp.zeros(sim_params.action_dims)
    tree, _ = rrtree.add_nodes(tree, init, controls, -1, 0.0, 1)
    grid = add_node_to_hierarchical_grid(grid, 0, init)

    print("\n\n=== Hierarchical Grid RRT for 27-DOF Manipulation ===")
    print(f"Object grid: {args.obj_bins}^3 = {args.obj_bins**3} cells, {grid.obj_max_nodes_per_cell} nodes/cell")
    print(f"Arm grid: {args.arm_bins}^7 = {args.arm_bins**7} cells, {grid.arm_max_nodes_per_cell} nodes/cell")
    print(f"Overflow buffer: {grid.max_overflow} nodes")
    print(f"State dims: {sim_params.dims} (7 joints + 7 vels + 3 obj_pos + 4 obj_quat + 3 obj_vel + 3 obj_angvel)\n")

    # Warm-up
    print("Warming up JIT compilation...")
    jit_while(tree, grid, sst_params, sim_params, callables, obstacles, 0)
    print("Compilation complete!\n")

    # Benchmark
    times = []
    iters = []
    sizes = []
    costs = []
    obj_overflows = []
    arm_overflows = []
    total_overflows = []
    
    for i in range(100):
        gc.collect()

        # Reset
        tree = rrtree.KinoTree.init(max_size=MAX_TREE_SIZE, state_dim=sim_params.dims, action_dim=sim_params.action_dims)
        tree = jax.device_put(tree)
        
        grid = HierarchicalGrid.init(
            max_tree_size=MAX_TREE_SIZE,
            obj_bounds=obj_bounds,
            arm_bounds=arm_bounds
        )
        grid = jax.device_put(grid)
        
        # Add root
        init = jnp.concatenate([
            jnp.asarray([sst_params.start.x, sst_params.start.y, sst_params.start.z]),
            jnp.zeros(sim_params.dims - 3, dtype=jnp.float32)
        ], axis=0)
        controls = jnp.zeros(sim_params.action_dims)
        tree, _ = rrtree.add_nodes(tree, init, controls, -1, 0.0, 1)
        grid = add_node_to_hierarchical_grid(grid, 0, init)

        start_p = time.perf_counter()

        # Run
        tree, grid, key, goal_mask, goal, states, start_idx, iter, size = jit_while(
            tree, grid, sst_params, sim_params, callables, obstacles, i
        )
        
        timer = time.perf_counter() - start_p
        cost = tree.costs[jnp.argmax(goal_mask) + start_idx]
        
        costs.append(cost)
        times.append(timer)
        iters.append(iter)
        sizes.append(size)
        
        # Count overflows
        obj_overflow = jnp.sum(grid.obj_cell_counts >= grid.obj_max_nodes_per_cell)
        arm_overflow = jnp.sum(grid.arm_cell_counts >= grid.arm_max_nodes_per_cell)
        total_overflow = grid.overflow_count
        
        obj_overflows.append(obj_overflow)
        arm_overflows.append(arm_overflow)
        total_overflows.append(total_overflow)
        
        print(f"Run {i+1:3d}: {iter:4d} iters, {timer*1e3:7.3f} ms, size: {size:5d}, cost: {cost:.3f} | "
              f"Overflow: obj={obj_overflow:3d} arm={arm_overflow:3d} total={total_overflow:4d}")

    times = jnp.array(times)
    iters = jnp.array(iters)
    sizes = jnp.array(sizes)
    costs = jnp.array(costs)
    
    print("\n=== Statistics over 100 runs ===")
    print(f"Time    - Avg: {jnp.mean(times)*1e3:7.3f} ms, Min: {jnp.min(times)*1e3:7.3f} ms, Max: {jnp.max(times)*1e3:7.3f} ms")
    print(f"Iters   - Avg: {jnp.mean(iters):6.2f}, Min: {jnp.min(iters):6d}, Max: {jnp.max(iters):6d}")
    print(f"Size    - Avg: {jnp.mean(sizes):7.2f}, Min: {jnp.min(sizes):7d}, Max: {jnp.max(sizes):7d}")
    print(f"Cost    - Avg: {jnp.mean(costs):6.3f}, Min: {jnp.min(costs):6.3f}, Max: {jnp.max(costs):6.3f}")
    print(f"Obj OF  - Avg: {jnp.mean(jnp.array(obj_overflows)):6.2f}")
    print(f"Arm OF  - Avg: {jnp.mean(jnp.array(arm_overflows)):6.2f}")
    print(f"Tot OF  - Avg: {jnp.mean(jnp.array(total_overflows)):6.2f}")
    
    # Extract solution
    idx = jnp.argmax(goal_mask)
    controls, states = helper.find_solution_path_rrt(tree, states, goal_mask, start_idx)
    init = jnp.concatenate([
        jnp.asarray([sst_params.start.x, sst_params.start.y, sst_params.start.z]),
        jnp.zeros(sim_params.dims - 3, dtype=jnp.float32)
    ], axis=0)
    
    print("\nSolution found!")
    print("Controls shape:", controls.shape)
    print("States shape:", states.shape)
    
    waypoints, states = helper.recreate_trajectory(init, controls, sim_params, callables.prop_fn)
    jnp.save('cache/waypoints_27dof.npy', waypoints)
    print("\nWaypoints saved to cache/waypoints_27dof.npy")