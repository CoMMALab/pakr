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

# ------------------------------------------------------------------
# UNICYCLE HELPERS
# ------------------------------------------------------------------


def propagate_unicycle(states, actions, dt, constants):
    """
    Batched dynamics for a Unicycle.
    states:  (batch, 3) = [x, y, theta]
    actions: (batch, 2) = [v, omega]
    """
    x, y, theta = states[:, 0], states[:, 1], states[:, 2]
    v, omega = actions[:, 0], actions[:, 1]

    # Handle the case where omega is very small to avoid division by zero
    # using a simple Euler or RK4 integration
    new_theta = theta + omega * dt
    
    # Using midpoint integration for better accuracy with rotation
    mid_theta = theta + 0.5 * omega * dt
    new_x = x + v * jnp.cos(mid_theta) * dt
    new_y = y + v * jnp.sin(mid_theta) * dt
    
    return jnp.stack([new_x, new_y, new_theta], axis=-1)
@jax.jit
def reached_goal_unicycle(states, goal, radius):
    # state: [x, y, theta] -> check distance on [x, y]
    pos = states[:, 0:2]
    goal_pos = jnp.array([goal.x, goal.y])
    diff2 = jnp.sum((pos - goal_pos)**2, axis=-1)
    return diff2 < radius**2

@partial(jax.jit, static_argnums=(0,))
def dist_unicycle(sim_params, diff):
    # diff: (..., 3) -> [dx, dy, dtheta]
    dq = diff[..., 0:2]
    dtheta = diff[..., 2]

    # Normalize angular difference to [-pi, pi]
    dtheta_norm = jnp.atan2(jnp.sin(dtheta), jnp.cos(dtheta))

    pos_cost = jnp.sum(dq**2, axis=-1)
    ang_cost = dtheta_norm**2

    # Weighting heading vs position
    w_pos, w_ang = 1.0, 0.3
    return (w_pos * pos_cost + w_ang * ang_cost).T

@partial(jax.jit, static_argnums=(0,))
def sample_actions_unicycle(sim_params, key):
    # Action: [v, omega]
    B = sim_params.batch_size
    # Note: Using max_vel for v and max_accel for omega in constraints mapping
    tau = jax.random.uniform(
        key, (B, 2),
        minval=jnp.array([0.0, sim_params.motion_constraints.min_accel]), 
        maxval=jnp.array([sim_params.motion_constraints.max_vel, sim_params.motion_constraints.max_accel])
    )
    return tau

@partial(jax.jit, static_argnums=(0,))
def sample_unicycle(sim_params, key):
    # State: [x, y, theta]
    B = sim_params.batch_size
    keys = jax.random.split(key, 2)
    
    pos = jax.random.uniform(
        keys[0], (B, 2), 
        minval=jnp.array([sim_params.bounds.min_x, sim_params.bounds.min_y]), 
        maxval=jnp.array([sim_params.bounds.max_x, sim_params.bounds.max_y])
    )
    
    theta = jax.random.uniform(
        keys[1], (B, 1), minval=-jnp.pi, maxval=jnp.pi
    )

    return jnp.concatenate([pos, theta], axis=-1)

@partial(jax.jit, static_argnums=(1))
def valid_unicycle(state, params, obstacles):
    # state: [x, y, theta]
    x, y = state[:, 0], state[:, 1]
    within_bounds = (x >= params.bounds.min_x) & (x <= params.bounds.max_x) & \
                    (y >= params.bounds.min_y) & (y <= params.bounds.max_y)
    
    collision_free = helper.collision_check_2d(state[:, :2], obstacles)
    return within_bounds & collision_free

# Logic for NN Tiers and rrt_iteration remains identical to your template, 
# but uses propagate.rollout_final_2d internally.

# Global references to allow the switch branches to access static objects via closure
SIM_PARAMS_RESERVED = None
CALLABLES_RESERVED = None

def nn_tier_factory(size):
    def nn_fn(operands):
        states, tree_size, query = operands
        sliced_states = states[:size]
        
        parents, _ = helper.nearest_neighbor_masked(
            SIM_PARAMS_RESERVED, 
            CALLABLES_RESERVED.dist_fn, 
            sliced_states, 
            tree_size, 
            query
        )
        return parents
    return nn_fn

TIERS = [512, 1024, 4096, 16384, 32768, 64536, 100000]
NN_BRANCHES = [nn_tier_factory(t) for t in TIERS]

@partial(jax.jit, static_argnums=(3, 4, 5))
def rrt_iteration(tree, rng_key, obstacles, sst_params, sim_params, callables):
    B = sim_params.batch_size
    A = 32
    K = B // A 
    rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)
    seed_pts = callables.sample_fn(sim_params, subkey1)[:K]

    branch_idx = jnp.digitize(tree.tree_size, jnp.array(TIERS))
    branch_idx = jnp.minimum(branch_idx, len(TIERS) - 1)

    operands = (tree.states, tree.tree_size, seed_pts)
    parents_small = jax.lax.switch(branch_idx, NN_BRANCHES, operands) 
    start_states_small = tree.states[parents_small] 

    start_states = jnp.repeat(start_states_small, A, axis=0) 
    parents = jnp.repeat(parents_small, A, axis=0) 
    actions = callables.sampact_fn(sim_params, subkey2)

    # Use the 2D rollout for [x, y] distance accumulation
    states_end, valid_mask, dist_traveled = propagate.rollout_final_2d(
        start_states, actions, obstacles, sst_params, sim_params, callables
    )

    # ... [Rest of the node addition logic from your template] ...
    valid_mask = valid_mask.at[-1].set(False)
    states_end = states_end.at[-1].set(jnp.zeros(sim_params.dims))
    actions = actions.at[-1].set(jnp.zeros(sim_params.action_dims))
    parents = parents.at[-1].set(-1)
    dist_traveled = dist_traveled.at[-1].set(0.0)

    num_new = jnp.sum(valid_mask)
    valid_idx = jnp.nonzero(valid_mask, size=B, fill_value=-1)[0]

    new_states = states_end[valid_idx]
    new_actions = actions[valid_idx]
    new_parents = parents[valid_idx]
    new_costs = tree.costs[new_parents] + 1

    tree, start_idx = rrtree.add_nodes(tree, new_states, new_actions, new_parents, new_costs, num_new)
    goal_mask = callables.goal_fn(new_states, sst_params.goal, sst_params.goal_radius)
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
    
def pad_trajectories(trajectories, max_len):
    """
    trajectories: list of np.arrays of shape (len_i, 3)
    Returns: np.array of shape (10, max_len, 3)
    """
    num_trajs = len(trajectories)
    state_dim = trajectories[0].shape[-1]
    # Initialize with the last position for padding (as requested)
    padded = np.zeros((num_trajs, max_len, state_dim))
    
    for i, traj in enumerate(trajectories):
        traj_len = len(traj)
        padded[i, :traj_len] = traj
        # Pad remaining with the last state in the trajectory
        padded[i, traj_len:] = traj[-1]
        
    return padded

MAX_TREE_SIZE = 50000
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Unicycle Configuration
    # ------------------------------------------------------------------
    batch_size = 2048
    time_to_evolve = 40 

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
        dt = 0.025,
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
    
    # Load obstacles
    obstacles = helper.get_obs('envs/tree2d.csv')
    SIM_PARAMS_RESERVED = sim_params
    CALLABLES_RESERVED = callables
    # ------------------------------------------------------------------
    # 1. Compilation Warm-up
    # ------------------------------------------------------------------
    print("\nStarting Unicycle RRT - Pre-compiling kernels...")
    # Initialize a dummy tree with correct dims (3 for state, 2 for action)
    dummy_tree = rrtree.KinoTree.init(MAX_TREE_SIZE, sim_params.dims, sim_params.action_dims)
    _ = jit_while(dummy_tree, sst_params, sim_params, callables, obstacles, 0)
    print("Compilation complete.\n")

    # ------------------------------------------------------------------
    # 2. Execution Loop
    # ------------------------------------------------------------------
        # ------------------------------------------------------------------
    # Trajectory Replay and Recording
    # ------------------------------------------------------------------
    print("\nReplaying successful trajectories for storage...")
    os.makedirs('videos', exist_ok=True)

    # 1. Collect and Replay trajectories
    all_full_trajs = []
    for sol in all_solutions:
        # sol['path'] is (N, 3), sol['actions'] is (N, 2)
        path = sol['path']
        actions = sol['actions']
        
        # Start the replay at the root node
        current_state = path[0] 
        full_traj = [current_state]
        
        # Iterate through actions to integrate the state
        for i in range(1, len(path)):
            # Apply the action that led to the next node
            # Using a sub-stepping loop for higher resolution if needed
            # Here we do it step-by-step per node-transition
            action = actions[i].reshape(1, 2)
            state_in = current_state.reshape(1, 3)
            
            # Integrate using your provided propagate_unicycle
            # Note: dt must match your sim_params.dt
            new_state = propagate_unicycle(state_in, action, sim_params.dt, None)
            
            current_state = new_state.flatten()
            full_traj.append(current_state)
        
        all_full_trajs.append(np.array(full_traj))

    # 2. Pad and Save
    if all_full_trajs:
        # Determine max length for padding
        max_len = max(len(t) for t in all_full_trajs)
        
        # Create array (10, max_len, 3)
        # Using 10 as per your request; adjust if len(all_full_trajs) < 10
        num_trajs = min(len(all_full_trajs), 10)
        traj_data = np.zeros((num_trajs, max_len, 3))
        
        for i in range(num_trajs):
            traj = all_full_trajs[i]
            length = len(traj)
            traj_data[i, :length] = traj
            # Pad with the last reached position
            traj_data[i, length:] = traj[-1]
            
        np.save('videos/uni_traj.npy', traj_data)
        print(f"Successfully saved {num_trajs} trajectories to videos/uni_traj.npy")