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
# 2D DOUBLE INTEGRATOR HELPERS
# ------------------------------------------------------------------

@jax.jit
def reached_goal_DI2(states, goal, radius):
    # state: [x, y, vx, vy] -> check distance on [x, y]
    pos = states[:, 0:2]
    goal_pos = jnp.array([goal.x, goal.y])
    diff2 = jnp.sum((pos - goal_pos)**2, axis=-1)
    return diff2 < radius**2

@partial(jax.jit, static_argnums=(0,))
def dist_DI2(sim_params, diff):
    # diff: (..., 4) -> [dx, dy, dvx, dvy]
    dq = diff[..., 0:2]  # position
    dv = diff[..., 2:4]  # velocity

    pos_cost = jnp.sum(dq**2, axis=-1)
    vel_cost = jnp.sum(dv**2, axis=-1)

    w_pos, w_vel = 1.0, 0.1
    return (w_pos * pos_cost + w_vel * vel_cost).T

@partial(jax.jit, static_argnums=(0,))
def sample_actions_DI2(sim_params, key):
    # Action is 2D Acceleration [ax, ay]
    B = sim_params.batch_size
    tau = jax.random.uniform(
        key, (B, 2),
        minval=sim_params.motion_constraints.min_accel,
        maxval=sim_params.motion_constraints.max_accel
    )
    return tau

@partial(jax.jit, static_argnums=(0,))
def sample_DI2(sim_params, key):
    # State is 4D [x, y, vx, vy]
    B = sim_params.batch_size
    keys = jax.random.split(key, 2)
    
    pos = jax.random.uniform(
        keys[0], (B, 2), 
        minval=jnp.array([sim_params.bounds.min_x, sim_params.bounds.min_y]), 
        maxval=jnp.array([sim_params.bounds.max_x, sim_params.bounds.max_y])
    )
    
    vel = jax.random.uniform(
        keys[1], (B, 2), 
        minval=sim_params.motion_constraints.min_vel, 
        maxval=sim_params.motion_constraints.max_vel
    )

    return jnp.concatenate([pos, vel], axis=-1)

@partial(jax.jit, static_argnums=(1))
def valid_DI2(state, params, obstacles):
    # state: [x, y, vx, vy]
    x, y, vx, vy = state.T
    within_bounds = (x >= params.bounds.min_x) & (x <= params.bounds.max_x) & \
                    (y >= params.bounds.min_y) & (y <= params.bounds.max_y) & \
                    (vx >= params.motion_constraints.min_vel) & (vx <= params.motion_constraints.max_vel) & \
                    (vy >= params.motion_constraints.min_vel) & (vy <= params.motion_constraints.max_vel)
    
    # Collision check on (x, y)
    collision_free = helper.collision_check_2d(state[:, :2], obstacles)
    return within_bounds & collision_free


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

    seed_pts = callables.sample_fn(sim_params, subkey1)
    seed_pts = seed_pts[:K]

    branch_idx = jnp.digitize(tree.tree_size, jnp.array(TIERS))
    branch_idx = jnp.minimum(branch_idx, len(TIERS) - 1)

    operands = (tree.states, tree.tree_size, seed_pts)
    parents_small = jax.lax.switch(branch_idx, NN_BRANCHES, operands) 

    start_states_small = tree.states[parents_small] 

    start_states = jnp.repeat(start_states_small, A, axis=0) 
    parents = jnp.repeat(parents_small, A, axis=0) 

    actions = callables.sampact_fn(sim_params, subkey2)

    states_end, valid_mask, dist_traveled = propagate.rollout_final_2d(
        start_states, actions, obstacles, sst_params, sim_params, callables
    )

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
    new_costs = tree.costs[new_parents] + dist_traveled[valid_idx]

    tree, start_idx = rrtree.add_nodes(
        tree, new_states, new_actions, new_parents, new_costs, num_new
    )

    goal_mask = callables.goal_fn(
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
    

MAX_TREE_SIZE = 10000
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the SST planner.')
    parser.add_argument('--env', type=str, default='envs/tree2d.csv', help='Path to environment config.')
    parser.add_argument('--motion', type=str, default='di', help='di')
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 2D Double Integrator Configuration
    # ------------------------------------------------------------------
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
    
    SIM_PARAMS_RESERVED = sim_params
    CALLABLES_RESERVED = callables

    # ------------------------------------------------------------------
    # 2. Compilation Warm-up
    # ------------------------------------------------------------------
    print("\nStarting RRT - Pre-compiling kernels...")
    dummy_tree = rrtree.KinoTree.init(MAX_TREE_SIZE, sim_params.dims, sim_params.action_dims)
    _ = jit_while(dummy_tree, sst_params, sim_params, callables, obstacles, 0)
    print("Compilation complete.\n")

    # ------------------------------------------------------------------
    # 3. Execution Loop & Statistics
    # ------------------------------------------------------------------
    times, iters, sizes, costs = [], [], [], []
    all_solutions = []
    for i in range(100):
        gc.collect()

        # Initialize tree with start state [x, y, vx, vy]
        tree = rrtree.KinoTree.init(max_size=MAX_TREE_SIZE, state_dim=sim_params.dims, action_dim=sim_params.action_dims)
        tree = jax.device_put(tree)
        
        # Start at 0 velocity
        init = jnp.array([sst_params.start.x, sst_params.start.y, 0.0, 0.0], dtype=jnp.float32)
        controls = jnp.zeros(sim_params.action_dims)
        tree, _ = rrtree.add_nodes(tree, init, controls, -1, 0.0, 1)

        start_p = time.perf_counter()

        # Solve
        result = jit_while(tree, sst_params, sim_params, callables, obstacles, i)
        tree, key, goal_mask, goal, states, start_idx, iter_val, size = jax.block_until_ready(result)
        timer = time.perf_counter() - start_p
        

        path, actions = extract_sol(tree, goal_mask, start_idx)
        #print(path)
        #print(actions)
        is_valid = verify_sol(path, actions, obstacles, sst_params, sim_params, callables)
        if not is_valid:
            print("Invalid solution found!")
            print("Path:", path)
            print("Actions:", actions)

        all_solutions.append({
            'path': np.array(path),       # The sparse nodes in the tree
            'actions': np.array(actions), # The constant actions applied between nodes
            'start': np.array([sst_params.start.x, sst_params.start.y, 0.0]),
            'goal': np.array([sst_params.goal.x, sst_params.goal.y])
        })
        # Calculate cost for stats
        cost = tree.costs[jnp.argmax(goal_mask) + start_idx]
        
        costs.append(cost)
        times.append(timer)
        iters.append(iter_val)
        sizes.append(size)
        
        print(f"Found goal after {iter_val} iterations, time: {(timer)*1e3:.3f} ms, tree size: {size}")


    np.savez('benchmarks/di2d_results.npz', solutions=all_solutions)
    # Final Statistics Logic
    times = jnp.array(times)
    iters = jnp.array(iters)
    sizes = jnp.array(sizes)
    costs = jnp.array(costs)

    print(f"Average time over 100 runs: {jnp.mean(times)*1e3:.3f} ms, {jnp.mean(iters):.2f} iterations, size {jnp.mean(sizes):.2f}")
    print(f"min time over 100 runs: {jnp.min(times)*1e3:.3f} ms, {jnp.min(iters)} iterations, min size {jnp.min(sizes)}")
    print(f"max time over 100 runs: {jnp.max(times)*1e3:.3f} ms, {jnp.max(iters)} iterations, max size {jnp.max(sizes)}")
    print(f"Average cost over 100 runs: {jnp.mean(costs):.3f}, min cost: {jnp.min(costs):.3f}, max cost: {jnp.max(costs):.3f}")