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

    rng_key, subkey1, subkey2, subkey3 = jax.random.split(rng_key, 4)

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
    timesteps = propagate.sample_duration(sst_params.time_to_evolve, sim_params.batch_size, subkey3)  # (B,)

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

# 8192, 16384, 32768, 65536, 131072
MAX_TREE_SIZE = 1_000_000
A = 128
batch_size = 32768  # Must be a multiple of A for the tiered NN to work correctly
dt = 0.2
tte = 10
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the SST planner.')
    parser.add_argument('--env', type=str, default='envs/tree.csv', help='Path to environment config.')
    parser.add_argument('--motion', type=str, default='di', help='di, da, qc')
    args = parser.parse_args()

    match args.motion:
        case 'di':
            sst_params, sim_params, callables = params.sst_params_DI, params.sim_params_DI, params.Callables()
        case 'da':
            sst_params, sim_params, callables = params.sst_params_DA, params.sim_params_DA, params.callables_DA
        case 'qc':
            sst_params, sim_params, callables = params.sst_params_QC, params.sim_params_QC, params.callables_QC
        case _:
            print("invalid motion type")
            sys.exit()


    sst_params = sst_params.replace(batch_size=batch_size)
    sim_params = sim_params.replace(batch_size=batch_size)
    obstacles = helper.get_obs(args.env)
    sim_params = sim_params.replace(dt=dt)
    sst_params = sst_params.replace(time_to_evolve=tte)

    # Update global references for the JIT closure
    SIM_PARAMS_RESERVED = sim_params
    CALLABLES_RESERVED = callables

    # ------------------------------------------------------------------
    # 2. Compilation Warm-up
    # ------------------------------------------------------------------
    print("\nStarting RRT - Pre-compiling kernels...")
    dummy_tree = rrtree.KinoTree.init(MAX_TREE_SIZE, sim_params.dims, sim_params.action_dims)
    # This triggers the JIT for the while loop and all switch branches
    init = jnp.concatenate([jnp.asarray([sst_params.start.x, sst_params.start.y, sst_params.start.z]), jnp.zeros(sim_params.dims - 3, dtype=jnp.float32)], axis=0)
    controls = jnp.zeros(sim_params.action_dims)
    dummy_tree, _ = rrtree.add_nodes(dummy_tree, init, controls, -1, 0.0, 1)
    _ = jit_while(dummy_tree, sst_params, sim_params, callables, obstacles, 0)
    print("Compilation complete.\n")

    # ------------------------------------------------------------------
    # 3. Execution Loop & Statistics
    # ------------------------------------------------------------------
    times, iters, sizes, costs = [], [], [], []

    for i in range(100):
        gc.collect()

        # Initialize tree with start state
        tree = rrtree.KinoTree.init(max_size=MAX_TREE_SIZE, state_dim=sim_params.dims, action_dim=sim_params.action_dims)
        tree = jax.device_put(tree)
        #init = jnp.concatenate([jnp.asarray([sst_params.start.x, sst_params.start.y, sst_params.start.z]), jnp.zeros(sim_params.dims - 3, dtype=jnp.float32)], axis=0)
        #controls = jnp.zeros(sim_params.action_dims)
        tree, _ = rrtree.add_nodes(tree, init, controls, -1, 0.0, 1)

        start_p = time.perf_counter()

        # Solve
        result = jit_while(tree, sst_params, sim_params, callables, obstacles, i)
        result[3].block_until_ready()  # Ensure all computations are done before timing
        tree, key, goal_mask, goal, states, start_idx, iter_val, size = jax.block_until_ready(result)
        timer = time.perf_counter() - start_p


        path, actions = extract_sol(tree, goal_mask, start_idx)
        #print(path)
        #print(actions)
        if actions is not None:
            is_valid = verify_sol(path, actions, obstacles, sst_params, sim_params, callables)
            if not is_valid:
                print("Invalid solution found!")
                print("Path:", path)
                print("Actions:", actions)


        # Calculate cost for stats
        cost = tree.costs[jnp.argmax(goal_mask) + start_idx]
        
        costs.append(cost)
        times.append(timer)
        iters.append(iter_val)
        sizes.append(size)
        
        print(f"Found goal after {iter_val} iterations, time: {(timer)*1e3:.3f} ms, tree size: {size}")

    # Final Statistics Logic
    times = jnp.array(times)
    iters = jnp.array(iters)
    sizes = jnp.array(sizes)
    costs = jnp.array(costs)

    print(f"\nbatch size: {sim_params.batch_size}, branch factor {A}")
    print(f"Average time over 100 runs: {jnp.mean(times)*1e3:.3f} ms, {jnp.mean(iters):.2f} iterations, size {jnp.mean(sizes):.2f}")
    print(f"Median time over 100 runs: {jnp.median(times)*1e3:.3f} ms, {jnp.median(iters)} iterations, median size {jnp.median(sizes)}")
    print(f"min time over 100 runs: {jnp.min(times)*1e3:.3f} ms, {jnp.min(iters)} iterations, min size {jnp.min(sizes)}")
    print(f"max time over 100 runs: {jnp.max(times)*1e3:.3f} ms, {jnp.max(iters)} iterations, max size {jnp.max(sizes)}")
    print(f"Average cost over 100 runs: {jnp.mean(costs):.3f}, min cost: {jnp.min(costs):.3f}, max cost: {jnp.max(costs):.3f}")
 