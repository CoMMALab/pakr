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
import csv

# ------------------------------------------------------------------
# 1. Tiered Nearest Neighbor Kernels
# ------------------------------------------------------------------

# Global references for static objects via closure
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

TIERS = [512, 1024, 4096, 16384, 32768, 64536, 131_072, 262_144, 500_000, 1_000_000, 2_000_000]
NN_BRANCHES = [nn_tier_factory(t) for t in TIERS]

# ------------------------------------------------------------------
# 2. JIT Functions with k_factor as static
# ------------------------------------------------------------------

@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def rrt_iteration(tree, rng_key, obstacles, sst_params, sim_params, callables, k_factor):
    """One batched iteration. k_factor is passed as a static argument."""
    B = sim_params.batch_size
    K = B // k_factor  # Recalculated for every unique k_factor

    rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)

    # 1. Sample K target points
    seed_pts = callables.sample_fn(sim_params, subkey1)[:K]

    # 2. Tiered NN Lookup
    branch_idx = jnp.digitize(tree.tree_size, jnp.array(TIERS))
    branch_idx = jnp.minimum(branch_idx, len(TIERS) - 1)
    parents_small = jax.lax.switch(branch_idx, NN_BRANCHES, (tree.states, tree.tree_size, seed_pts))

    # 3. Repeat parents k_factor times
    start_states = jnp.repeat(tree.states[parents_small], k_factor, axis=0)
    parents = jnp.repeat(parents_small, k_factor, axis=0)

    # 4. Sample B actions and Rollout
    actions = callables.sampact_fn(sim_params, subkey2)
    states_end, valid_mask, dist_traveled = propagate.rollout_final(
        start_states, actions, obstacles, sst_params, sim_params, callables
    )

    # 5. Static padding & Filter
    valid_mask = valid_mask.at[-1].set(False)
    valid_idx = jnp.nonzero(valid_mask, size=B, fill_value=-1)[0]

    new_states = states_end[valid_idx]
    new_costs = tree.costs[parents[valid_idx]] + dist_traveled[valid_idx]

    # 6. Insert & Goal Check
    tree, start_idx = rrtree.add_nodes(tree, new_states, actions[valid_idx], parents[valid_idx], new_costs, jnp.sum(valid_mask))
    goal_mask = helper.reached_goal(new_states, sst_params.goal, sst_params.goal_radius)

    return tree, rng_key, goal_mask, jnp.sum(goal_mask), new_states, start_idx

@partial(jax.jit, static_argnums=(1, 2, 3, 6))
def jit_while(tree, sst_params, sim_params, callables, obstacles, i, k_factor):
    def body_fn(carry):
        tree, key, _, _, _, _, iter = carry
        tree, key, goal_mask, goal, states, start_idx = rrt_iteration(
            tree, key, obstacles, sst_params, sim_params, callables, k_factor
        )
        return (tree, key, goal_mask, goal, states, start_idx, iter + 1)

    def cond_fn(carry):
        tree, _, _, goal, _, _, _ = carry
        return (goal == 0) & (tree.tree_size < 500_000 - sim_params.batch_size)

    init_carry = (tree, jax.random.PRNGKey(i), jnp.zeros(sim_params.batch_size, dtype=bool),
                  jnp.array(0, dtype=jnp.int32), jnp.zeros([sim_params.batch_size, sim_params.dims]),
                  jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32))

    return jax.lax.while_loop(cond_fn, body_fn, init_carry)

# ------------------------------------------------------------------
# 3. Execution Sweep Logic
# ------------------------------------------------------------------

# ... (Imports and JIT functions remain unchanged) ...

# ... (Imports and JIT functions remain unchanged) ...

# --- Configuration ---
BATCH_SIZES = [4096, 8192, 16384, 32768]
BRANCHING_FACTORS = [16, 32, 64, 128]
NUM_RUNS = 100
MAX_TREE_SIZE = 500000

# Environment-specific parameters
ENV_CONFIGS = {
    "tree":   {"dt": 0.2, "tte": 10},
    "narrow": {"dt": 0.2, "tte": 10},
    "house":  {"dt": 0.1, "tte": 20}
}

def save_results(env_name, motion, b_size, k_factor, data):
    """Saves results with environment-specific naming."""
    valid_mask = np.isfinite(data[:, 3])
    success_data = data[valid_mask]
    num_runs = len(data)
    num_success = len(success_data)
    success_rate = (num_success / num_runs) * 100

    if num_success > 0:
        stats = {
            "MIN": np.min(success_data, axis=0),
            "AVG": np.mean(success_data, axis=0),
            "MEDIAN": np.median(success_data, axis=0),
            "MAX": np.max(success_data, axis=0),
            "STDEV": np.std(success_data, axis=0)
        }
    else:
        stats = {k: np.zeros(4) for k in ["MIN", "AVG", "MEDIAN", "MAX", "STDEV"]}

    os.makedirs("results", exist_ok=True)
    filepath = f"results/results_{env_name}_{motion}_B{b_size}_k{k_factor}.csv"
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Runtime_s", "Iterations", "Tree_Size", "Cost"])
        writer.writerow(["SUCCESS_RATE", f"{success_rate:.2f}%", f"{num_success}/{num_runs}", "-", "-"])
        
        for name, values in stats.items():
            writer.writerow([name, f"{values[0]:.6f}", f"{values[1]:.2f}", f"{values[2]:.2f}", f"{values[3]:.4f}"])
        
        writer.writerow([]) 
        writer.writerow(["Run_ID", "Runtime_s", "Iterations", "Tree_Size", "Cost"])
        for idx, row in enumerate(data):
            writer.writerow([idx + 1, f"{row[0]:.6f}", int(row[1]), int(row[2]), f"{row[3]:.4f}"])

# --- Updated Configuration ---
BATCH_SIZES = [4096, 8192, 16384, 32768]
BRANCHING_FACTORS = [2] # We will now loop through these
NUM_RUNS = 100
MAX_TREE_SIZE = 2_000_000

# Environment-specific parameters (Only 'tree' as requested)
ENV_CONFIGS = {
    "quadtree": {"dt": 0.1, "tte": 10},
}

def run_parameter_sweep():
    global SIM_PARAMS_RESERVED, CALLABLES_RESERVED
    parser = argparse.ArgumentParser()
    parser.add_argument('--motion', type=str, default='qc')
    args = parser.parse_args()

    match args.motion:
        case 'di': sst_base, sim_base, CALLABLES_RESERVED = params.sst_params_DI, params.sim_params_DI, params.Callables()
        case 'da': sst_base, sim_base, CALLABLES_RESERVED = params.sst_params_DA, params.sim_params_DA, params.callables_DA
        case 'qc': sst_base, sim_base, CALLABLES_RESERVED = params.sst_params_QC, params.sim_params_QC, params.callables_QC

    # 1. Select only the 'tree' environment
    env_name = "quadtree"
    config = ENV_CONFIGS[env_name]
    print(f"\n========== SWEEPING ENV: {env_name.upper()} (dt={config['dt']}, tte={config['tte']}) ==========")
    obstacles = helper.get_obs(f"envs/{env_name}.csv")

    # 2. Loop: Batch Sizes
    for b_size in BATCH_SIZES:
        
        # 3. Inner Loop: Branching Factors (k_factor)
        for k_factor in BRANCHING_FACTORS:
            
            # Ensure batch size is divisible by k_factor to avoid K=0 or logic errors
            if b_size < k_factor:
                print(f"Skipping: B={b_size}, k={k_factor} (B must be >= k)")
                continue
                
            print(f">>> Config: B={b_size}, k={k_factor} | Progress: [", end="", flush=True)
            
            # Apply Environment-specific dt and tte
            sim_params = sim_base.replace(batch_size=b_size, dt=config['dt'])
            sst_params = sst_base.replace(batch_size=b_size, time_to_evolve=config['tte'])
            SIM_PARAMS_RESERVED = sim_params

            # Prepare initial state
            controls = jnp.zeros(sim_params.action_dims)
            init = jnp.concatenate([
                jnp.asarray([sst_params.start.x, sst_params.start.y, sst_params.start.z]), 
                jnp.zeros(sim_params.dims - 3, dtype=jnp.float32)
            ], axis=0)

            # Warm-up (Important: Re-JIT because b_size and k_factor are static)
            tree_init = rrtree.KinoTree.init(MAX_TREE_SIZE, sim_params.dims, sim_params.action_dims)
            tree_init, _ = rrtree.add_nodes(tree_init, init, controls, -1, 0.0, 1)
            # block_until_ready on warm-up ensures JIT finishes before timing starts
            _ = jax.block_until_ready(jit_while(tree_init, sst_params, sim_params, CALLABLES_RESERVED, obstacles, 0, k_factor))
            
            run_data = []
            for i in range(NUM_RUNS):
                if (i+1) % 20 == 0: print("#", end="", flush=True)
                
                gc.collect()
                tree = rrtree.KinoTree.init(MAX_TREE_SIZE, sim_params.dims, sim_params.action_dims)
                tree = jax.device_put(tree)
                tree, _ = rrtree.add_nodes(tree, init, controls, -1, 0.0, 1)
                
                start_p = time.perf_counter()
                res = jit_while(tree, sst_params, sim_params, CALLABLES_RESERVED, obstacles, i, k_factor)
                jax.block_until_ready(res)
                duration = time.perf_counter() - start_p
                
                final_tree, _, goal_mask, goal_count, _, start_idx, iters = res
                
                if goal_count > 0:
                    cost = final_tree.costs[jnp.argmax(goal_mask) + start_idx]
                else:
                    cost = float('inf') 
                
                run_data.append([duration, float(iters), float(final_tree.tree_size), float(cost)])

            save_results(env_name, args.motion, b_size, k_factor, np.array(run_data))
            print("] Done.")

if __name__ == "__main__":
    run_parameter_sweep()