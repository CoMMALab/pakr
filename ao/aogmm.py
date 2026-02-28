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

import jax
import jax.numpy as jnp
from jax import random, lax
from typing import NamedTuple
from functools import partial

# -------------------------
# 1. JAX-Native GMM Structures
# -------------------------
class GMMParams(NamedTuple):
    weights: jnp.ndarray  # (K,)
    means: jnp.ndarray    # (K, D)
    covs: jnp.ndarray     # (K, D) - Diagonal variances

def initialize_gmm(sim_params, K=10):
    """Returns a flat, uniform GMM."""
    D = sim_params.dims
    return GMMParams(
        weights=jnp.ones(K) / K,
        means=jnp.zeros((K, D)),
        covs=jnp.ones((K, D)) * 5.0  # High initial variance to cover space
    )

# -------------------------
# 2. JIT-able GMM Fitting (EM Algorithm)
# -------------------------
@jax.jit
def get_log_probs(X, means, covs):
    """Calculates log-pdf of multivariate normal with diagonal covariance."""
    # X: (N, D), means: (K, D), covs: (K, D)
    D = X.shape[-1]
    # Expand dims for broadcasting: (K, N, D)
    diff = X[None, :, :] - means[:, None, :]
    log_det = jnp.sum(jnp.log(covs), axis=-1)  # (K,)
    mahalanobis = jnp.sum((diff**2) / covs[:, None, :], axis=-1)  # (K, N)
    return -0.5 * (D * jnp.log(2 * jnp.pi) + log_det[:, None] + mahalanobis)

# -------------------------
# 3. Sampling Logic (Remains JIT-compatible)
# -------------------------
@jax.jit
def sample_gmm(key, gmm: GMMParams):
    key_cat, key_gauss = random.split(key)
    # 1. pick a component
    k = random.categorical(key_cat, jnp.log(gmm.weights))
    # 2. sample from diagonal Gaussian
    eps = random.normal(key_gauss, shape=(gmm.means.shape[1],))
    sample = gmm.means[k] + eps * jnp.sqrt(gmm.covs[k])
    return sample

@partial(jax.jit, static_argnums=(0, 2,))
def biased_sample_fn(sim_params, key, callables, gmm: GMMParams, p_gmm: float = 0.5):
    key_sel, key_gmm, key_uni = random.split(key, 3)
    use_gmm = random.uniform(key_sel) < p_gmm
    
    x_gmm = sample_gmm(key_gmm, gmm)
    x_uni = callables.sample_fn(sim_params, key_uni)
    
    return jnp.where(use_gmm, x_gmm, x_uni)



# -------------------------
# 4. JIT-compatible Path Extraction
# -------------------------
@partial(jax.jit, static_argnums=(2,))
def extract_path_states(tree, best_node_idx, max_path_len=20):
    """Traces back from best_node_idx to root using lax.scan."""
    def trace_step(curr_idx, _):
        parent = tree.parents[curr_idx]
        # If we hit root (-1), stay at root or use a sentinel
        next_idx = jnp.where(curr_idx == -1, curr_idx, parent)
        return next_idx, curr_idx

    _, path_indices = lax.scan(trace_step, best_node_idx, None, length=max_path_len)
    
    # Mask out invalid indices (sentinels)
    valid_mask = path_indices != -1
    states = tree.states[path_indices]
    return states, valid_mask

# -------------------------
# 1. Core EM Logic (Static X)
# -------------------------
@partial(jax.jit, static_argnums=(1, 2))
def fit_gmm_core(X, K, max_iter=20):
    """Internal EM loop. Expects X to have a static shape."""
    N, D = X.shape
    
    # Initialize: Use first K points as means
    means = X[:K, :] 
    weights = jnp.ones(K) / K
    covs = jnp.ones((K, D))

    def em_step(carry, _):
        w, m, c = carry
        
        # --- E-Step ---
        log_p = get_log_probs(X, m, c) 
        log_resp = log_p + jnp.log(w + 1e-10)[:, None]
        log_resp_norm = jax.nn.logsumexp(log_resp, axis=0)
        resp = jnp.exp(log_resp - log_resp_norm) 
        
        # --- M-Step ---
        N_k = jnp.sum(resp, axis=1) + 1e-6 
        new_w = N_k / N
        new_m = (resp @ X) / N_k[:, None]
        
        diff = X[None, :, :] - new_m[:, None, :] 
        new_c = jnp.sum(resp[:, :, None] * (diff**2), axis=1) / N_k[:, None]
        new_c = jnp.maximum(new_c, 1e-4) # Regularization
        
        return (new_w, new_m, new_c), None

    (final_w, final_m, final_c), _ = lax.scan(em_step, (weights, means, covs), None, length=max_iter)
    return GMMParams(weights=final_w, means=final_m, covs=final_c)

# -------------------------
# 2. Tiered Factory (The Switch Branches)
# -------------------------
def gmm_tier_factory(num_top_nodes, K_comp, max_iter, path_len):
    """Returns a function that fits a GMM to a specific tier of tree data."""
    def branch_fn(operands):
        states, costs, tree_size, sol_idx = operands
        
        # 1. Extract Solution Path (Static length for this tier)
        # Note: extract_path_states handles backtracking to root
        path_states, _ = extract_path_states(tree, sol_idx, max_path_len=path_len)
        
        # 2. Extract Top Cost Nodes (Static size for this tier)
        mask = jnp.arange(costs.shape[0]) < tree_size
        effective_costs = jnp.where(mask, costs, jnp.inf)
        _, indices = lax.top_k(-effective_costs, num_top_nodes)
        top_states = states[indices]
        
        # 3. Combine into a static training set X
        X = jnp.concatenate([path_states, top_states], axis=0)
        
        # 4. Fit and return GMMParams (Shape [K, D] is consistent across all branches)
        return fit_gmm_core(X, K_comp, max_iter)
        
    return branch_fn

# Define Tiers: (Number of Top Nodes, Max Path Length)
# As the tree grows, we look at more 'good' nodes and deeper paths.
TIER_CONFIGS = [
    (128, 20),   # Tier 0
    (256, 50),   # Tier 1
    (512, 100),  # Tier 2
    (1024, 200)  # Tier 3
]

GMM_BRANCHES = [gmm_tier_factory(n, 10, 20, p) for n, p in TIER_CONFIGS]

# -------------------------
# 3. Final JIT-able Entry Point
# -------------------------
@partial(jax.jit, static_argnums=(2,))
def fit_gmm_jax(tree, sol_idx, K=10):
    """
    Fits GMM using tiered branches to maintain static shapes for JIT.
    sol_idx: Index of the best node (to extract solution path).
    """
    # Select branch based on current tree size
    thresholds = jnp.array([c[0] for c in TIER_CONFIGS])
    branch_idx = jnp.digitize(tree.tree_size, thresholds)
    branch_idx = jnp.minimum(branch_idx, len(GMM_BRANCHES) - 1)
    
    operands = (tree.states, tree.costs, tree.tree_size, sol_idx)
    
    # Switch logic: returns GMMParams regardless of which branch is taken
    return jax.lax.switch(branch_idx, GMM_BRANCHES, operands)


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
TIERS = [512, 1024, 4096, 16384, 32768, 64536, 200000]
NN_BRANCHES = [nn_tier_factory(t) for t in TIERS]



print("JAX version:", jax.__version__)
print("Devices:", jax.devices())


@partial(jax.jit, static_argnums=(3, 4, 5))
def aorrt_iteration(tree, rng_key, obstacles, sst_params, sim_params, callables, best_cost, gmm, p_gmm):
    """
    Optimized AO-RRT iteration: 
    1. Tiered NN via jax.lax.switch
    2. Multi-action expansion (A actions per parent)
    3. AO-Pruning (f-cost < best_cost)
    """
    B = sim_params.batch_size
    A = 128            # Actions per parent
    K = B // A         # Number of unique NN queries
    
    rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)

    # 1. Sample K seed points for NN
    seed_pts = biased_sample_fn(sim_params, subkey1, callables, gmm, p_gmm)[:K]

    # 2. Tiered Nearest Neighbor (Switch logic)
    branch_idx = jnp.digitize(tree.tree_size, jnp.array(TIERS))
    branch_idx = jnp.minimum(branch_idx, len(TIERS) - 1)
    
    operands = (tree.states, tree.tree_size, seed_pts)
    parents_small = jax.lax.switch(branch_idx, NN_BRANCHES, operands) 

    # 3. Expand Parents and Sample B Actions
    # This repeats each parent A times to fill the batch B
    start_states = jnp.repeat(tree.states[parents_small], A, axis=0)
    parents = jnp.repeat(parents_small, A, axis=0)
    actions = callables.sampact_fn(sim_params, subkey2)

    # 4. Rollout
    states_end, valid_mask, dist_traveled = propagate.rollout_final(
        start_states, actions, obstacles, sst_params, sim_params, callables
    )

    # 5. Static padding for JIT (ensure indices don't OOB)
    valid_mask = valid_mask.at[-1].set(False)
    states_end = states_end.at[-1].set(jnp.zeros(sim_params.dims))
    actions = actions.at[-1].set(jnp.zeros(sim_params.action_dims))
    parents = parents.at[-1].set(-1)
    dist_traveled = dist_traveled.at[-1].set(0.0)

    # 6. Calculate Costs for AO-Pruning
    # g(n) = parent_cost + edge_cost
    new_costs = tree.costs[parents] + dist_traveled
    
    # h(n) = Euclidean distance to goal (Admissible heuristic)
    goal = sst_params.goal
    goal_vec = jnp.array([goal.x, goal.y, goal.z])
    h = jnp.linalg.norm(states_end[:, :3] - goal_vec, axis=-1)
    
    # 7. AO-Pruning Mask: f(n) = g(n) + h(n)
    ao_mask = (new_costs + h <= best_cost) & valid_mask

    # 8. Filter and Compact via Nonzero
    # We use B as the size to keep shapes static for JIT
    ao_idx = jnp.nonzero(ao_mask, size=B, fill_value=-1)[0]
    
    num_new = jnp.sum(ao_mask)
    final_states = states_end[ao_idx]
    final_actions = actions[ao_idx]
    final_parents = parents[ao_idx]
    final_costs = new_costs[ao_idx]

    # 9. Insert into Tree
    tree, start_idx = rrtree.add_nodes(
        tree,
        final_states,
        final_actions,
        final_parents,
        final_costs,
        num_new
    )

    # 10. Goal check on the newly inserted nodes
    goal_mask = helper.reached_goal(
        final_states, sst_params.goal, sst_params.goal_radius
    )

    return tree, rng_key, goal_mask, jnp.sum(goal_mask), final_states, start_idx




@partial(jax.jit, static_argnums=(1,2,3))
def jit_while(tree, sst_params, sim_params, callables, obstacles, best_cost, i, gmm_params, p_gmm):

    def body_fn(carry):
        tree, key, goal_mask, goal, states, start_idx, iter = carry

        key, subkey = jax.random.split(key)

        tree, subkey, goal_mask, goal, states, start_idx = aorrt_iteration(
            tree,
            subkey,
            obstacles,
            sst_params,
            sim_params,
            callables,
            best_cost,
            gmm_params,
            p_gmm
        )


        return (tree, key, goal_mask, goal, states, start_idx, iter + 1)

    def cond_fn(carry):
        tree, key, goal_mask, goal, states, start_idx, iter = carry
        # Continue while goal not reached
        return goal == 0  # or whatever scalar stopping condition
    init_carry = (tree, 
                  jax.random.PRNGKey(i),
                  jnp.zeros(sim_params.batch_size, dtype=bool),
                  jnp.array(0, dtype=jnp.int32),
                  jnp.zeros([sim_params.batch_size, sim_params.dims], dtype=jnp.float32),
                  jnp.array(0, dtype=jnp.int32),
                  jnp.array(0, dtype=jnp.int32))

    tree, key, goal_mask, goal, states, start_idx, iter = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    return tree, key, goal_mask, goal, states, start_idx, iter, tree.tree_size

MAX_TREE_SIZE = 70000
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the SST planner.')
    parser.add_argument('--env', type=str, default='envs/tree.csv', help='Path to the environment config file.')
    parser.add_argument('--motion', type=str, default='di', help='Define motion type: Double Integrator (di), Dubins Airplane (da), Quadcopter (qc), Mjx Cartpole (mcp)')
    args = parser.parse_args()

    match args.motion:
        case 'di':
            sst_params = params.sst_params_DI
            sim_params = params.sim_params_DI
            callables = params.Callables()
        case 'da':
            sst_params = params.sst_params_DA
            sim_params = params.sim_params_DA
            callables = params.callables_DA
        case 'qc':
            sst_params = params.sst_params_QC
            sim_params = params.sim_params_QC
            callables = params.callables_QC
        # case 'mcp':
        #     sst_params = params.sst_params_DI
        #     sim_params = params.sim_params_DI
        #     callables = params.callables_MCP
        #     # model = mujoco.MjModel.from_xml_path(xml_path)
        #     # mjx_model = mjx.put_model(model)
        #     # propagate_fn = make_propagate_fn(mjx_model)
        case _:
            print("invalid motion type")
            exit

    
    
    SIM_PARAMS_RESERVED = sim_params
    CALLABLES_RESERVED = callables

    obstacles = helper.get_obs(args.env)
    
    # ------------------------------------------------------------
    # AO-RRT driver (single instance, 10s anytime run)
    # ------------------------------------------------------------
    import matplotlib.pyplot as plt

    MAX_TIME = 3.0  # seconds
    MAX_TREE_SIZE = 100000

    # ------------------------------------------------------------
    # Tree initialization
    # ------------------------------------------------------------
    tree = rrtree.KinoTree.init(
        max_size=MAX_TREE_SIZE,
        state_dim=sim_params.dims,
        action_dim=sim_params.action_dims
    )
    tree = jax.device_put(tree)

    init_state = jnp.concatenate(
        [
            jnp.asarray(
                [sst_params.start.x, sst_params.start.y, sst_params.start.z],
                dtype=jnp.float32
            ),
            jnp.zeros(sim_params.dims - 3, dtype=jnp.float32),
        ],
        axis=0,
    )

    init_action = jnp.zeros(sim_params.action_dims, dtype=jnp.float32)
    tree, _ = rrtree.add_nodes(tree, init_state, init_action, -1, 0.0, 1)

    # ------------------------------------------------------------
    # AO bookkeeping
    # ------------------------------------------------------------
    best_cost = jnp.inf
    times = []
    costs = []


    # ------------------------------------------------------------
    # 🔥 JIT warm-up (compile once)
    # ------------------------------------------------------------
    print("JIT warm-up...")


    gmm_params = initialize_gmm(sim_params, K=10)

    p_gmm = 0
    # 1. Warm up the inner iteration function directly
    # This ensures every branch of the logic is traced.
    _ = aorrt_iteration(
        tree,
        jax.random.PRNGKey(0),
        obstacles,
        sst_params,
        sim_params,
        callables,
        jnp.inf,
        gmm_params,
        p_gmm
    )

    # 2. Warm up the while loop with a small, guaranteed execution
    # We pass a dummy 'best_cost' and ensure it runs at least once
    _ = jit_while(
        tree,
        sst_params,
        sim_params,
        callables,
        obstacles,
        jnp.array(float('inf')), 
        0,
        gmm_params,
        p_gmm
    )

    jax.block_until_ready(_)
    print("Warm-up complete.")

    # ------------------------------------------------------------
    # AO-RRT anytime loop
    # ------------------------------------------------------------
    print("\nStarting AO-RRT...\n")

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import time
    import jax
    import jax.numpy as jnp
    import gc

    # --- Configuration ---
    N_RUNS = 100
    MAX_TIME = 3.0       
    COST_THRESHOLD = 1.55
    GT_MIN_COST = 1.403  
    all_run_data = []

    # To track the final result of each run for the global average
    run_summaries = []

    p_gmm = 0.5
    for run_id in range(N_RUNS):
        print(f"\n--- Starting Run {run_id+1}/{N_RUNS} ---")
        gc.collect() 
        
        # Initialize Tree & Root (Reset for each run)
        tree = rrtree.KinoTree.init(max_size=MAX_TREE_SIZE, state_dim=sim_params.dims, action_dim=sim_params.action_dims)
        tree = jax.device_put(tree)
        init_state = jnp.concatenate([
            jnp.asarray([sst_params.start.x, sst_params.start.y, sst_params.start.z]), 
            jnp.zeros(sim_params.dims - 3, dtype=jnp.float32)
        ], axis=0)
        tree, _ = rrtree.add_nodes(tree, init_state, jnp.zeros(sim_params.action_dims), -1, 0.0, 1)
        
        ao_iter = 0
        best_cost = float('inf')
        t0 = time.perf_counter()
        
        while True:
            
            

            rnd = np.random.randint(0, 2**31 - 1)
            
            # Call the JIT function
            # Ensure best_cost is passed as a JAX array to prevent recompilation
            tree, key, goal_mask, goal, states, start_idx, iters, size = jit_while(
                tree, sst_params, sim_params, callables, obstacles,
                jnp.array(best_cost, dtype=jnp.float32), rnd,
                gmm_params,
                p_gmm
            )

            if goal > 0:
                idx = jnp.argmax(goal_mask) + start_idx
                sol_cost = float(tree.costs[idx])
                if sol_cost < best_cost:
                    best_cost = sol_cost


            # fitting gmm
            gmm_params = fit_gmm_jax(tree, K=10, sol_idx=idx)


            elapsed = time.perf_counter() - t0
            all_run_data.append({
                "Run": run_id,
                "Iteration": ao_iter,
                "Best Cost": best_cost if best_cost != float('inf') else None,
                "Cumulative Time": elapsed
            })
            
            ao_iter += 1
            # Quick print for progress
            # if ao_iter % 5 == 0: # Print every 5 iters to keep console clean
            #     print(f"  Iter {ao_iter:02d} | Cost: {best_cost:.4f} | Nodes: {size}")
            # Exit conditions
            if (elapsed >= MAX_TIME or best_cost <= COST_THRESHOLD):
                # Capture the final state of this run before breaking
                if run_id > 0:
                    run_summaries.append({
                        "cost": best_cost if best_cost != float('inf') else None,
                        "time": elapsed,
                        "nodes": int(tree.tree_size),
                        "iters": ao_iter
                    })
                print(f"Run {run_id+1} Finished | Final Cost: {best_cost:.4f} | Time: {elapsed:.2f}s | iters: {ao_iter} | nodes: {tree.tree_size}")
                break

            tree = rrtree.KinoTree.init(max_size=MAX_TREE_SIZE, state_dim=sim_params.dims, action_dim=sim_params.action_dims)
            tree = jax.device_put(tree)
            tree, _ = rrtree.add_nodes(tree, init_state, jnp.zeros(sim_params.action_dims), -1, 0.0, 1)
            

    # --- Calculate and Print Final Averages (With Outlier Removal) ---
    
    # 1. Filter for valid costs first
    valid_summaries = [s for s in run_summaries if s['cost'] is not None]
    
    if valid_summaries:
        costs = np.array([s['cost'] for s in valid_summaries])
        
        # Calculate Mean and Std Dev
        mean_cost = np.mean(costs)
        std_cost = np.std(costs)
        
        # Define bounds (2x standard deviation)
        lower_bound = mean_cost - 2 * std_cost
        upper_bound = mean_cost + 2 * std_cost
        
        # 2. Filter out the outliers
        filtered_summaries = [
            s for s in valid_summaries 
            if lower_bound <= s['cost'] <= upper_bound
        ]
        
        # Calculate final stats from filtered data
        final_costs = [s['cost'] for s in filtered_summaries]
        final_times = [s['time'] for s in filtered_summaries]
        final_nodes = [s['nodes'] for s in filtered_summaries]
        final_iters = [s['iters'] for s in filtered_summaries]
        
        avg_cost = np.mean(final_costs)
        avg_time = np.mean(final_times)
        avg_nodes = np.mean(final_nodes)
        avg_iters = np.mean(final_iters)
        outliers_removed = len(valid_summaries) - len(filtered_summaries)
    else:
        avg_cost = float('inf')
        avg_time = np.mean([s['time'] for s in run_summaries])
        avg_nodes = np.mean([s['nodes'] for s in run_summaries])
        outliers_removed = 0

    success_rate = (len(valid_summaries) / N_RUNS) * 100

    print("\n" + "="*30)
    print("      GLOBAL STATISTICS (Filtered)")
    print("="*30)
    print(f"Total Runs:        {N_RUNS}")
    print(f"Success Rate:      {success_rate:.1f}%")
    print(f"Outliers Removed:  {outliers_removed} (outside 2σ)")
    print(f"Avg Best Cost:     {avg_cost:.4f}")
    print(f"Avg Run Time:      {avg_time:.3f}s")
    print(f"Avg Tree Size:     {avg_nodes:.0f} nodes")
    print(f"Avg Iterations:    {avg_iters:.1f} iters")
    print("="*30)

    # --- Data Processing & Alignment ---
    df = pd.DataFrame(all_run_data)
    max_iters_found = df['Iteration'].max()
    full_index = pd.MultiIndex.from_product([range(N_RUNS), range(max_iters_found + 1)], names=['Run', 'Iteration'])
    df_aligned = df.set_index(['Run', 'Iteration']).reindex(full_index)
    df_aligned['Best Cost'] = df_aligned.groupby('Run')['Best Cost'].ffill()
    df_aligned['Cumulative Time'] = df_aligned.groupby('Run')['Cumulative Time'].ffill()
    df_final = df_aligned.reset_index().dropna(subset=["Best Cost"])

    # --- Visualization ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Graph 1: Convergence
    sns.lineplot(data=df_final, x="Iteration", y="Best Cost", ax=ax1, color="dodgerblue", errorbar='sd', label="Mean AO-RRT Cost")
    ax1.axhline(y=GT_MIN_COST, color='red', linestyle='--', label=f'GT Min ({GT_MIN_COST})')
    ax1.set_title("Cost Convergence (N=5 Runs)")
    ax1.set_ylabel("Best Cost")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Graph 2: Iteration Timing
    sns.lineplot(data=df_final, x="Iteration", y="Cumulative Time", ax=ax2, color="forestgreen", errorbar='sd')
    ax2.set_title("Cumulative Runtime Deviation")
    ax2.set_ylabel("Time (s)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ao/aorrt_convergence.png")




