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
import pandas as pd
import seaborn as sns

import jax
from jax import random
import matplotlib.pyplot as plt

import numpy as np
from sklearn.mixture import GaussianMixture
import jax.numpy as jnp
from typing import NamedTuple

# -------------------------
# 1. Define JAX-compatible GMM
# -------------------------
class GMMParams(NamedTuple):
    weights: jnp.ndarray  # (K,)
    means: jnp.ndarray    # (K, D)
    covs: jnp.ndarray     # (K, D) diagonal


def initialize_gmm(init_state, obstacles, sst_params, sim_params, callables, K=10, seed_rollouts=True, N_seed=5000):
    """
    Initialize a GMM for AO-RRT.
    
    If seed_rollouts=True, performs random rollouts from the start state.
    If False, returns an 'empty' GMM with zeros and uniform weights.
    """
    D = sim_params.dims

    if seed_rollouts:
        print("Seeding initial GMM from random rollouts...")
        key_seed = jax.random.PRNGKey(0)

        # replicate start state
        init_states = jnp.tile(init_state, (N_seed, 1))  # (N_seed, D)

        # random actions
        seed_actions = jax.random.uniform(
            key_seed, (N_seed, sim_params.action_dims),
            minval=-sim_params.action_max,
            maxval=sim_params.action_max
        )

        # rollout dynamics
        states_end, valid_mask, _ = propagate.rollout_final(
            init_states, seed_actions, obstacles, sst_params, sim_params, callables
        )

        # collect valid endpoints
        X_seed = np.array(states_end[valid_mask])

        # fit GMM
        gmm_params = fit_gmm_from_data(X_seed, K=K)

    else:
        print("Initializing empty GMM (no seed rollouts)...")
        gmm_params = GMMParams(
            weights=jnp.ones(K) / K,                # uniform weights
            means=jnp.zeros((K, D), dtype=jnp.float32),  # zero means
            covs=jnp.ones((K, D), dtype=jnp.float32)     # large variance, e.g. 1
        )

    return gmm_params

def extract_gmm_training_data(tree, states, goal_mask, start_idx, top_fraction=0.2, min_samples=100):
    """
    Extract states from the tree to refit the GMM.

    Parameters
    ----------
    tree : rrtree.KinoTree
        The current tree.
    states : jnp.ndarray
        Latest batch of states from the iteration.
    goal_mask : jnp.ndarray
        Boolean mask indicating which states reached the goal.
    start_idx : int
        Start index for the latest batch in the tree.
    top_fraction : float
        Fraction of best-cost nodes to include.
    min_samples : int
        Minimum number of samples to include (fallback to entire tree if too few).

    Returns
    -------
    X_update : np.ndarray
        Array of states to fit the GMM.
    sample_weights : np.ndarray
        Corresponding weights (higher for lower-cost nodes).
    """

    tree_size = int(tree.tree_size)
    all_states = np.array(tree.states[:tree_size])
    all_costs = np.array(tree.costs[:tree_size])

    # 1. Include solution path states
    solution_states = states[goal_mask]  # shape (num_goal, D)

    # 2. Include a fraction of lowest-cost nodes
    num_top = max(int(tree_size * top_fraction), min_samples)
    top_indices = np.argsort(all_costs)[:num_top]
    top_states = all_states[top_indices]
    top_costs = all_costs[top_indices]

    # 3. Combine solution states and top nodes
    X_update = np.vstack([solution_states, top_states])

    # 4. Sample weights: inverse cost (add small eps to avoid division by zero)
    top_weights = 1.0 / (1e-6 + np.concatenate([np.zeros(solution_states.shape[0]), top_costs]))
    # give solution states extra weight by setting their cost=0
    sample_weights = top_weights / np.sum(top_weights)  # normalize to sum to 1

    return X_update, sample_weights


# -------------------------
# 2. Fit GMM from data
# -------------------------
def fit_gmm_from_data(X: np.ndarray, K: int = 10, sample_weights: np.ndarray = None):
    """
    X: (N, D) array of points to fit
    K: number of components
    sample_weights: optional sample weights
    """
    gmm = GaussianMixture(
        n_components=K,
        covariance_type="diag",
        reg_covar=1e-4,
        max_iter=200,
    )
    if sample_weights is not None:
        gmm.fit(X, sample_weight=sample_weights)
    else:
        gmm.fit(X)

    gmm_params = GMMParams(
        weights=jnp.asarray(gmm.weights_, dtype=jnp.float32),
        means=jnp.asarray(gmm.means_, dtype=jnp.float32),
        covs=jnp.asarray(gmm.covariances_, dtype=jnp.float32),  # (K, D)
    )

    return gmm_params


def sample_gmm(key, gmm: GMMParams):
    key_cat, key_gauss = random.split(key)

    # 1. pick a component
    k = random.categorical(key_cat, jnp.log(gmm.weights))

    # 2. sample from diagonal Gaussian
    eps = random.normal(key_gauss, shape=(gmm.means.shape[1],))
    sample = gmm.means[k] + eps * jnp.sqrt(gmm.covs[k])

    return sample

def biased_sample_fn(sim_params, key, callables, gmm: GMMParams, p_gmm: float = 0.5):
    key_sel, key_gmm, key_uni = random.split(key, 3)

    use_gmm = random.uniform(key_sel) < p_gmm

    x_gmm = sample_gmm(key_gmm, gmm)
    x_uni = callables.sample_fn(sim_params, key_uni)  # your existing uniform sampling

    return jnp.where(use_gmm, x_gmm, x_uni)


# Global markers for the factory to access
SIM_PARAMS_RESERVED = None
CALLABLES_RESERVED = None

# --- NN TIER LOGIC ---
# Adjust these buckets based on your MAX_TREE_SIZE
TIERS = [512, 1024, 4096, 16384, 32768, 64536, 100000]

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

NN_BRANCHES = [nn_tier_factory(t) for t in TIERS]

# --- CORE AO-RRT LOGIC ---

@partial(jax.jit, static_argnums=(3, 4, 5))
def aorrt_iteration(tree, rng_key, obstacles, sst_params, sim_params, callables, best_cost, gmm, p_gmm):
    """
    Optimized AO-RRT iteration: 
    1. Biased Sampling (GMM vs Uniform)
    2. Tiered NN via jax.lax.switch (K parents)
    3. Multi-action expansion (A actions per parent)
    4. AO-Pruning (f-cost < best_cost)
    """
    B = sim_params.batch_size
    A = 128            # Actions per parent
    K = B // A         # Number of unique NN queries (parents)
    
    rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)

    # 1. Sample K seed points (using GMM bias function)
    seed_pts = biased_sample_fn(sim_params, subkey1, callables, gmm, p_gmm)[:K]

    # 2. Tiered Nearest Neighbor (Switch logic)
    branch_idx = jnp.digitize(tree.tree_size, jnp.array(TIERS))
    branch_idx = jnp.minimum(branch_idx, len(TIERS) - 1)
    
    operands = (tree.states, tree.tree_size, seed_pts)
    parents_small = jax.lax.switch(branch_idx, NN_BRANCHES, operands) 

    # 3. Expand Parents and Sample B Actions
    # This repeats each of the K parents A times to fill the batch B
    start_states = jnp.repeat(tree.states[parents_small], A, axis=0)
    parents = jnp.repeat(parents_small, A, axis=0)
    actions = callables.sampact_fn(sim_params, subkey2)

    # 4. Rollout
    states_end, valid_mask, dist_traveled = propagate.rollout_final(
        start_states, actions, obstacles, sst_params, sim_params, callables
    )

    # 5. Static padding for JIT consistency
    valid_mask = valid_mask.at[-1].set(False)
    states_end = states_end.at[-1].set(jnp.zeros(sim_params.dims))
    actions = actions.at[-1].set(jnp.zeros(sim_params.action_dims))
    parents = parents.at[-1].set(-1)
    dist_traveled = dist_traveled.at[-1].set(0.0)

    # 6. Calculate Costs for AO-Pruning
    # g(n) = parent_cost + edge_cost
    new_costs = tree.costs[parents] + dist_traveled
    
    # h(n) = Euclidean distance to goal (Admissible heuristic)
    goal_vec = jnp.array([sst_params.goal.x, sst_params.goal.y, sst_params.goal.z])
    h = jnp.linalg.norm(states_end[:, :3] - goal_vec, axis=-1)
    
    # 7. AO-Pruning Mask: f(n) = g(n) + h(n)
    ao_mask = (new_costs + h <= best_cost) & valid_mask

    # 8. Filter and Compact via Nonzero
    ao_idx = jnp.nonzero(ao_mask, size=B, fill_value=-1)[0]
    num_new = jnp.sum(ao_mask)
    
    final_states = states_end[ao_idx]
    final_actions = actions[ao_idx]
    final_parents = parents[ao_idx]
    final_costs = new_costs[ao_idx]

    # 9. Insert into Tree
    tree, start_idx = rrtree.add_nodes(
        tree, final_states, final_actions, final_parents, final_costs, num_new
    )

    # 10. Goal check on the newly inserted nodes
    goal_mask = helper.reached_goal(final_states, sst_params.goal, sst_params.goal_radius)

    return tree, rng_key, goal_mask, jnp.sum(goal_mask), final_states, start_idx


@partial(jax.jit, static_argnums=(1,2,3))
def jit_while(tree, sst_params, sim_params, callables, obstacles, best_cost, i, gmm_params, p_gmm):
    def body_fn(carry):
        tree, key, goal_mask, goal, states, start_idx, iter = carry
        key, subkey = jax.random.split(key)
        tree, subkey, goal_mask, goal, states, start_idx = aorrt_iteration(
            tree, subkey, obstacles, sst_params, sim_params, callables, best_cost, gmm_params, p_gmm
        )
        return (tree, key, goal_mask, goal, states, start_idx, iter + 1)

    def cond_fn(carry):
        # Continue while goal not reached in this batch
        return carry[3] == 0 

    init_carry = (
        tree, 
        jax.random.PRNGKey(i),
        jnp.zeros(sim_params.batch_size, dtype=bool),
        jnp.array(0, dtype=jnp.int32),
        jnp.zeros([sim_params.batch_size, sim_params.dims], dtype=jnp.float32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32)
    )

    tree, key, goal_mask, goal, states, start_idx, iter = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    return tree, key, goal_mask, goal, states, start_idx, iter, tree.tree_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='envs/tree.csv')
    parser.add_argument('--motion', type=str, default='di')
    args = parser.parse_args()

    # Param Selection
    match args.motion:
        case 'di':
            sst_params, sim_params, callables = params.sst_params_DI, params.sim_params_DI, params.Callables()
        case 'da':
            sst_params, sim_params, callables = params.sst_params_DA, params.sim_params_DA, params.callables_DA
        case 'qc':
            sst_params, sim_params, callables = params.sst_params_QC, params.sim_params_QC, params.callables_QC
    
    SIM_PARAMS_RESERVED = sim_params
    CALLABLES_RESERVED = callables
    obstacles = helper.get_obs(args.env)

    MAX_TIME, MAX_TREE_SIZE = 3.0, 100000
    N_RUNS, COST_THRESHOLD, GT_MIN_COST = 10, 1.55, 1.403
    all_run_data, run_summaries = [], []

    # --- JIT Warmup Prep ---
    # We initialize a temporary tree just to prime the XLA compiler
    temp_tree = rrtree.KinoTree.init(MAX_TREE_SIZE, sim_params.dims, sim_params.action_dims)
    init_state = jnp.concatenate([
        jnp.array([sst_params.start.x, sst_params.start.y, sst_params.start.z]), 
        jnp.zeros(sim_params.dims - 3)
    ])
    temp_tree, _ = rrtree.add_nodes(temp_tree, init_state, jnp.zeros(sim_params.action_dims), -1, 0.0, 1)
    
    gmm_params = GMMParams(
        weights=jnp.ones(10) / 10,
        means=jnp.zeros((10, sim_params.dims)),
        covs=jnp.ones((10, sim_params.dims))  # Assuming diagonal covariance based on your fit_gmm logic
    )
    
    print("JIT warm-up (Priming Switch Branches and Multi-Action Logic)...")
    _ = jit_while(temp_tree, sst_params, sim_params, callables, obstacles, jnp.inf, 0, gmm_params, 0.0)
    jax.block_until_ready(_)
    print("Warm-up complete.")

    # --- Main Anytime Loop ---
    for run_id in range(N_RUNS):
        print(f"\n--- Starting Run {run_id+1}/{N_RUNS} ---")
        gc.collect()
        
        # Fresh Tree
        tree = rrtree.KinoTree.init(MAX_TREE_SIZE, sim_params.dims, sim_params.action_dims)
        tree, _ = rrtree.add_nodes(tree, init_state, jnp.zeros(sim_params.action_dims), -1, 0.0, 1)
        
        best_cost, ao_iter, t0 = float('inf'), 0, time.perf_counter()
        p_gmm = 0.0 # Start with purely uniform sampling

        while True:
            elapsed = time.perf_counter() - t0
            if elapsed >= MAX_TIME or best_cost <= COST_THRESHOLD:
                run_summaries.append({"cost": best_cost, "time": elapsed, "nodes": int(tree.tree_size)})
                break

            rnd = np.random.randint(0, 2**31 - 1)
            
            # Execute JIT loop
            tree, _, goal_mask, goal, states, start_idx, iters, size = jit_while(
                tree, sst_params, sim_params, callables, obstacles,
                jnp.array(best_cost, dtype=jnp.float32), rnd, gmm_params, p_gmm
            )

            # Update solution
            if goal > 0:
                sol_cost = float(tree.costs[start_idx + jnp.argmax(goal_mask)])
                if sol_cost < best_cost:
                    best_cost = sol_cost

            all_run_data.append({
                "Run": run_id, "Iteration": ao_iter, 
                "Best Cost": best_cost if best_cost != float('inf') else None,
                "Cumulative Time": elapsed
            })

            # # Update GMM based on the new expansion
            # X_update, _ = extract_gmm_training_data(tree, states, goal_mask, start_idx)
            # gmm_params = fit_gmm_from_data(X_update, K=10)
            
            # # Gradually increase GMM influence as the tree grows
            # p_gmm = min(0.7, p_gmm + 0.05) 
            p_gmm = 0
            
            ao_iter += 1
            print(f"  Iter {ao_iter:02d} | Time: {elapsed:5.2f}s | Best Cost: {best_cost:.4f} | Nodes: {size}")

    # --- Final Statistics ---
    valid_costs = [s['cost'] for s in run_summaries if s['cost'] != float('inf')]
    success_rate = (len(valid_costs) / N_RUNS) * 100
    
    print("\n" + "="*35)
    print(f"      GLOBAL STATISTICS (N={N_RUNS})")
    print("="*35)
    print(f"Success Rate:    {success_rate:.1f}%")
    if valid_costs:
        print(f"Avg Best Cost:   {np.mean(valid_costs):.4f}")
    print(f"Avg Run Time:    {np.mean([s['time'] for s in run_summaries]):.3f}s")
    print(f"Avg Tree Size:   {np.mean([s['nodes'] for s in run_summaries]):.0f} nodes")
    print("="*35)

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
    plt.savefig("ao/aogmm_convergence.png")




