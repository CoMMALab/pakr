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


@partial(jax.jit, static_argnums=(3, 4, 5))
def aorrt_iteration(tree, rng_key, obstacles, sst_params, sim_params, callables, best_cost, gmm, p_gmm):
    rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)

    # 1. Sample + NN
    seed_pts = biased_sample_fn(sim_params, subkey1, callables, gmm, p_gmm)
    parents, _ = helper.nearest_neighbor_masked(
        sim_params, callables.dist_fn, tree.states, tree.tree_size, seed_pts
    )
    start_states = tree.states[parents]

    # 2. Rollout
    actions = callables.sampact_fn(sim_params, subkey2)
    states_end, valid_mask, dist_traveled = propagate.rollout_final(
        start_states, actions, obstacles, sst_params, sim_params, callables
    )

    # 3. Static padding
    valid_mask = valid_mask.at[-1].set(False)
    states_end = states_end.at[-1].set(jnp.zeros(sim_params.dims))
    actions = actions.at[-1].set(jnp.zeros(sim_params.action_dims))
    parents = parents.at[-1].set(-1)
    dist_traveled = dist_traveled.at[-1].set(0.0)

    # 4. Filter valid indices
    valid_idx = jnp.nonzero(
        valid_mask,
        size=sim_params.batch_size,
        fill_value=-1
    )[0]

    new_states = states_end[valid_idx]
    new_actions = actions[valid_idx]
    new_parents = parents[valid_idx]

    # g-cost
    new_costs = tree.costs[new_parents] + dist_traveled[valid_idx]

    # ---------- AO PRUNING ----------
    goal = sst_params.goal
    h = jnp.linalg.norm(new_states[:, :3] - jnp.asarray([goal.x, goal.y, goal.z]), axis=-1)
    ao_mask = new_costs + h <= best_cost

    # combine with validity
    ao_mask = ao_mask & (valid_idx >= 0)
    num_new = jnp.sum(ao_mask)

    # index-based masking
    ao_idx = jnp.nonzero(
        ao_mask,
        size=sim_params.batch_size,
        fill_value=-1
    )[0]

    new_states = new_states[ao_idx]
    new_actions = new_actions[ao_idx]
    new_parents = new_parents[ao_idx]
    new_costs = new_costs[ao_idx]

    # 5. Insert
    tree, start_idx = rrtree.add_nodes(
        tree,
        new_states,
        new_actions,
        new_parents,
        new_costs,
        num_new
    )

    # 6. Goal check
    goal_mask = helper.reached_goal(
        new_states, sst_params.goal, sst_params.goal_radius
    )

    return tree, rng_key, goal_mask, jnp.sum(goal_mask), new_states, start_idx




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
            gmm_params,    # NEW argument
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


####################
# hyperparmaeter test
#####################

import itertools
from dataclasses import dataclass
import pandas as pd

# -------------------------
# New Strategy Functions
# -------------------------
def get_training_data(strategy, tree, states, goal_mask, start_idx, best_cost, sst_params):
    tree_size = int(tree.tree_size)
    all_states = np.array(tree.states[:tree_size])
    all_costs = np.array(tree.costs[:tree_size])
    all_parents = np.array(tree.parents[:tree_size])
    
    goal = jnp.array([sst_params.goal.x, sst_params.goal.y, sst_params.goal.z])

    if strategy == "entire_tree":
        return all_states, np.ones(len(all_states))

    # Identify solution path indices
    sol_indices = []
    if jnp.any(goal_mask):
        curr = int(start_idx + jnp.argmax(goal_mask))
        while curr != -1:
            sol_indices.append(curr)
            curr = all_parents[curr]
    
    if strategy == "solution_only":
        return all_states[sol_indices], np.ones(len(sol_indices))

    if strategy == "sol_plus_ancestors":
        # Nodes whose lineage eventually hits the solution path (simplified: nodes with parents in solution)
        mask = np.isin(all_parents, sol_indices) | np.isin(np.arange(tree_size), sol_indices)
        return all_states[mask], np.ones(np.sum(mask))

    if strategy == "sol_ancestor_goodcost":
        # Ancestors + nodes where g+h is close to optimal
        h = np.linalg.norm(all_states[:, :3] - goal, axis=-1)
        f_costs = all_costs + h
        # Threshold: nodes within 20% of current best cost
        cost_mask = f_costs <= (best_cost * 1.2)
        anc_mask = np.isin(all_parents, sol_indices) | np.isin(np.arange(tree_size), sol_indices)
        final_mask = cost_mask | anc_mask
        return all_states[final_mask], np.ones(np.sum(final_mask))

# -------------------------
# Grid Search Harness
# -------------------------
def run_grid_search(obstacles, sst_params, sim_params, callables):
    # Hyperparameter Grid
    param_grid = {
        'K': [5, 10, 20],
        'p_gmm_scale': [0.3, 0.5, 0.8],
        'strategy': ["entire_tree", "solution_only", "sol_plus_ancestors", "sol_ancestor_goodcost"]
    }
    
    keys = param_grid.keys()
    combinations = list(itertools.product(*param_grid.values()))
    results = []

    print(f"Starting Grid Search: {len(combinations)} combinations, 10 trials each.")

    for combo in combinations:
        params_dict = dict(zip(keys, combo))
        K, p_gmm_val, strat = params_dict['K'], params_dict['p_gmm_scale'], params_dict['strategy']
        
        trial_costs = []
        
        for trial in range(10):
            # Reset Tree
            tree = rrtree.KinoTree.init(MAX_TREE_SIZE, sim_params.dims, sim_params.action_dims)
            # ... (init code same as your original) ...
            
            # Run AO-RRT for a fixed duration (e.g., 5s for search)
            # Return the best_cost found
            final_cost = run_ao_rrt_instance(tree, K, p_gmm_val, strat, sst_params, sim_params, callables, obstacles, duration=5.0)
            trial_costs.append(final_cost)
            
        avg_cost = np.mean([c for c in trial_costs if c != np.inf])
        success_rate = np.mean([1 if c != np.inf else 0 for c in trial_costs])
        
        results.append({
            **params_dict,
            'avg_cost': avg_cost,
            'success_rate': success_rate
        })
        print(f"Finished: {params_dict} -> Success: {success_rate:.2f}, Avg Cost: {avg_cost:.2f}")

    return pd.DataFrame(results)

# (Helper to wrap your loop logic into a function)
def run_ao_rrt_instance(tree, K, p_gmm, strategy, sst_params, sim_params, callables, obstacles, duration):
    # This contains your while loop logic, returning best_cost
    # ...
    return best_cost















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

    
    


    obstacles = helper.get_obs(args.env)

    # ------------------------------------------------------------
    # AO-RRT driver (single instance, 10s anytime run)
    # ------------------------------------------------------------

    MAX_TIME = 10.0  # seconds
    MAX_TREE_SIZE = 70000
    K = 10  # number of GMM components
    D = sim_params.dims

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
    # GMM initialization (seed with random rollouts)
    # ------------------------------------------------------------
    print("Seeding initial GMM from random rollouts...")
    gmm_params = initialize_gmm(
        init_state,
        obstacles,
        sst_params,
        sim_params,
        callables,
        K=K,
        seed_rollouts=False,
        N_seed=5000
    )

    # Start with p_gmm = 0
    p_gmm = 0.0

    # ------------------------------------------------------------
    # 🔥 JIT warm-up (compile once)
    # ------------------------------------------------------------
    print("JIT warm-up...")

    _ = jit_while(
        tree,
        sst_params,
        sim_params,
        callables,
        obstacles,
        best_cost=jnp.inf,
        i=0,
        gmm_params=gmm_params,
        p_gmm=p_gmm
    )

    # Force compilation to finish
    jax.block_until_ready(_)
    print("Warm-up complete.")

    # ------------------------------------------------------------
    # AO-RRT anytime loop
    # ------------------------------------------------------------
    print("\nStarting AO-RRT...\n")

    t0 = time.perf_counter()
    ao_iter = 0
    times = []
    costs = []
    best_cost = jnp.inf

    while True:
        gc.collect
        elapsed = time.perf_counter() - t0
        if elapsed >= MAX_TIME:
            break

        rnd = np.random.randint(0, 2**31 - 1)
        tree, key, goal_mask, goal, states, start_idx, iters, size = jit_while(
            tree,
            sst_params,
            sim_params,
            callables,
            obstacles,
            best_cost,
            rnd,
            gmm_params,
            p_gmm
        )

        # If at least one goal reached, update incumbent
        if goal > 0:
            idx = jnp.argmax(goal_mask)
            sol_cost = tree.costs[start_idx + idx]

            if sol_cost < best_cost:
                best_cost = sol_cost

                times.append(elapsed)
                costs.append(sol_cost)

                print(
                    f"[AO iter {ao_iter:03d}] "
                    f"time={elapsed:6.2f}s | "
                    f"cost={sol_cost:8.3f} | "
                    f"tree size={int(size)} | "
                    f"iters={int(iters)}"
                )

        # ------------------------------------------------------------
        # Update GMM after each iteration
        # ------------------------------------------------------------
        # Extract states to fit GMM: use all valid expanded nodes in tree
        # For simplicity, use tree.states[:tree.tree_size]
        X_update, sample_weights = extract_gmm_training_data(tree, states, goal_mask, start_idx)

        gmm_params = fit_gmm_from_data(X_update, K=10)

        p_gmm = 0.5

        ao_iter += 1

    # ------------------------------------------------------------
    # Print final GMM metrics
    # ------------------------------------------------------------
    print("\nFinal GMM metrics:")
    print("Weights:", gmm_params.weights)
    print("Means (first 3 components):", gmm_params.means[:3])
    print("Covariances (first 3 components):", gmm_params.covs[:3])


    print("\nAO-RRT finished.")
    lower_bound_cost = jnp.linalg.norm(
        jnp.asarray([sst_params.start.x, sst_params.start.y, sst_params.start.z]) -
        jnp.asarray([sst_params.goal.x, sst_params.goal.y, sst_params.goal.z])
    )
    # ------------------------------------------------------------
    # Plot cost vs runtime (with lower bound)
    # ------------------------------------------------------------
    if len(times) > 0:
        times = np.asarray(times)
        costs = np.asarray(costs)

        plt.figure(figsize=(6, 4))

        # AO-RRT cost improvement
        plt.plot(
            times,
            costs,
            marker="o",
            label="AO-RRT best cost",
        )

        # Lower-bound (straight-line)
        plt.hlines(
            y=lower_bound_cost,
            xmin=0.0,
            xmax=times[-1],
            linestyles="dashed",
            label="Straight-line lower bound",
        )

        plt.xlabel("Runtime (s)")
        plt.ylabel("Cost")
        plt.title("AO-RRT anytime performance")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("No solution found within time limit.")



