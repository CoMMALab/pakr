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
import mujoco
import mujoco.mjx as mjx
from franka_prop import make_ball_block_propagate_fn, make_franka_propagate_fn
from rrtsolcheck import extract_sol, verify_sol

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
TIERS = [512, 1024, 4096, 16384, 32768, 250000]
NN_BRANCHES = [nn_tier_factory(t) for t in TIERS]


@partial(jax.jit, static_argnums=(3, 4, 5))
def rrt_iteration(tree, rng_key, obstacles, sst_params, sim_params, callables):
    """
    Upgraded MJX RRT Iteration:
    - Tiered NN buckets to avoid re-compiling as tree grows.
    - K NN queries expanded to B rollouts (B/A actions per parent).
    """
    B = sim_params.batch_size
    A = 64  # Actions per parent
    K = B // A  # Total unique parents to find

    rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)

    # ------------------------------------------------------------------
    # 1. Sample K target points + Tiered NN Lookup
    # ------------------------------------------------------------------
    seed_pts = callables.sample_fn(sim_params, subkey1)
    seed_pts_small = seed_pts[:K] # Only need K seeds

    # Use lax.switch to choose a bucketed NN kernel based on tree_size
    branch_idx = jnp.digitize(tree.tree_size, jnp.array(TIERS))
    branch_idx = jnp.minimum(branch_idx, len(TIERS) - 1)
    
    operands = (tree.states, tree.tree_size, seed_pts_small)
    parents_small = jax.lax.switch(branch_idx, NN_BRANCHES, operands)

    # ------------------------------------------------------------------
    # 2. Expand K parents to B rollouts
    # ------------------------------------------------------------------
    start_states_small = tree.states[parents_small]
    
    # Repeat states and parent indices A times each to reach Batch Size B
    start_states = jnp.repeat(start_states_small, A, axis=0)
    parents = jnp.repeat(parents_small, A, axis=0)

    # Sample unique actions for every rollout in the batch
    actions = callables.sampact_fn(sim_params, subkey2)

    # ------------------------------------------------------------------
    # 3. MJX Propagate (Rollout)
    # ------------------------------------------------------------------
    # Using callables.prop_fn which points to your MJX make_ball_block_propagate_fn
    states_end, valid_mask, dist_traveled = callables.prop_fn(
        start_states, actions, obstacles, sst_params, sim_params
    )

    # ------------------------------------------------------------------
    # 4. Padding & Filtering (Standard JAX-RRT Logic)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 5. Insert & MJX-Specific Goal Check
    # ------------------------------------------------------------------
    tree, start_idx = rrtree.add_nodes(
        tree, new_states, new_actions, new_parents, new_costs, num_new
    )

    # Maintain your specific ball-block goal check
    goal_mask = helper.reached_goal_EEB(
        new_states, sst_params.goal, sst_params.goal_radius
    )

    return tree, rng_key, goal_mask, jnp.sum(goal_mask), new_states, start_idx



@partial(jax.jit, static_argnums=(1,2,3))
def jit_while(tree, sst_params, sim_params, callables, obstacles, i):
    def body_fn(carry):
        tree, key, goal_mask, goal, states, start_idx, iter = carry

        key, subkey = jax.random.split(key)

        tree, subkey, goal_mask, goal, states, start_idx = rrt_iteration(
            tree, subkey, obstacles, sst_params, sim_params, callables
        )
        #jax.debug.print("Iteration: {x}, size: {y}", x=iter, y=tree.tree_size)

        return (tree, key, goal_mask, goal, states, start_idx, iter + 1)

    def cond_fn(carry):
        tree, key, goal_mask, goal, states, start_idx, iter = carry
        # Continue while goal not reached
        return goal == 0 # or whatever scalar stopping condition
    init_carry = (tree, 
                  jax.random.PRNGKey(i),
                  jnp.zeros(sim_params.batch_size, dtype=bool),
                  jnp.array(0, dtype=jnp.int32),
                  jnp.zeros([sim_params.batch_size, sim_params.dims], dtype=jnp.float32),
                  jnp.array(0, dtype=jnp.int32),
                  jnp.array(0, dtype=jnp.int32))

    tree, key, goal_mask, goal, states, start_idx, iter = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    return tree, key, goal_mask, goal, states, start_idx, iter, tree.tree_size

MAX_TREE_SIZE = 250000
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the SST planner.')
    parser.add_argument('--env', type=str, default='envs/tree.csv', help='Path to the environment config file.')
    parser.add_argument('--motion', type=str, default='eeb', help='Define motion type: Double Integrator (di), Dubins Airplane (da), Quadcopter (qc), Mjx Cartpole (mcp)')
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
        case 'frb':
            sst_params = params.sst_params_FRB
            sim_params = params.sim_params_FRB

            # --- MJX model init ---
            model = mujoco.MjModel.from_xml_path("models/franka_block.xml")
            mjx_model = mjx.put_model(model)

            # --- MJX propagate ---
            franka_prop = make_franka_propagate_fn(mjx_model)

            # --- rollout adapter ---
            rollout_fn = propagate.make_frb_rollout(franka_prop)

            callables = params.Callables(
                prop_fn=rollout_fn,
                valid_fn=helper.valid_FRB,
                sample_fn=helper.sample_FRB,
                dist_fn=helper.dist_FRB,
                sampact_fn=helper.sample_actions_FRB,
            )
        case 'eeb':
            sst_params = params.sst_params_EEB
            sim_params = params.sim_params_EEB

            # --- MJX model init ---
            model = mujoco.MjModel.from_xml_path("models/eeonly.xml")
            mjx_model = mjx.put_model(model)

            # --- MJX propagate ---
            franka_prop = make_ball_block_propagate_fn(mjx_model)

            # --- rollout adapter ---
            rollout_fn = propagate.make_frb_rollout(franka_prop)

            callables = params.Callables(
                prop_fn=rollout_fn,
                valid_fn=helper.valid_FRB,
                sample_fn=helper.sample_EEB,
                dist_fn=helper.dist_EEB,
                sampact_fn=helper.sample_actions_EEB,
            )
        case _:
            print("invalid motion type")
            exit

    
    SIM_PARAMS_RESERVED = sim_params
    CALLABLES_RESERVED = callables


    obstacles = helper.get_obs("envs/empty.csv")
    

    tree = rrtree.KinoTree.init(
            max_size=MAX_TREE_SIZE,
            state_dim=sim_params.dims,
            action_dim=sim_params.action_dims
        )
    tree = jax.device_put(tree)

    init_state = jnp.zeros(sim_params.dims, dtype=jnp.float32)

    # set block xyz at indices 4:7
    init_state = init_state.at[4:7].set(
        jnp.array([sst_params.start.x, sst_params.start.y, sst_params.start.z], dtype=jnp.float32)
    )

    init_state = init_state.at[0:2].set(
        jnp.array([0.3, 0.0], dtype=jnp.float32)
    )

    print(init_state, sst_params.goal.x, sst_params.goal.y, sst_params.goal.z)
    # rest of the state (q, dq, rpy, dxyz, drpy) stays zero
    init_controls = jnp.zeros(sim_params.action_dims, dtype=jnp.float32)

    # add initial node to tree
    tree, _ = rrtree.add_nodes(tree, init_state, init_controls, -1, 0.0, 1)


    # ---------------------------
    # 5. Multi-Step Warmup
    # ---------------------------
    print("Compiling JIT (Step 1: Empty Tree)...")
    dummy_key = jax.random.PRNGKey(0)
    
    # First pass: Tree size 1 -> size (1 + batch_size)
    tree, dummy_key, _, _, _, _ = rrt_iteration(
        tree, dummy_key, obstacles, sst_params, sim_params, callables
    )
    tree.states.block_until_ready()

    print("Compiling JIT (Step 2: Non-Empty Tree)...")
    # Second pass: Tree size (1 + batch_size) -> size (1 + 2*batch_size)
    # This triggers JAX to handle the 'non-empty' logic path in NN search
    tree, dummy_key, _, _, _, _ = rrt_iteration(
        tree, dummy_key, obstacles, sst_params, sim_params, callables
    )
    tree.states.block_until_ready()
    
    print("JIT compilation done! Resetting tree for actual run...")
    
    # Reset tree to original state (size 1) before the actual loop starts
    tree = rrtree.KinoTree.init(
            max_size=MAX_TREE_SIZE,
            state_dim=sim_params.dims,
            action_dim=sim_params.action_dims
        )
    tree = jax.device_put(tree)
    tree, _ = rrtree.add_nodes(tree, init_state, init_controls, -1, 0.0, 1)

    # ---------------------------
    # 6. Run RRT once
    # ---------------------------
    print("\nRunning RRT...")
    start_time = time.perf_counter()

    i = 0
    # result = jit_while(tree, sst_params, sim_params, callables, obstacles, i)
    # tree, key, goal_mask, goal, states, start_idx, iterations, size = jax.block_until_ready(result)
    # print(states)
    # path, actions = extract_sol(tree, goal_mask, start_idx)
    # print(path, actions)

    # elapsed = time.perf_counter() - start_time
    # print(f"RRT finished in {elapsed*1e3:.3f} ms")
    # print(f"Iterations: {iterations}, tree size: {size}, goal reached: {jnp.sum(goal_mask)}")

    best_dist = jnp.inf
    ky = np.random.randint(0, 1e6)
    key = jax.random.PRNGKey(ky)
    for i in range(200):
        key, subkey = jax.random.split(key)
        start_p = time.perf_counter()
        tree, key, goal_mask, goal_count, states, start_idx = rrt_iteration(
            tree,
            subkey,
            obstacles,
            sst_params,
            sim_params,
            callables,
        )

        # Force execution (important for timing + debugging)
        goal_mask = goal_mask.block_until_ready()
        states = states.block_until_ready()
        timer = time.perf_counter() - start_p
        #print(states)
        # --------------------------------------------------
        # Distance-to-goal tracking
        # --------------------------------------------------
        # assumes goal is on block xyz
        tree_states = tree.states[:tree.tree_size - 1, :]
        #print(tree_states)
        block_xyz = tree_states[:, 4:7]  # adjust if your indexing differs
        goal_xyz = jnp.array([
            sst_params.goal.x,
            sst_params.goal.y,
            sst_params.goal.z,
        ])

        dists = jnp.linalg.norm(block_xyz - goal_xyz, axis=1)
        iter_best = jnp.min(dists)

        if iter_best < best_dist:
            best_iter = i

        best_dist = iter_best
        print(
            f"[iter {i:03d}] "
            f"tree_size={int(tree.tree_size)} | "
            f"new_goal={int(jnp.sum(goal_mask))} | "
            f"best_dist={float(best_dist):.4f} | "
            f"iter_time={timer*1e3:.2f} ms"
        )

        # --------------------------------------------------
        # Early exit if goal reached
        # --------------------------------------------------
        if jnp.any(goal_mask):
            print(f"\n🎯 Goal reached at iteration {i}")
            path, actions = extract_sol(tree, goal_mask, start_idx)
            print(path)
            print(actions)
            np.save("solution_actions.npy", np.array(actions))
            np.save("solution_states.npy", np.array(path))
            break



