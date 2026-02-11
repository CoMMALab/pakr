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
from franka_prop import make_franka_propagate_fn

#@partial(jax.jit, static_argnums=(3, 4, 5))
def rrt_iteration(tree, rng_key, obstacles, sst_params, sim_params, callables):
    """
    One batched kinodynamic RRT iteration:
    - sample target states
    - nearest-neighbor lookup (Voronoi bias)
    - sample actions
    - rollout
    - insert valid endpoints
    """
    rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)

    # ------------------------------------------------------------------
    # 1. Sample target points + nearest neighbor
    # ------------------------------------------------------------------
    seed_pts = callables.sample_fn(sim_params, subkey1)
    parents, _ = helper.nearest_neighbor_mjx(
        sim_params, callables.dist_fn, tree.states, tree.tree_size, seed_pts
    )
    start_states = tree.states[parents]
    print(start_states)

    # ------------------------------------------------------------------
    # 2. Sample actions and rollout
    # ------------------------------------------------------------------
    actions = callables.sampact_fn(sim_params, subkey2)

    states_end, valid_mask, dist_traveled = callables.prop_fn(
        start_states, actions, obstacles, sst_params, sim_params
    )
    print(states_end)

    # ------------------------------------------------------------------
    # 3. Padding for static shapes
    # ------------------------------------------------------------------
    valid_mask = valid_mask.at[-1].set(False)
    states_end = states_end.at[-1].set(jnp.zeros(sim_params.dims))
    actions = actions.at[-1].set(jnp.zeros(sim_params.action_dims))
    parents = parents.at[-1].set(-1)
    dist_traveled = dist_traveled.at[-1].set(0.0)

    # ------------------------------------------------------------------
    # 4. Filter valid insertions (index-based masking)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 5. Insert nodes
    # ------------------------------------------------------------------
    tree, start_idx = rrtree.add_nodes(
        tree,
        new_states,
        new_actions,
        new_parents,
        new_costs,
        num_new
    )

    # ------------------------------------------------------------------
    # 6. Goal check
    # ------------------------------------------------------------------
    goal_mask = helper.reached_goal_FRB(
        new_states, sst_params.goal, sst_params.goal_radius
    )

    return tree, rng_key, goal_mask, jnp.sum(goal_mask), new_states, start_idx



#@partial(jax.jit, static_argnums=(1,2,3))
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
    parser.add_argument('--motion', type=str, default='frb', help='Define motion type: Double Integrator (di), Dubins Airplane (da), Quadcopter (qc), Mjx Cartpole (mcp)')
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

        case _:
            print("invalid motion type")
            exit

    
    


    obstacles = helper.get_obs("envs/empty.csv")
    

    tree = rrtree.KinoTree.init(
            max_size=MAX_TREE_SIZE,
            state_dim=sim_params.dims,
            action_dim=sim_params.action_dims
        )
    tree = jax.device_put(tree)

    init_state = jnp.zeros(sim_params.dims, dtype=jnp.float32)

    # set block xyz at indices 14:17
    init_state = init_state.at[14:17].set(
        jnp.array([sst_params.start.x, sst_params.start.y, sst_params.start.z], dtype=jnp.float32)
    )

    # rest of the state (q, dq, rpy, dxyz, drpy) stays zero
    init_controls = jnp.zeros(sim_params.action_dims, dtype=jnp.float32)

    # add initial node to tree
    tree, _ = rrtree.add_nodes(tree, init_state, init_controls, -1, 0.0, 1)

    # ---------------------------
    # 5. Dummy run to trigger JIT
    # ---------------------------
    print("Compiling JIT...")
    dummy_key = jax.random.PRNGKey(0)
    _, _, xx= callables.prop_fn(
        jnp.tile(init_state[None, :], (sim_params.batch_size, 1)),
        jnp.zeros((sim_params.batch_size, sim_params.action_dims)),
        obstacles,
        sst_params,
        sim_params,
    )
    xx.block_until_ready()
    print("JIT compilation done!")

    # print("\nRunning ONE rrt_iteration (no JIT, no while)...")

    # key = jax.random.PRNGKey(0)

    # tree2, key2, goal_mask2, goal2, states2, start_idx2 = rrt_iteration(
    #     tree,
    #     key,
    #     obstacles,
    #     sst_params,
    #     sim_params,
    #     callables,
    # )

    # # Force execution
    # jax.tree_util.tree_map(
    #     lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
    #     (tree2, key2, goal_mask2, goal2, states2, start_idx2),
    # )

    # print("rrt_iteration finished")
    # print("num new states:", states2.shape[0])
    # print("goal count:", jnp.sum(goal_mask2))
    # print("tree size after:", tree2.tree_size)

    print("\nRunning RRT (Python loop, 100 iters)...")

    key = jax.random.PRNGKey(0)

    best_dist = jnp.inf
    best_iter = -1

    start_time = time.perf_counter()
    print(sst_params.start.x, sst_params.start.y, sst_params.start.z)

    for i in range(2):
        key, subkey = jax.random.split(key)

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
        print(states)
        # --------------------------------------------------
        # Distance-to-goal tracking
        # --------------------------------------------------
        # assumes goal is on block xyz
        tree_states = tree.states[tree.tree_size - 1:, :]
        print(tree_states)
        block_xyz = tree_states[:, 14:17]  # adjust if your indexing differs
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
            f"best_dist={float(best_dist):.4f}"
        )

        # --------------------------------------------------
        # Early exit if goal reached
        # --------------------------------------------------
        if jnp.any(goal_mask):
            print(f"\n🎯 Goal reached at iteration {i}")
            break

    elapsed = time.perf_counter() - start_time

    print("\n===== RRT SUMMARY =====")
    print(f"Elapsed time: {elapsed:.3f} s")
    print(f"Best distance to goal: {float(best_dist):.4f}")
    print(f"Achieved at iteration: {best_iter}")
    print(f"Final tree size: {int(tree.tree_size)}")
