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

@partial(jax.jit, static_argnums=(3, 4, 5))
def aorrt_iteration(tree, rng_key, obstacles, sst_params, sim_params, callables, best_cost):
    rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)

    # 1. Sample + NN
    seed_pts = callables.sample_fn(sim_params, subkey1)
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
def jit_while(tree, sst_params, sim_params, callables, obstacles, best_cost, i):

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
            best_cost
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

    
    


    obstacles = helper.get_obs(args.env)
    
    # ------------------------------------------------------------
    # AO-RRT driver (single instance, 10s anytime run)
    # ------------------------------------------------------------
    import matplotlib.pyplot as plt

    MAX_TIME = 10.0  # seconds
    MAX_TREE_SIZE = 70000

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

    _ = jit_while(
        tree,
        sst_params,
        sim_params,
        callables,
        obstacles,
        best_cost,
        0,
    )

    # Force completion before timing
    jax.block_until_ready(_)

    print("Warm-up complete.")

    # ------------------------------------------------------------
    # AO-RRT anytime loop
    # ------------------------------------------------------------
    print("\nStarting AO-RRT...\n")

    t0 = time.perf_counter()
    ao_iter = 0

    while True:
        elapsed = time.perf_counter() - t0
        if elapsed >= MAX_TIME:
            break

        tree, key, goal_mask, goal, states, start_idx, iters, size = jit_while(
            tree,
            sst_params,
            sim_params,
            callables,
            obstacles,
            best_cost,
            ao_iter,
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
                    f"tree size={int(size)}"
                )

        ao_iter += 1

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



