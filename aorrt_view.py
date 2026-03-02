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
import rrt_view

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
def aorrt_iteration(tree, rng_key, obstacles, sst_params, sim_params, callables, best_cost):
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
    seed_pts = callables.sample_fn(sim_params, subkey1)[:K]

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
    parser.add_argument('--env', type=str, default='envs/narrow.csv', help='Path to the environment config file.')
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

    # 1. Warm up the inner iteration function directly
    # This ensures every branch of the logic is traced.
    _ = aorrt_iteration(
        tree,
        jax.random.PRNGKey(0),
        obstacles,
        sst_params,
        sim_params,
        callables,
        jnp.inf
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
    N_RUNS = 10
    MAX_TIME = 3.0       
    COST_THRESHOLD = 1.7
    GT_MIN_COST = 1.403  
    all_trajectories = [] 

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
                jnp.array(best_cost, dtype=jnp.float32), rnd
            )

            if goal > 0:
                idx = jnp.argmax(goal_mask)
                sol_cost = float(tree.costs[start_idx + idx])
                if sol_cost < best_cost:
                    best_cost = sol_cost

            elapsed = time.perf_counter() - t0

            
            ao_iter += 1
            # Quick print for progress
            # if ao_iter % 5 == 0: # Print every 5 iters to keep console clean
            #     print(f"  Iter {ao_iter:02d} | Cost: {best_cost:.4f} | Nodes: {size}")
            # Exit conditions
            if elapsed >= MAX_TIME or best_cost <= COST_THRESHOLD:
                # Capture the final state of this run before breaking
                path_nodes, actions = rrt_view.extract_sol(tree, goal_mask, start_idx)
        
                if actions is not None:
                    traj = rrt_view.rollout_full_trajectory(path_nodes[0], actions, sst_params, sim_params, callables)
                    all_trajectories.append(traj)
                    print(f"Run {run_id}: Path found and reconstructed.")
                else:
                    print(f"Run {run_id}: No path found.")

                print(f"Run {run_id+1} Finished | Final Cost: {best_cost:.4f} | Time: {elapsed:.2f}s | iters: {ao_iter} | nodes: {tree.tree_size}")
                break

            tree = rrtree.KinoTree.init(max_size=MAX_TREE_SIZE, state_dim=sim_params.dims, action_dim=sim_params.action_dims)
            tree = jax.device_put(tree)
            tree, _ = rrtree.add_nodes(tree, init_state, jnp.zeros(sim_params.action_dims), -1, 0.0, 1)
            

    if all_trajectories:
        rrt_view.visualize_multi_trajectories(args.env, all_trajectories, sst_params, "./aosolution2.html")
    else:
        print("No successful runs found to visualize.")