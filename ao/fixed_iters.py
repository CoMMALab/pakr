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
    COST_THRESHOLD = 1.55
    GT_MIN_COST = 1.403  
    all_run_data = []
    AO_ITERS = 3  # Hardcoded as requested
    # ... other params ...

    def run_single_ao_rrt(key, unused_rng):
        """Encapsulates a single run of 3 AO-RRT iterations."""
        # Initialize Tree
        init_state = jnp.concatenate([
            jnp.asarray([sst_params.start.x, sst_params.start.y, sst_params.start.z]), 
            jnp.zeros(sim_params.dims - 3, dtype=jnp.float32)
        ], axis=0)
        
        tree = rrtree.KinoTree.init(max_size=MAX_TREE_SIZE, state_dim=sim_params.dims, action_dim=sim_params.action_dims)
        tree, _ = rrtree.add_nodes(tree, init_state, jnp.zeros(sim_params.action_dims), -1, 0.0, 1)

        # Initial state for the fori_loop: (tree, best_cost, run_costs, run_times)
        # We track costs and times for each of the 3 iterations
        initial_val = (tree, jnp.inf, jnp.zeros(AO_ITERS), jnp.zeros(AO_ITERS), time.perf_counter())

        def ao_step(i, state):
            tree, best_cost, costs, times, start_time = state
            
            # Use a deterministic-ish seed for JIT
            rnd = jax.random.PRNGKey(key * 1000 + i)[0] 
            
            # Call your JIT function
            tree, _, goal_mask, goal, _, start_idx, _, _ = jit_while(
                tree, sst_params, sim_params, callables, obstacles, best_cost, rnd
            )

            # Update best cost if a goal was found
            # Note: In JAX we use select/where instead of if-statements for better JIT
            sol_cost = tree.costs[start_idx + jnp.argmax(goal_mask)]
            new_best = jnp.where((goal > 0) & (sol_cost < best_cost), sol_cost, best_cost)
            
            # Log data
            costs = costs.at[i].set(new_best)
            times = times.at[i].set(time.perf_counter() - start_time)
            
            return tree, new_best, costs, times, start_time

        # Run the 3 iterations via JAX loop
        _, final_best, all_costs, all_times, _ = lax.fori_loop(0, AO_ITERS, ao_step, initial_val)
        
        return all_costs, all_times, final_best

    # --- Execute All Runs ---
    print(f"\nStarting {N_RUNS} runs with {AO_ITERS} iters each...")
    # --- Configuration ---

    run_summaries = []

    print(f"\nStarting {N_RUNS} runs (AO-RRT with {AO_ITERS} internal iterations)...\n")

    for run_id in range(N_RUNS):
        t_start_run = time.perf_counter()
        
        # 1. Execute the 3-iteration JAX loop for this specific run
        # Note: run_single_ao_rrt returns (all_costs, all_times, final_best)
        rnd = np.random.randint(0, 100000)  # Random seed for this run
        iter_costs, iter_times, final_best = run_single_ao_rrt(rnd, None)
        
        # Ensure values are moved from GPU/TPU to CPU for logging
        iter_costs = np.array(iter_costs)
        iter_times = np.array(iter_times)
        final_best = float(final_best)
        elapsed_run = time.perf_counter() - t_start_run

        # 2. Log data for every iteration (for the convergence plot)
        for i in range(AO_ITERS):
            all_run_data.append({
                "Run": run_id,
                "Iteration": i,
                "Best Cost": iter_costs[i] if iter_costs[i] < float('inf') else None,
                "Cumulative Time": iter_times[i]
            })

        # 3. Log summary for this specific run (for global stats)
        run_info = {
            "cost": final_best if final_best < float('inf') else None,
            "time": elapsed_run,
            "success": final_best <= COST_THRESHOLD
        }
        run_summaries.append(run_info)

        # 4. Immediate feedback to terminal
        status = "SUCCESS" if run_info["success"] else "FAIL"
        print(f"Run {run_id+1:03d}/{N_RUNS} | {status} | Final Cost: {final_best:.4f} | Time: {elapsed_run:.3f}s")
        
        # Clean up memory every run
        gc.collect()

    # Unpack results into numpy arrays [N_RUNS, AO_ITERS]
    all_costs_matrix = np.array([r[0] for r in results])
    all_times_matrix = np.array([r[1] for r in results])
            

    # --- Process Final Statistics ---
    valid_costs = [s['cost'] for s in run_summaries if s['cost'] is not None]
    success_count = sum(1 for s in run_summaries if s['success'])

    if valid_costs:
        avg_cost = np.mean(valid_costs)
        std_cost = np.std(valid_costs)
        avg_time = np.mean([s['time'] for s in run_summaries])
        
        print("\n" + "="*40)
        print(f"{'GLOBAL STATISTICS (N=' + str(N_RUNS) + ')':^40}")
        print("="*40)
        print(f"Success Rate:      {(success_count/N_RUNS)*100:>10.1f}%")
        print(f"Avg Best Cost:     {avg_cost:>10.4f} (±{std_cost:.3f})")
        print(f"Avg Total Time:    {avg_time:>10.3f}s")
        print(f"Goal Threshold:    {COST_THRESHOLD:>10.4f}")
        print(f"GT Min Cost:       {GT_MIN_COST:>10.4f}")
        print("="*40)
    else:
        print("\nNo valid paths found in any run.")

        # Create a simple long-form DataFrame
    data_list = []
    for run_id in range(N_RUNS):
        for iter_id in range(AO_ITERS):
            cost = all_costs_matrix[run_id, iter_id]
            data_list.append({
                "Run": run_id,
                "Iteration": iter_id,
                "Best Cost": cost if cost != float('inf') else None,
                "Cumulative Time": all_times_matrix[run_id, iter_id]
            })

    df_final = pd.DataFrame(data_list)

    # --- Visualization ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Graph 1: Convergence (Mean + Std Dev)
    sns.lineplot(data=df_final, x="Iteration", y="Best Cost", ax=ax1, 
                color="dodgerblue", errorbar='sd', marker='o')
    ax1.axhline(y=GT_MIN_COST, color='red', linestyle='--', label='Ground Truth')
    ax1.set_xticks(range(AO_ITERS))
    ax1.set_title(f"Cost Convergence over {AO_ITERS} Fixed Iters")

    # Graph 2: Timing
    sns.lineplot(data=df_final, x="Iteration", y="Cumulative Time", ax=ax2, 
                color="forestgreen", errorbar='sd', marker='o')
    ax2.set_xticks(range(AO_ITERS))
    ax2.set_title("Runtime per Iteration Step")

    plt.tight_layout()
    plt.savefig("ao/fixed_iters_convergence.png")




