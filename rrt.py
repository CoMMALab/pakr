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
    parents, _ = helper.nearest_neighbor_masked(
        sim_params, callables.dist_fn, tree.states, tree.tree_size, seed_pts
    )
    start_states = tree.states[parents]

    # ------------------------------------------------------------------
    # 2. Sample actions and rollout
    # ------------------------------------------------------------------
    actions = callables.sampact_fn(sim_params, subkey2)

    states_end, valid_mask, dist_traveled = propagate.rollout_final(
        start_states, actions, obstacles, sst_params, sim_params, callables
    )

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
    goal_mask = helper.reached_goal(
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

def extract_sol(tree, goal_mask, start_idx):
    if jnp.sum(goal_mask) == 0:
        print("error: no goal reached")
        return None, None
    
    print(jnp.sum(goal_mask))

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
    for i in range(len(actions)):
        start = path[i][None, :]    # shape: (1, state_dim)
        action = actions[i][None, :]
        states_end, valid_mask, _ = propagate.rollout_final(
            start, action, obstacles, sst_params, sim_params, callables
        )
        if not valid_mask[0]:
            return False
    return True


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
    
    # dummy_tree = rrtree.KinoTree.init(max_size=MAX_TREE_SIZE, state_dim=sim_params.dims, action_dim=sim_params.action_dims, frontier_size=sim_params.batch_size*2)
    # dummy_tree = jax.device_put(dummy_tree)
    # #jax.config.update("jax_log_compiles", True)
    # init_carry = (dummy_tree, jax.random.PRNGKey(0), jnp.zeros(sim_params.batch_size, dtype=bool), 0, jnp.zeros([sim_params.batch_size, sim_params.dims], dtype=jnp.float32), 0, 0)

    # tree, key, goal_mask, goal, states, start_idx, iter = jax.lax.while_loop(dummy_cond_fn, body_fn, init_carry)
    # dummy_tree = None
    # gc.collect()

    tree = rrtree.KinoTree.init(max_size=MAX_TREE_SIZE, state_dim=sim_params.dims, action_dim=sim_params.action_dims)
    tree = jax.device_put(tree)
    init = jnp.concatenate([jnp.asarray([sst_params.start.x, sst_params.start.y, sst_params.start.z]), jnp.zeros(sim_params.dims - 3, dtype=jnp.float32)], axis=0)
    controls = jnp.zeros(sim_params.action_dims)
    tree, _ = rrtree.add_nodes(tree, init, controls, -1, 0.0, 1)

    # key = jax.random.PRNGKey(0)
    # start = time.time()
    # tree, key = rrt(tree, key, obstacles, sst_params, sim_params, callables, num_iters=100)
    # print('Total RRT time for 5 iterations:', time.time() - start)

    goal = 0
    i = 0
    print("\n\n start rrt \n\n")
    start_p = time.perf_counter()

    jit_while(tree, sst_params, sim_params, callables, obstacles, 0)


    # testing
    times = []
    iters = []
    sizes = []
    costs = []
    for i in range(10):
        gc.collect

        tree = rrtree.KinoTree.init(max_size=MAX_TREE_SIZE, state_dim=sim_params.dims, action_dim=sim_params.action_dims)
        tree = jax.device_put(tree)
        init = jnp.concatenate([jnp.asarray([sst_params.start.x, sst_params.start.y, sst_params.start.z]), jnp.zeros(sim_params.dims - 3, dtype=jnp.float32)], axis=0)
        controls = jnp.zeros(sim_params.action_dims)
        tree, _ = rrtree.add_nodes(tree, init, controls, -1, 0.0, 1)
        # key = jax.random.PRNGKey(0)
        # start = time.time()
        # tree, key = rrt(tree, key, obstacles, sst_params, sim_params, callables, num_iters=100)
        # print('Total RRT time for 5 iterations:', time.time() - start)

        goal = 0
        #print("\n\n start rrt \n\n")
        start_p = time.perf_counter()

        # Initialize
        tree, key, goal_mask, goal, states, start_idx, iter, size = jit_while(tree, sst_params, sim_params, callables, obstacles, i)
        timer = time.perf_counter() - start_p


        path, actions = extract_sol(tree, goal_mask, start_idx)
        print(path)
        print(actions)
        is_valid = verify_sol(path, actions, obstacles, sst_params, sim_params, callables)
        print(is_valid)
        cost = tree.costs[jnp.argmax(goal_mask) + start_idx]
        costs.append(cost)
        times.append(timer)
        iters.append(iter)
        sizes.append(size)
        print(f"Found goal after {iter} iterations, time: {(timer)*1e3:.3f} ms, tree size: {size}")

    times = jnp.array(times)
    iters = jnp.array(iters)
    sizes = jnp.array(sizes)
    costs = jnp.array(costs)
    print(goal.dtype)
    print(f"Average time over 100 runs: {jnp.mean(times)*1e3:.3f} ms, {jnp.mean(iters):.2f} iterations, size {jnp.mean(sizes):.2f}")
    print(f"min time over 100 runs: {jnp.min(times)*1e3:.3f} ms, {jnp.min(iters)} iterations, min size {jnp.min(sizes)}")
    print(f"max time over 100 runs: {jnp.max(times)*1e3:.3f} ms, {jnp.max(iters)} iterations, max size {jnp.max(sizes)}")
    print(f"Average cost over 100 runs: {jnp.mean(costs):.3f}, min cost: {jnp.min(costs):.3f}, max cost: {jnp.max(costs):.3f}")

    
