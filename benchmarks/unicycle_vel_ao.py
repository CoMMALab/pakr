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
import matplotlib.pyplot as plt
from params import Bounds, Position, MotionConstraints, PhysicsConstants, MJXparams, SSTparams, Callables

# ------------------------------------------------------------------
# UNICYCLE HELPERS
# ------------------------------------------------------------------

@jax.jit
def reached_goal_unicycle(states, goal, radius):
    pos = states[:, 0:2]
    goal_pos = jnp.array([goal.x, goal.y])
    diff2 = jnp.sum((pos - goal_pos)**2, axis=-1)
    return diff2 < radius**2

@partial(jax.jit, static_argnums=(0,))
def dist_unicycle(sim_params, diff):
    dq = diff[..., 0:2]
    dtheta = diff[..., 2]
    dtheta_norm = jnp.atan2(jnp.sin(dtheta), jnp.cos(dtheta))
    pos_cost = jnp.sum(dq**2, axis=-1)
    ang_cost = dtheta_norm**2
    w_pos, w_ang = 1.0, 0.3
    return (w_pos * pos_cost + w_ang * ang_cost).T

@partial(jax.jit, static_argnums=(0,))
def sample_actions_unicycle(sim_params, key):
    B = sim_params.batch_size
    tau = jax.random.uniform(
        key, (B, 2),
        minval=jnp.array([0.0, sim_params.motion_constraints.min_accel]), 
        maxval=jnp.array([sim_params.motion_constraints.max_vel, sim_params.motion_constraints.max_accel])
    )
    return tau

@partial(jax.jit, static_argnums=(0,))
def sample_unicycle(sim_params, key):
    B = sim_params.batch_size
    keys = jax.random.split(key, 2)
    pos = jax.random.uniform(keys[0], (B, 2), 
        minval=jnp.array([sim_params.bounds.min_x, sim_params.bounds.min_y]), 
        maxval=jnp.array([sim_params.bounds.max_x, sim_params.bounds.max_y]))
    theta = jax.random.uniform(keys[1], (B, 1), minval=-jnp.pi, maxval=jnp.pi)
    return jnp.concatenate([pos, theta], axis=-1)

@partial(jax.jit, static_argnums=(1))
def valid_unicycle(state, params, obstacles):
    x, y = state[:, 0], state[:, 1]
    within_bounds = (x >= params.bounds.min_x) & (x <= params.bounds.max_x) & \
                    (y >= params.bounds.min_y) & (y <= params.bounds.max_y)
    collision_free = helper.collision_check_2d(state[:, :2], obstacles)
    return within_bounds & collision_free

# Global references for NN Tiers
SIM_PARAMS_RESERVED = None
CALLABLES_RESERVED = None

def nn_tier_factory(size):
    def nn_fn(operands):
        states, tree_size, query = operands
        sliced_states = states[:size]
        parents, _ = helper.nearest_neighbor_masked(
            SIM_PARAMS_RESERVED, CALLABLES_RESERVED.dist_fn, 
            sliced_states, tree_size, query
        )
        return parents
    return nn_fn

TIERS = [512, 1024, 4096, 16384, 32768, 64536, 100000]
NN_BRANCHES = [nn_tier_factory(t) for t in TIERS]

@partial(jax.jit, static_argnums=(3, 4, 5))
def aorrt_iteration(tree, rng_key, obstacles, sst_params, sim_params, callables, best_cost):
    B = sim_params.batch_size
    A = 32
    K = B // A 
    rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)
    seed_pts = callables.sample_fn(sim_params, subkey1)[:K]

    branch_idx = jnp.digitize(tree.tree_size, jnp.array(TIERS))
    branch_idx = jnp.minimum(branch_idx, len(TIERS) - 1)

    operands = (tree.states, tree.tree_size, seed_pts)
    parents_small = jax.lax.switch(branch_idx, NN_BRANCHES, operands) 
    
    start_states = jnp.repeat(tree.states[parents_small], A, axis=0) 
    parents = jnp.repeat(parents_small, A, axis=0) 
    actions = callables.sampact_fn(sim_params, subkey2)

    states_end, valid_mask, dist_traveled = propagate.rollout_final_2d(
        start_states, actions, obstacles, sst_params, sim_params, callables
    )

    # Static padding for JIT
    valid_mask = valid_mask.at[-1].set(False)
    states_end = states_end.at[-1].set(jnp.zeros(sim_params.dims))
    actions = actions.at[-1].set(jnp.zeros(sim_params.action_dims))
    parents = parents.at[-1].set(-1)
    dist_traveled = dist_traveled.at[-1].set(0.0)

    # AO-Pruning: f(n) = g(n) + h(n)
    new_costs = tree.costs[parents] + 1
    goal_vec = jnp.array([sst_params.goal.x, sst_params.goal.y])
    h = jnp.linalg.norm(states_end[:, :2] - goal_vec, axis=-1)
    ao_mask = (new_costs + h <= best_cost) & valid_mask

    num_new = jnp.sum(ao_mask)
    ao_idx = jnp.nonzero(ao_mask, size=B, fill_value=-1)[0]

    tree, start_idx = rrtree.add_nodes(tree, states_end[ao_idx], actions[ao_idx], parents[ao_idx], new_costs[ao_idx], num_new)
    goal_mask = callables.goal_fn(states_end[ao_idx], sst_params.goal, sst_params.goal_radius)
    return tree, rng_key, goal_mask, jnp.sum(goal_mask), states_end[ao_idx], start_idx

@partial(jax.jit, static_argnums=(1, 2, 3))
def jit_while(tree, sst_params, sim_params, callables, obstacles, best_cost, i):
    def body_fn(carry):
        tree, key, goal_mask, goal, states, start_idx, iter = carry
        key, subkey = jax.random.split(key)
        tree, subkey, goal_mask, goal, states, start_idx = aorrt_iteration(
            tree, subkey, obstacles, sst_params, sim_params, callables, best_cost
        )
        return (tree, key, goal_mask, goal, states, start_idx, iter + 1)

    def cond_fn(carry):
        tree, key, goal_mask, goal, states, start_idx, iter = carry
        return (goal == 0) & (tree.tree_size < MAX_TREE_SIZE - sim_params.batch_size)

    init_carry = (tree, jax.random.PRNGKey(i), jnp.zeros(sim_params.batch_size, dtype=bool),
                  jnp.array(0, dtype=jnp.int32), jnp.zeros([sim_params.batch_size, sim_params.dims], dtype=jnp.float32),
                  jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32))

    tree, key, goal_mask, goal, states, start_idx, iter = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    return tree, key, goal_mask, goal, states, start_idx, iter, tree.tree_size

MAX_TREE_SIZE = 100000
if __name__ == "__main__":
    batch_size = 2048
    motion_constraints = MotionConstraints(max_vel=0.4, min_vel=0.0, max_accel=1.5, min_accel=-1.5)
    sim_params = MJXparams(motion_constraints=motion_constraints, physics_constants=PhysicsConstants(),
                           batch_size=batch_size, bounds=Bounds(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0, min_z=0.0, max_z=0.0),
                           dims=3, action_dims=2, dt=0.1, seed=42)
    sst_params = SSTparams(batch_size=batch_size, δBN=0.04, δs=0.02, decay=0.8,
                           start=Position(x=0.05, y=0.05, z=0.0), goal=Position(x=0.95, y=0.95, z=0.0),
                           goal_radius=0.05, time_to_evolve=10)
    callables = Callables(prop_fn=propagate.propagate_unicycle, valid_fn=valid_unicycle, sample_fn=sample_unicycle,
                          dist_fn=dist_unicycle, sampact_fn=sample_actions_unicycle, goal_fn=reached_goal_unicycle)
    
    obstacles = helper.get_obs('envs/tree2d.csv')
    SIM_PARAMS_RESERVED, CALLABLES_RESERVED = sim_params, callables

    print("\nStarting Unicycle AO-RRT - Warm-up...")
    dummy_tree = rrtree.KinoTree.init(MAX_TREE_SIZE, sim_params.dims, sim_params.action_dims)
    _ = jit_while(dummy_tree, sst_params, sim_params, callables, obstacles, jnp.inf, 0)
    print("Warm-up complete.")

    N_RUNS, MAX_TIME, COST_THRESHOLD = 100, 3.0, 5.0
    run_summaries = []

    for run_id in range(N_RUNS):
        gc.collect()
        tree = rrtree.KinoTree.init(MAX_TREE_SIZE, sim_params.dims, sim_params.action_dims)
        init_state = jnp.array([sst_params.start.x, sst_params.start.y, sst_params.start.z], dtype=jnp.float32)
        tree, _ = rrtree.add_nodes(tree, init_state, jnp.zeros(sim_params.action_dims), -1, 0.0, 1)
        
        best_cost, t0, ao_iter = float('inf'), time.perf_counter(), 0
        
        while True:
            rnd = np.random.randint(0, 2**31 - 1)
            tree, key, goal_mask, goal, states, start_idx, iters, size = jit_while(
                tree, sst_params, sim_params, callables, obstacles, jnp.array(best_cost, dtype=jnp.float32), rnd
            )

            if goal > 0:
                sol_cost = float(tree.costs[start_idx + jnp.argmax(goal_mask)])
                if sol_cost < best_cost: best_cost = sol_cost

            elapsed = time.perf_counter() - t0
            ao_iter += 1
            if elapsed >= MAX_TIME or best_cost <= COST_THRESHOLD or tree.tree_size >= MAX_TREE_SIZE - batch_size:
                run_summaries.append({"cost": best_cost if best_cost != float('inf') else None, "time": elapsed, "nodes": int(tree.tree_size), "iters": ao_iter})
                print(f"Run {run_id+1}: Cost: {best_cost:.4f} | Time: {elapsed:.2f}s | Nodes: {tree.tree_size}")
                break

    # Stats Calculation
    valid_costs = [s['cost'] for s in run_summaries if s['cost'] is not None]
    if valid_costs:
        print("\n" + "="*30 + f"\nGLOBAL STATS (N={N_RUNS})\n" + "="*30)
        print(f"Success Rate: {(len(valid_costs)/N_RUNS)*100:.1f}%")
        print(f"Avg Cost: {np.average(valid_costs):.4f} | Avg Time: {np.average([s['time'] for s in run_summaries]):.3f}s")
