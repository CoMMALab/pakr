import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
import time
import rrtree
import propagate
import helper
import gc
from params import Bounds, Position, MotionConstraints, PhysicsConstants, MJXparams, SSTparams, Callables

# ------------------------------------------------------------------
# ACROBOT DYNAMICS (DOUBLE PENDULUM)
# ------------------------------------------------------------------

@jax.jit
def propagate_acrobot(states, actions, dt, constants):
    """
    states:  (B, 4) -> [theta1, theta2, dtheta1, dtheta2]
    actions: (B, 1) -> [torque] applied to joint 2 (elbow)
    """
    l1, l2 = 1.0, 1.0
    m1, m2 = 1.0, 1.0
    lc1, lc2 = 0.5, 0.5
    I1, I2 = 0.2, 0.2
    g = 9.81

    theta1, theta2, dtheta1, dtheta2 = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
    u = actions[:, 0]

    d11 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * jnp.cos(theta2)) + I1 + I2
    d22 = m2 * lc2**2 + I2
    d12 = m2 * (lc2**2 + l1 * lc2 * jnp.cos(theta2)) + I2
    
    phi2 = m2 * lc2 * g * jnp.cos(theta1 + theta2 - jnp.pi/2)
    phi1 = -m2 * l1 * lc2 * dtheta2**2 * jnp.sin(theta2) \
           - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * jnp.sin(theta2) \
           + (m1 * lc1 + m2 * l1) * g * jnp.cos(theta1 - jnp.pi/2) + phi2

    accel2 = (u + d12 / d11 * phi1 - m2 * l1 * lc2 * dtheta1**2 * jnp.sin(theta2) - phi2) / (d22 - d12**2 / d11)
    accel1 = -(d12 * accel2 + phi1) / d11

    new_dtheta1 = dtheta1 + accel1 * dt
    new_dtheta2 = dtheta2 + accel2 * dt
    new_theta1 = (theta1 + new_dtheta1 * dt + jnp.pi) % (2 * jnp.pi) - jnp.pi
    new_theta2 = (theta2 + new_dtheta2 * dt + jnp.pi) % (2 * jnp.pi) - jnp.pi

    return jnp.stack([new_theta1, new_theta2, new_dtheta1, new_dtheta2], axis=-1)

@jax.jit
def dist_acrobot(sim_params, diff):
    dq = diff[..., 0:2]
    dv = diff[..., 2:4]
    return (jnp.sum(dq**2, axis=-1) + 0.1 * jnp.sum(dv**2, axis=-1)).T

@partial(jax.jit, static_argnums=(0,))
def sample_acrobot(sim_params, key):
    B = sim_params.batch_size
    k1, k2 = jax.random.split(key)
    angles = jax.random.uniform(k1, (B, 2), minval=-jnp.pi, maxval=jnp.pi)
    vels = jax.random.uniform(k2, (B, 2), minval=-5.0, maxval=5.0)
    return jnp.concatenate([angles, vels], axis=-1)

@partial(jax.jit, static_argnums=(0,))
def sample_actions_acrobot(sim_params, key):
    B = sim_params.batch_size
    limit = sim_params.motion_constraints.max_accel
    return jax.random.uniform(key, (B, 1), minval=-limit, maxval=limit)

@jax.jit
def reached_goal_acrobot(states, goal, radius):
    goal_angles = jnp.array([jnp.pi/2, 0.0])
    diff_angles = states[:, 0:2] - goal_angles
    diff_angles = (diff_angles + jnp.pi) % (2 * jnp.pi) - jnp.pi
    angle_dist_sq = jnp.sum(diff_angles**2, axis=-1)
    vel_dist_sq = jnp.sum(states[:, 2:4]**2, axis=-1)
    return (angle_dist_sq + 0.1 * vel_dist_sq) < radius**2

@partial(jax.jit, static_argnums=(1))
def valid_acrobot(state, params, obstacles):
    v = state[:, 2:4]
    vel_ok = jnp.all(jnp.abs(v) < 5.0, axis=-1)
    return vel_ok

# ------------------------------------------------------------------
# AO-RRT LOGIC
# ------------------------------------------------------------------

MAX_TREE_SIZE = 500000
TIERS = [512, 1024, 4096, 16384, 32768, 64536, 100000, 250000, 500000]
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

    # AO-Pruning: f(n) = g(n) + h(n)
    # admissible heuristic for Acrobot: angular distance to vertical
    new_costs = tree.costs[parents] + 0.15
    goal_angles = jnp.array([jnp.pi/2, 0.0])
    diff_angles = (states_end[:, 0:2] - goal_angles + jnp.pi) % (2 * jnp.pi) - jnp.pi
    h = jnp.linalg.norm(diff_angles, axis=-1) 
    
    ao_mask = (new_costs + h <= best_cost) & valid_mask
    ao_mask = ao_mask.at[-1].set(False)

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

# ------------------------------------------------------------------
# EXECUTION
# ------------------------------------------------------------------

if __name__ == "__main__":
    batch_size = 4096
    A = 32
    sim_params = MJXparams(
        motion_constraints=MotionConstraints(max_accel=10.0, min_accel=-10.0),
        physics_constants=PhysicsConstants(),
        batch_size=batch_size,
        bounds=Bounds(min_x=-jnp.pi, max_x=jnp.pi, min_y=-jnp.pi, max_y=jnp.pi),
        dims=4, action_dims=1, dt=0.05, seed=42
    )

    sst_params = SSTparams(
        batch_size=batch_size, δBN=0.1, δs=0.05, decay=0.8,
        start=Position(x=-jnp.pi/2, y=0.0, z=0.0), 
        goal=Position(x=jnp.pi/2, y=0.0, z=0.0),    
        goal_radius=0.2, time_to_evolve=3,
    )

    callables = Callables(
        prop_fn=propagate_acrobot, valid_fn=valid_acrobot, sample_fn=sample_acrobot,
        dist_fn=dist_acrobot, sampact_fn=sample_actions_acrobot, goal_fn=reached_goal_acrobot
    )

    SIM_PARAMS_RESERVED, CALLABLES_RESERVED = sim_params, callables
    obstacles = jnp.array([])

    print("\nStarting Acrobot AO-RRT - Warm-up...")
    dummy_tree = rrtree.KinoTree.init(MAX_TREE_SIZE, sim_params.dims, sim_params.action_dims)
    _ = jit_while(dummy_tree, sst_params, sim_params, callables, obstacles, jnp.inf, 0)
    print("Warm-up complete.")

    N_RUNS, MAX_TIME, COST_THRESHOLD = 100, 5.0, 10
    run_summaries = []

    for run_id in range(N_RUNS):
        gc.collect()
        tree = rrtree.KinoTree.init(MAX_TREE_SIZE, sim_params.dims, sim_params.action_dims)
        init_state = jnp.array([sst_params.start.x, sst_params.start.y, 0.0, 0.0], dtype=jnp.float32)
        tree, _ = rrtree.add_nodes(tree, init_state, jnp.zeros(sim_params.action_dims), -1, 0.0, 1)
        
        best_cost, t0, ao_iter = float('inf'), time.perf_counter(), 0
        
        while True:
            rnd = np.random.randint(0, 2**31 - 1)
            tree, key, goal_mask, goal, states, start_idx, iters, size = jit_while(
                tree, sst_params, sim_params, callables, obstacles, jnp.array(best_cost, dtype=jnp.float32), rnd
            )

            if goal > 0:
                sol_cost = float(tree.costs[start_idx + jnp.argmax(goal_mask)])
                if sol_cost < best_cost: 
                    best_cost = sol_cost

            elapsed = time.perf_counter() - t0
            ao_iter += 1
            
            # Termination: Time limit, Cost threshold, or Tree Full
            if elapsed >= MAX_TIME or best_cost <= COST_THRESHOLD or tree.tree_size >= MAX_TREE_SIZE - batch_size:
                run_summaries.append({
                    "cost": best_cost if best_cost != float('inf') else None, 
                    "time": elapsed, 
                    "nodes": int(tree.tree_size), 
                    "iters": ao_iter
                })
                print(f"Run {run_id+1}: Cost: {best_cost:.4f} | Time: {elapsed:.2f}s | Nodes: {tree.tree_size}")
                break

    # Stats Calculation
    valid_costs = [s['cost'] for s in run_summaries if s['cost'] is not None]
    if valid_costs:
        print("\n" + "="*30 + f"\nGLOBAL STATS (N={N_RUNS})\n" + "="*30)
        print(f"Success Rate: {(len(valid_costs)/N_RUNS)*100:.1f}%")
        print(f"Avg Cost: {np.average(valid_costs):.4f} | Avg Time: {np.average([s['time'] for s in run_summaries]):.3f}s")