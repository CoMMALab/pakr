import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
import time
import gc
import rrtree
import propagate
import helper
from params import Bounds, Position, MotionConstraints, PhysicsConstants, MJXparams, SSTparams, Callables

# ------------------------------------------------------------------
# DYNAMIC UNICYCLE HELPERS (FORCE CONTROL)
# ------------------------------------------------------------------
@jax.jit
def reached_goal_unicycle(states, goal, radius):
    pos = states[:, 0:2]
    goal_pos = jnp.array([goal.x, goal.y])
    diff2 = jnp.sum((pos - goal_pos)**2, axis=-1)
    return diff2 < radius**2

@partial(jax.jit, static_argnums=(0,))
def sample_actions_unicycle(sim_params, key):
    B = sim_params.batch_size
    tau = jax.random.uniform(
        key, (B, 2),
        minval=jnp.array([sim_params.motion_constraints.min_accel, sim_params.motion_constraints.min_accel]), 
        maxval=jnp.array([sim_params.motion_constraints.max_accel, sim_params.motion_constraints.max_accel])
    )
    return tau

@jax.jit
def propagate_unicycle_dynamic(states, actions, dt, constants):
    x, y, theta, v, omega = states[:, 0], states[:, 1], states[:, 2], states[:, 3], states[:, 4]
    a, alpha = actions[:, 0], actions[:, 1]
    new_v = v + a * dt
    new_omega = omega + alpha * dt
    mid_v = v + 0.5 * a * dt
    mid_omega = omega + 0.5 * alpha * dt
    mid_theta = theta + 0.5 * mid_omega * dt
    new_x = x + mid_v * jnp.cos(mid_theta) * dt
    new_y = y + mid_v * jnp.sin(mid_theta) * dt
    new_theta = theta + new_omega * dt
    return jnp.stack([new_x, new_y, new_theta, new_v, new_omega], axis=-1)

@partial(jax.jit, static_argnums=(0,))
def dist_unicycle_dynamic(sim_params, diff):
    dq = diff[..., 0:2]
    dtheta = diff[..., 2]
    dvels = diff[..., 3:5]
    dtheta_norm = jnp.atan2(jnp.sin(dtheta), jnp.cos(dtheta))
    pos_cost = jnp.sum(dq**2, axis=-1)
    ang_cost = dtheta_norm**2
    vel_cost = jnp.sum(dvels**2, axis=-1)
    w_pos, w_ang, w_vel = 1.0, 0.3, 0.1
    return (w_pos * pos_cost + w_ang * ang_cost + w_vel * vel_cost).T

@partial(jax.jit, static_argnums=(0,))
def sample_unicycle_dynamic(sim_params, key):
    B = sim_params.batch_size
    k1, k2, k3 = jax.random.split(key, 3)
    pos = jax.random.uniform(k1, (B, 2), 
        minval=jnp.array([sim_params.bounds.min_x, sim_params.bounds.min_y]), 
        maxval=jnp.array([sim_params.bounds.max_x, sim_params.bounds.max_y]))
    theta = jax.random.uniform(k2, (B, 1), minval=-jnp.pi, maxval=jnp.pi)
    vels = jax.random.uniform(k3, (B, 2), 
        minval=jnp.array([sim_params.motion_constraints.min_vel, -1.0]), 
        maxval=jnp.array([sim_params.motion_constraints.max_vel, 1.0]))
    return jnp.concatenate([pos, theta, vels], axis=-1)

@partial(jax.jit, static_argnums=(1))
def valid_unicycle_dynamic(state, params, obstacles):
    x, y, v = state[:, 0], state[:, 1], state[:, 3]
    within_bounds = (x >= params.bounds.min_x) & (x <= params.bounds.max_x) & \
                    (y >= params.bounds.min_y) & (y <= params.bounds.max_y)
    vel_ok = (v >= params.motion_constraints.min_vel) & (v <= params.motion_constraints.max_vel)
    collision_free = helper.collision_check_2d(state[:, :2], obstacles)
    return within_bounds & collision_free & vel_ok

# ------------------------------------------------------------------
# AO-PLANNER SETUP
# ------------------------------------------------------------------

MAX_TREE_SIZE = 100000
TIERS = [512, 1024, 4096, 16384, 32768, 64536, 100000]
SIM_PARAMS_RESERVED = None
CALLABLES_RESERVED = None

def nn_tier_factory(size):
    def nn_fn(operands):
        states, tree_size, query = operands
        return helper.nearest_neighbor_masked(
            SIM_PARAMS_RESERVED, CALLABLES_RESERVED.dist_fn, 
            states[:size], tree_size, query
        )[0]
    return nn_fn

NN_BRANCHES = [nn_tier_factory(t) for t in TIERS]

@partial(jax.jit, static_argnums=(3, 4, 5))
def aorrt_iteration(tree, rng_key, obstacles, sst_params, sim_params, callables, best_cost):
    B, A = sim_params.batch_size, 32
    K = B // A 
    rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)
    
    seed_pts = callables.sample_fn(sim_params, subkey1)[:K]
    branch_idx = jnp.minimum(jnp.digitize(tree.tree_size, jnp.array(TIERS)), len(TIERS) - 1)
    parents_small = jax.lax.switch(branch_idx, NN_BRANCHES, (tree.states, tree.tree_size, seed_pts)) 
    
    start_states = jnp.repeat(tree.states[parents_small], A, axis=0) 
    parents = jnp.repeat(parents_small, A, axis=0) 
    actions = callables.sampact_fn(sim_params, subkey2)

    states_end, valid_mask, dist_traveled = propagate.rollout_final_2d(
        start_states, actions, obstacles, sst_params, sim_params, callables
    )

    # Static padding for JIT consistency
    valid_mask = valid_mask.at[-1].set(False)
    
    # AO-Pruning logic: f(n) = g(n) + h(n)
    new_costs = tree.costs[parents] + 1
    goal_vec = jnp.array([sst_params.goal.x, sst_params.goal.y])
    # Euclidean distance as admissible heuristic for position
    h = jnp.linalg.norm(states_end[:, :2] - goal_vec, axis=-1)
    
    ao_mask = (new_costs + h <= best_cost) & valid_mask
    num_new = jnp.sum(ao_mask)
    ao_idx = jnp.nonzero(ao_mask, size=B, fill_value=-1)[0]

    tree, start_idx = rrtree.add_nodes(
        tree, states_end[ao_idx], actions[ao_idx], 
        parents[ao_idx], new_costs[ao_idx], num_new
    )
    
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

def extract_sol(tree, goal_mask, start_idx):
    if jnp.sum(goal_mask) == 0: return None, None
    goal_idxs = jnp.argmax(goal_mask)
    goal_idx = goal_idxs + start_idx
    path, actions = [], []
    while goal_idx != -1:
        path.append(tree.states[goal_idx])
        actions.append(tree.actions[goal_idx])
        goal_idx = tree.parents[goal_idx]
    return jnp.array(path[::-1]), jnp.array(actions[::-1])

# ------------------------------------------------------------------
# EXECUTION
# ------------------------------------------------------------------

if __name__ == "__main__":
    batch_size = 2048
    motion_constraints = MotionConstraints(
        max_vel = 0.6, min_vel = -0.1, 
        max_accel = 1.0, min_accel = -1.0
    )
    sim_params = MJXparams(
        motion_constraints=motion_constraints, physics_constants=PhysicsConstants(),
        batch_size=batch_size, bounds = Bounds(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0, min_z=0.0, max_z=0.0),
        dims=5, action_dims=2, dt = 0.1, seed = 42
    )
    sst_params = SSTparams(
        batch_size=batch_size, δBN=0.04, δs=0.02, decay=0.8,
        start=Position(x=0.05, y=0.05, z=0.0), goal=Position(x=0.95, y=0.95, z=0.0),
        goal_radius=0.05, time_to_evolve=10,
    )
    callables = Callables(
        prop_fn=propagate_unicycle_dynamic, valid_fn=valid_unicycle_dynamic,
        sample_fn=sample_unicycle_dynamic, dist_fn=dist_unicycle_dynamic,
        sampact_fn=sample_actions_unicycle, goal_fn=reached_goal_unicycle
    )

    SIM_PARAMS_RESERVED, CALLABLES_RESERVED = sim_params, callables
    obstacles = helper.get_obs('envs/tree2d.csv')

    print("\nStarting Dynamic Unicycle AO-RRT - Pre-compiling...")
    dummy_tree = rrtree.KinoTree.init(MAX_TREE_SIZE, sim_params.dims, sim_params.action_dims)
    _ = jit_while(dummy_tree, sst_params, sim_params, callables, obstacles, jnp.inf, 0)
    print("Compilation complete.\n")

    N_RUNS, MAX_TIME, COST_THRESHOLD = 100, 5.0, 6
    run_summaries = []

    for i in range(N_RUNS):
        gc.collect()
        tree = rrtree.KinoTree.init(MAX_TREE_SIZE, sim_params.dims, sim_params.action_dims)
        init_state = jnp.array([sst_params.start.x, sst_params.start.y, 0.0, 0.0, 0.0])
        tree, _ = rrtree.add_nodes(tree, init_state, jnp.zeros(2), -1, 0.0, 1)
        
        best_cost, t0, ao_iter = float('inf'), time.perf_counter(), 0
        
        while True:
            rnd = np.random.randint(0, 2**31 - 1)
            # Pass best_cost as a JAX array to prevent recompilation
            result = jit_while(tree, sst_params, sim_params, callables, obstacles, jnp.array(best_cost), rnd)
            tree, key, goal_mask, goal_found, states, start_idx, iter_val, size = jax.block_until_ready(result)
            
            if goal_found > 0:
                sol_cost = float(tree.costs[start_idx + jnp.argmax(goal_mask)])
                if sol_cost < best_cost:
                    best_cost = sol_cost

            elapsed = time.perf_counter() - t0
            ao_iter += 1
            
            if elapsed >= MAX_TIME or best_cost <= COST_THRESHOLD or tree.tree_size >= MAX_TREE_SIZE - batch_size:
                run_summaries.append({"cost": best_cost, "time": elapsed, "nodes": int(tree.tree_size)})
                print(f"Run {i:02d}: Final Cost: {best_cost:.4f} | Time: {elapsed:.2f}s | Nodes: {tree.tree_size}")
                break

    # Final Statistics
    costs = np.array([s['cost'] for s in run_summaries if s['cost'] < float('inf')])
    if len(costs) > 0:
        print("\n" + "="*30)
        print(f"Average Best Cost: {np.mean(costs):.4f}")
        print(f"Average Tree Size: {np.mean([s['nodes'] for s in run_summaries]):.0f}")
        print("="*30)