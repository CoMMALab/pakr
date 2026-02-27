import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
import time
import rrtree
import propagate
import helper
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
    # Standard Acrobot parameters
    l1, l2 = 1.0, 1.0
    m1, m2 = 1.0, 1.0
    lc1, lc2 = 0.5, 0.5
    I1, I2 = 0.2, 0.2
    g = 9.81

    theta1, theta2, dtheta1, dtheta2 = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
    u = actions[:, 0]

    # Mass Matrix components
    d11 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * jnp.cos(theta2)) + I1 + I2
    d22 = m2 * lc2**2 + I2
    d12 = m2 * (lc2**2 + l1 * lc2 * jnp.cos(theta2)) + I2
    
    # Coriolis and Gravity
    phi2 = m2 * lc2 * g * jnp.cos(theta1 + theta2 - jnp.pi/2)
    phi1 = -m2 * l1 * lc2 * dtheta2**2 * jnp.sin(theta2) \
           - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * jnp.sin(theta2) \
           + (m1 * lc1 + m2 * l1) * g * jnp.cos(theta1 - jnp.pi/2) + phi2

    # Accelerations
    accel2 = (u + d12 / d11 * phi1 - m2 * l1 * lc2 * dtheta1**2 * jnp.sin(theta2) - phi2) / (d22 - d12**2 / d11)
    accel1 = -(d12 * accel2 + phi1) / d11

    # Integration
    new_dtheta1 = dtheta1 + accel1 * dt
    new_dtheta2 = dtheta2 + accel2 * dt
    new_theta1 = (theta1 + new_dtheta1 * dt + jnp.pi) % (2 * jnp.pi) - jnp.pi
    new_theta2 = (theta2 + new_dtheta2 * dt + jnp.pi) % (2 * jnp.pi) - jnp.pi

    return jnp.stack([new_theta1, new_theta2, new_dtheta1, new_dtheta2], axis=-1)

@jax.jit
def dist_acrobot(sim_params, diff):
    # diff: [dtheta1, dtheta2, ddtheta1, ddtheta2]
    dq = diff[..., 0:2]
    dv = diff[..., 2:4]
    # Simple weighted Euclidean
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
    # Goal: Vertically upright [pi/2, 0, 0, 0]
    goal_state = jnp.array([jnp.pi/2, 0.0, 0.0, 0.0])
    diff = states - goal_state
    # Angle wrapping for distance
    diff = diff.at[:, 0:2].set((diff[:, 0:2] + jnp.pi) % (2 * jnp.pi) - jnp.pi)
    return jnp.sum(diff**2, axis=-1) < radius**2

@partial(jax.jit, static_argnums=(1))
def valid_acrobot(state, params, obstacles):
    # Basic velocity clamping
    v = state[:, 2:4]
    vel_ok = jnp.all(jnp.abs(v) < 15.0, axis=-1)
    return vel_ok

# ------------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------------

def extract_sol(tree, goal_mask, start_idx):
    if jnp.sum(goal_mask) == 0:
        return None, None
    goal_idx = jnp.argmax(goal_mask) + start_idx
    path, actions = [], []
    while goal_idx != -1:
        path.append(tree.states[goal_idx])
        actions.append(tree.actions[goal_idx])
        goal_idx = tree.parents[goal_idx]
    return jnp.array(path[::-1]), jnp.array(actions[::-1])

def verify_sol(path, actions, obstacles, sst_params, sim_params, callables):
    for i in range(len(actions)-1):
        start = path[i][None, :]
        action = actions[i+1][None, :]
        states_end, valid_mask, _ = propagate.rollout_final_2d(
            start, action, obstacles, sst_params, sim_params, callables
        )
        if not valid_mask: return False
    return True

# ------------------------------------------------------------------
# PLANNER SETUP
# ------------------------------------------------------------------

MAX_TREE_SIZE = 50000
TIERS = [512, 1024, 4096, 16384, 32768, 50000]
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
def rrt_iteration(tree, rng_key, obstacles, sst_params, sim_params, callables):
    B, A = sim_params.batch_size, 16 
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

    valid_mask = valid_mask.at[-1].set(False)
    num_new = jnp.sum(valid_mask)
    valid_idx = jnp.nonzero(valid_mask, size=B, fill_value=-1)[0]

    tree, start_idx = rrtree.add_nodes(
        tree, states_end[valid_idx], actions[valid_idx], 
        parents[valid_idx], tree.costs[parents[valid_idx]] + dist_traveled[valid_idx], num_new
    )
    goal_mask = callables.goal_fn(states_end[valid_idx], sst_params.goal, sst_params.goal_radius)
    return tree, rng_key, goal_mask, jnp.sum(goal_mask), states_end[valid_idx], start_idx

@partial(jax.jit, static_argnums=(1, 2, 3))
def jit_while(tree, sst_params, sim_params, callables, obstacles, i):
    def body_fn(carry):
        tree, key, goal_mask, goal, states, start_idx, iter = carry
        tree, key, goal_mask, goal, states, start_idx = rrt_iteration(
            tree, key, obstacles, sst_params, sim_params, callables
        )
        return (tree, key, goal_mask, goal, states, start_idx, iter + 1)

    def cond_fn(carry):
        tree, _, _, goal, _, _, _ = carry
        return (goal == 0) & (tree.tree_size < MAX_TREE_SIZE - sim_params.batch_size)

    init_carry = (tree, jax.random.PRNGKey(i), jnp.zeros(sim_params.batch_size, dtype=bool),
                  jnp.array(0, dtype=jnp.int32), jnp.zeros([sim_params.batch_size, sim_params.dims]),
                  jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32))

    tree, key, goal_mask, goal, states, start_idx, iter = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    return tree, key, goal_mask, goal, states, start_idx, iter, tree.tree_size

# ------------------------------------------------------------------
# EXECUTION
# ------------------------------------------------------------------

if __name__ == "__main__":
    batch_size = 2048
    sim_params = MJXparams(
        motion_constraints=MotionConstraints(max_accel=10.0, min_accel=-10.0),
        physics_constants=PhysicsConstants(),
        batch_size=batch_size,
        bounds=Bounds(min_x=-jnp.pi, max_x=jnp.pi, min_y=-jnp.pi, max_y=jnp.pi),
        dims=4, action_dims=1, dt=0.05, seed=42
    )

    sst_params = SSTparams(
        batch_size=batch_size, δBN=0.1, δs=0.05, decay=0.8,
        start=Position(x=-jnp.pi/2, y=0.0, z=0.0), # theta1 = -pi/2 (down)
        goal=Position(x=jnp.pi/2, y=0.0, z=0.0),    # theta1 = pi/2 (up)
        goal_radius=0.15, time_to_evolve=5,
    )

    callables = Callables(
        prop_fn=propagate_acrobot, valid_fn=valid_acrobot, sample_fn=sample_acrobot,
        dist_fn=dist_acrobot, sampact_fn=sample_actions_acrobot, goal_fn=reached_goal_acrobot
    )

    SIM_PARAMS_RESERVED, CALLABLES_RESERVED = sim_params, callables
    obstacles = jnp.array([])

    print("\nStarting Acrobot RRT - Pre-compiling...")
    dummy_tree = rrtree.KinoTree.init(MAX_TREE_SIZE, sim_params.dims, sim_params.action_dims)
    _ = jit_while(dummy_tree, sst_params, sim_params, callables, obstacles, 0)
    print("Compilation complete.\n")

    times, iters, sizes, costs = [], [], [], []

    for i in range(10):
        print(i)
        tree = rrtree.KinoTree.init(MAX_TREE_SIZE, sim_params.dims, sim_params.action_dims)
        
        # UPDATED INIT STATE: [theta1, theta2, dtheta1, dtheta2]
        init_state = jnp.array([sst_params.start.x, sst_params.start.y, 0.0, 0.0])
        tree, _ = rrtree.add_nodes(tree, init_state, jnp.zeros(1), -1, 0.0, 1)
        
        start_p = time.perf_counter()
        result = jit_while(tree, sst_params, sim_params, callables, obstacles, i)
        tree, key, goal_mask, goal_found, states, start_idx, iter_val, size = jax.block_until_ready(result)
        timer = time.perf_counter() - start_p

        if jnp.any(goal_mask):
            path, actions = extract_sol(tree, goal_mask, start_idx)
            is_valid = verify_sol(path, actions, obstacles, sst_params, sim_params, callables)
            
            if not is_valid:
                print(f"Run {i}: Found path but verification failed!")

            cost = tree.costs[jnp.argmax(goal_mask) + start_idx]
            costs.append(cost); times.append(timer); iters.append(iter_val); sizes.append(size)
            print(f"Run {i:02d}: Goal reached! Iters: {iter_val}, Time: {timer*1e3:.2f}ms, Cost: {cost:.3f}")

    # Statistics (Consistent with original script)
    times, iters, sizes, costs = jnp.array(times), jnp.array(iters), jnp.array(sizes), jnp.array(costs)
    print(f"\nAverage time over {len(times)} runs: {jnp.mean(times)*1e3:.3f} ms, {jnp.mean(iters):.2f} iterations, size {jnp.mean(sizes):.2f}")
    print(f"min time: {jnp.min(times)*1e3:.3f} ms, max time: {jnp.max(times)*1e3:.3f} ms")
    print(f"Average cost: {jnp.mean(costs):.3f}, min cost: {jnp.min(costs):.3f}, max cost: {jnp.max(costs):.3f}")