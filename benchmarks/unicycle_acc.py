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
    # state: [x, y, theta] -> check distance on [x, y]
    pos = states[:, 0:2]
    goal_pos = jnp.array([goal.x, goal.y])
    diff2 = jnp.sum((pos - goal_pos)**2, axis=-1)
    return diff2 < radius**2

@partial(jax.jit, static_argnums=(0,))
def sample_actions_unicycle(sim_params, key):
    # Action: [v, omega]
    B = sim_params.batch_size
    # Note: Using max_vel for v and max_accel for omega in constraints mapping
    tau = jax.random.uniform(
        key, (B, 2),
        minval=jnp.array([0.0, sim_params.motion_constraints.min_accel]), 
        maxval=jnp.array([sim_params.motion_constraints.max_vel, sim_params.motion_constraints.max_accel])
    )
    return tau

@jax.jit
def propagate_unicycle_dynamic(states, actions, dt, constants):
    """
    states:  (batch, 5) -> [x, y, theta, v, omega]
    actions: (batch, 2) -> [accel, alpha]
    """
    x, y, theta, v, omega = states[:, 0], states[:, 1], states[:, 2], states[:, 3], states[:, 4]
    a, alpha = actions[:, 0], actions[:, 1]

    # Update velocities
    new_v = v + a * dt
    new_omega = omega + alpha * dt

    # Midpoint integration for pose
    mid_v = v + 0.5 * a * dt
    mid_omega = omega + 0.5 * alpha * dt
    mid_theta = theta + 0.5 * mid_omega * dt

    new_x = x + mid_v * jnp.cos(mid_theta) * dt
    new_y = y + mid_v * jnp.sin(mid_theta) * dt
    new_theta = theta + new_omega * dt

    return jnp.stack([new_x, new_y, new_theta, new_v, new_omega], axis=-1)

@partial(jax.jit, static_argnums=(0,))
def dist_unicycle_dynamic(sim_params, diff):
    """
    diff: (..., 5) -> [dx, dy, dtheta, dv, domega]
    Weights: Position (1.0), Heading (0.3), Velocity (0.1)
    """
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
    
    # Sample v and omega within reasonable starting ranges
    vels = jax.random.uniform(k3, (B, 2), 
        minval=jnp.array([sim_params.motion_constraints.min_vel, -1.0]), 
        maxval=jnp.array([sim_params.motion_constraints.max_vel, 1.0]))

    return jnp.concatenate([pos, theta, vels], axis=-1)

@partial(jax.jit, static_argnums=(1))
def valid_unicycle_dynamic(state, params, obstacles):
    # state: [x, y, theta, v, omega]
    x, y, v = state[:, 0], state[:, 1], state[:, 3]
    within_bounds = (x >= params.bounds.min_x) & (x <= params.bounds.max_x) & \
                    (y >= params.bounds.min_y) & (y <= params.bounds.max_y)
    
    # Dynamic constraint: keep velocity within bounds
    vel_ok = (v >= params.motion_constraints.min_vel) & (v <= params.motion_constraints.max_vel)
    collision_free = helper.collision_check_2d(state[:, :2], obstacles)
    
    return within_bounds & collision_free & vel_ok

# ------------------------------------------------------------------
# PLANNER SETUP
# ------------------------------------------------------------------

MAX_TREE_SIZE = 100000
TIERS = [512, 1024, 4096, 16384, 32768, 64536, 100000]

# Use global references for the NN factory as in your previous snippet
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

    # Masking and Node Addition
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
        key, subkey = jax.random.split(key)
        tree, subkey, goal_mask, goal, states, start_idx = rrt_iteration(
            tree, subkey, obstacles, sst_params, sim_params, callables
        )
        return (tree, key, goal_mask, goal, states, start_idx, iter + 1)

    def cond_fn(carry):
        tree, key, goal_mask, goal, states, start_idx, iter = carry
        return (goal == 0) & (tree.tree_size < MAX_TREE_SIZE - sim_params.batch_size)

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
    
    #print(jnp.sum(goal_mask))

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
    for i in range(len(actions)-1):
        start = path[i][None, :]    # shape: (1, state_dim)
        action = actions[i+1][None, :]
        states_end, valid_mask, _ = propagate.rollout_final(
            start, action, obstacles, sst_params, sim_params, callables
        )
        #print(states_end)
        if not valid_mask:
            return False
    return True
    

MAX_TREE_SIZE = 50000

# ------------------------------------------------------------------
# EXECUTION
# ------------------------------------------------------------------

if __name__ == "__main__":
    batch_size = 2048
    
    motion_constraints = MotionConstraints(
        max_vel = 0.6, min_vel = -0.1, 
        max_accel = 1.0, min_accel = -1.0 # These are now linear/angular accelerations
    )

    sim_params = MJXparams(
        motion_constraints=motion_constraints,
        physics_constants=PhysicsConstants(),
        batch_size=batch_size,
        bounds = Bounds(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0, min_z=0.0, max_z=0.0),
        dims=5, action_dims=2, dt = 0.1, seed = 42
    )

    sst_params = SSTparams(
        batch_size=batch_size, δBN=0.04, δs=0.02, decay=0.8,
        start=Position(x=0.05, y=0.05, z=0.0),
        goal=Position(x=0.95, y=0.95, z=0.0),
        goal_radius=0.05, time_to_evolve=10,
    )

    callables = Callables(
        prop_fn=propagate_unicycle_dynamic,
        valid_fn=valid_unicycle_dynamic,
        sample_fn=sample_unicycle_dynamic,
        dist_fn=dist_unicycle_dynamic,
        sampact_fn=sample_actions_unicycle, # Uses [accel, alpha] ranges
        goal_fn=reached_goal_unicycle
    )

    SIM_PARAMS_RESERVED = sim_params
    CALLABLES_RESERVED = callables
    obstacles = helper.get_obs('envs/tree2d.csv')

    # ------------------------------------------------------------------
    # 1. Compilation Warm-up
    # ------------------------------------------------------------------
    print("\nStarting Unicycle RRT - Pre-compiling kernels...")
    # Initialize a dummy tree with correct dims (3 for state, 2 for action)
    dummy_tree = rrtree.KinoTree.init(MAX_TREE_SIZE, sim_params.dims, sim_params.action_dims)
    _ = jit_while(dummy_tree, sst_params, sim_params, callables, obstacles, 0)
    print("Compilation complete.\n")

    # ------------------------------------------------------------------
    # 2. Execution Loop
    # ------------------------------------------------------------------
    times, iters, sizes, costs = [], [], [], []
    # Execution Loop
    for i in range(10):
        tree = rrtree.KinoTree.init(MAX_TREE_SIZE, sim_params.dims, sim_params.action_dims)
        
        # FIXED INIT STATE: [x, y, theta, v, omega]
        init_state = jnp.array([sst_params.start.x, sst_params.start.y, 0.0, 0.0, 0.0])
        tree, _ = rrtree.add_nodes(tree, init_state, jnp.zeros(2), -1, 0.0, 1)
        start_p = time.perf_counter()
        # Solve via JIT-compiled while loop
        result = jit_while(tree, sst_params, sim_params, callables, obstacles, i)
        
        # Unpack results
        tree, key, goal_mask, goal_found, states, start_idx, iter_val, size = jax.block_until_ready(result)
        
        timer = time.perf_counter() - start_p

        # Extract path if a goal was reached
        if jnp.any(goal_mask):
            path, actions = extract_sol(tree, goal_mask, start_idx)
            
            # Verify the solution using the unicycle dynamics
            is_valid = verify_sol(path, actions, obstacles, sst_params, sim_params, callables)
            
            if not is_valid:
                print(f"Run {i}: Found path but verification failed!")
            
            # Calculate cost (using the index of the first state that hit the goal)
            goal_node_idx = jnp.where(goal_mask, size, -1).min() # Simplistic index recovery
            cost = tree.costs[jnp.argmax(goal_mask) + start_idx]
            
            costs.append(cost)
            times.append(timer)
            iters.append(iter_val)
            sizes.append(size)
            
            print(f"Run {i:02d}: Goal reached! Iters: {iter_val}, Time: {timer*1e3:.2f}ms, Cost: {cost:.3f}")

        # Final Statistics Logic
    times = jnp.array(times)
    iters = jnp.array(iters)
    sizes = jnp.array(sizes)
    costs = jnp.array(costs)

    print(f"Average time over 100 runs: {jnp.mean(times)*1e3:.3f} ms, {jnp.mean(iters):.2f} iterations, size {jnp.mean(sizes):.2f}")
    print(f"min time over 100 runs: {jnp.min(times)*1e3:.3f} ms, {jnp.min(iters)} iterations, min size {jnp.min(sizes)}")
    print(f"max time over 100 runs: {jnp.max(times)*1e3:.3f} ms, {jnp.max(iters)} iterations, max size {jnp.max(sizes)}")
    print(f"Average cost over 100 runs: {jnp.mean(costs):.3f}, min cost: {jnp.min(costs):.3f}, max cost: {jnp.max(costs):.3f}")