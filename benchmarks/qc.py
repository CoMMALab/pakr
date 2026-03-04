import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
import time
import rrtree
import propagate
import helper
from params import Bounds, Position, MotionConstraints, PhysicsConstants, MJXparams, SSTparams, Callables


import jax
import jax.numpy as jnp
from functools import partial

# ------------------------------------------------------------------
# COLLISION & BOUNDARY CONSTANTS (Hardcoded from quadrotor_v0-window)
# ------------------------------------------------------------------
# Obstacles: [center_x, center_y, center_z, size_x, size_y, size_z]
OBSTACLES = jnp.array([
    [4.0, 3.0, 2.0, 2.0, 0.3, 2.0],  # Main block
    [1.1, 3.0, 1.9, 0.2, 0.3, 1.0],  # Side post
    [2.0, 3.0, 2.7, 2.0, 0.3, 0.6],  # Horizontal top
    [2.0, 3.0, 1.2, 2.0, 0.3, 0.4]   # Horizontal bottom
])

ROBOT_RADIUS = 0.25 

# ------------------------------------------------------------------
# CORE HELPERS
# ------------------------------------------------------------------

def quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return jnp.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def rotate_vector(q, v):
    qv = jnp.array([v[0], v[1], v[2], 0.0])
    q_inv = q * jnp.array([-1., -1., -1., 1.])
    return quat_multiply(quat_multiply(q, qv), q_inv)[:3]

@jax.jit
def propagate_quadcopter(states, actions, dt, constants):
    # Dynobench alignment: m=0.034, g=9.81
    m = 0.034
    g = 9.81
    arm = 0.707106781 * 0.046
    t2t = 0.006
    J = jnp.array([16.571710e-6, 16.655602e-6, 29.261652e-6])
    invJ = 1.0 / J

    pos = states[:, 0:3]
    quat = states[:, 3:7] 
    vel = states[:, 7:10]
    omega = states[:, 10:13]
    u = actions 

    # Mixer Logic (B0 matrix from dynobench)
    thrust = jnp.sum(u, axis=1)
    tau_x = arm * (-u[:, 0] - u[:, 1] + u[:, 2] + u[:, 3])
    tau_y = arm * (-u[:, 0] + u[:, 1] + u[:, 2] - u[:, 3])
    tau_z = t2t * (-u[:, 0] + u[:, 1] - u[:, 2] + u[:, 3])

    # Linear Physics
    f_body = jnp.stack([jnp.zeros_like(thrust), jnp.zeros_like(thrust), thrust], axis=-1)
    f_world = jax.vmap(rotate_vector)(quat, f_body)
    accel = (f_world / m) + jnp.array([0, 0, -g])

    # Angular Physics
    torque = jnp.stack([tau_x, tau_y, tau_z], axis=-1)
    alpha = (torque - jnp.cross(omega, omega * J)) * invJ

    # Integration
    new_pos = pos + vel * dt
    new_vel = vel + accel * dt
    new_omega = omega + alpha * dt
    
    def quat_step(q, w):
        # Local rotation vector for dt
        angle_vec = w * dt
        angle = jnp.linalg.norm(angle_vec)
        axis = angle_vec / (angle + 1e-9)
        dq = jnp.where(angle > 1e-9, 
                       jnp.concatenate([jnp.sin(angle/2)*axis, jnp.cos(angle/2)[None]]),
                       jnp.array([0., 0., 0., 1.]))
        res = quat_multiply(q, dq)
        return res / jnp.linalg.norm(res)

    new_quat = jax.vmap(quat_step)(quat, omega)
    return jnp.concatenate([new_pos, new_quat, new_vel, new_omega], axis=-1)

@jax.jit
def check_collision(pos):
    """Checks if a sphere at pos with ROBOT_RADIUS hits any hardcoded box."""
    # center: obs[:, 0:3], size: obs[:, 3:6]
    half_size = OBSTACLES[:, 3:6] / 2.0
    
    # Distance from point to box (vectorized over obstacles)
    # dx = max(abs(px - cx) - half_w, 0)
    def single_box_dist(p, center, h_size):
        dist_vec = jnp.maximum(jnp.abs(p - center) - h_size, 0.0)
        return jnp.linalg.norm(dist_vec)

    # vmap over obstacles for a single position
    dists = jax.vmap(single_box_dist, in_axes=(None, 0, 0))(pos, OBSTACLES[:, 0:3], half_size)
    return jnp.any(dists < ROBOT_RADIUS)

@partial(jax.jit, static_argnums=(1))
def valid_quadcopter(state, params, obstacles):
    # 1. Boundary Check
    in_bounds = jnp.all((state[:, 0:3] >= jnp.array([1., 0.5, 1.])) & 
                        (state[:, 0:3] <= jnp.array([5., 5.5, 3.])), axis=-1)
    # 2. Velocity Check
    vel_valid = jnp.all(jnp.abs(state[:, 7:10]) <= 4.0, axis=-1)
    # 3. Collision Check (vmap over the batch of states)
    not_colliding = ~jax.vmap(check_collision)(state[:, 0:3])
    
    return in_bounds & vel_valid & not_colliding

@jax.jit
def dist_quadcopter(sim_params, diff):
    # Weights: [pos: 1.0, quat: 0.5, vel: 0.1, omega: 0.05]
    w = jnp.array([1.0, 0.5, 0.1, 0.05])
    pos_d = jnp.linalg.norm(diff[..., 0:3], axis=-1)
    quat_d = jnp.linalg.norm(diff[..., 3:7], axis=-1) # Euclidean approx for SO3
    vel_d = jnp.linalg.norm(diff[..., 7:10], axis=-1)
    omg_d = jnp.linalg.norm(diff[..., 10:13], axis=-1)
    return (w[0]*pos_d + w[1]*quat_d + w[2]*vel_d + w[3]*omg_d).T

@jax.jit
def reached_goal_quadcopter(states, goal_vec, radius):
    # Dynobench often uses a threshold on both position and velocity
    pos_err = jnp.linalg.norm(states[:, 0:3] - goal_vec[0:3], axis=-1)
    vel_err = jnp.linalg.norm(states[:, 7:10] - goal_vec[7:10], axis=-1)
    return (pos_err < radius) & (vel_err < radius)

@partial(jax.jit, static_argnums=(0,))
def sample_quadcopter(sim_params, key):
    B = sim_params.batch_size
    k1, k2, k3, k4 = jax.random.split(key, 4)
    pos = jax.random.uniform(k1, (B, 3), minval=jnp.array([1., 0.5, 1.]), maxval=jnp.array([5., 5.5, 3.]))
    quat = jax.random.normal(k2, (B, 4))
    quat = quat / jnp.linalg.norm(quat, axis=-1, keepdims=True)
    vel = jax.random.uniform(k3, (B, 3), minval=-4.0, maxval=4.0)
    omega = jax.random.uniform(k4, (B, 3), minval=-8.0, maxval=8.0)
    return jnp.concatenate([pos, quat, vel, omega], axis=-1)

@partial(jax.jit, static_argnums=(0,))
def sample_actions_quadcopter(sim_params, key):
    # Motor range [0.0, 1.3]
    return jax.random.uniform(key, (sim_params.batch_size, 4), minval=0.0, maxval=1.3)

# ------------------------------------------------------------------
# PLANNER LOGIC
# ------------------------------------------------------------------

MAX_TREE_SIZE = 2_000_000
TIERS = [512, 1024, 4096, 16384, 32768, 65536, 131072, 262144, 524288, 1_000_000, 2_000_000]  # Exponentially increasing tiers for NN search
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

@partial(jax.jit, static_argnums=(4, 5, 6))
def rrt_iteration(tree, rng_key, goal_vec, obstacles, sst_params, sim_params, callables):
    B, A = sim_params.batch_size, 64
    K = B // A 
    rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)
    seed_pts = callables.sample_fn(sim_params, subkey1)[:K]
    branch_idx = jnp.minimum(jnp.digitize(tree.tree_size, jnp.array(TIERS)), len(TIERS) - 1)
    parents_small = jax.lax.switch(branch_idx, NN_BRANCHES, (tree.states, tree.tree_size, seed_pts)) 
    
    start_states = jnp.repeat(tree.states[parents_small], A, axis=0) 
    parents = jnp.repeat(parents_small, A, axis=0) 
    actions = callables.sampact_fn(sim_params, subkey2)

    states_end, valid_mask, _ = propagate.rollout_final_2d(
        start_states, actions, obstacles, sst_params, sim_params, callables
    )

    valid_mask = valid_mask.at[-1].set(False)
    num_new = jnp.sum(valid_mask)
    valid_idx = jnp.nonzero(valid_mask, size=B, fill_value=-1)[0]

    tree, start_idx = rrtree.add_nodes(
        tree, states_end[valid_idx], actions[valid_idx], 
        parents[valid_idx], tree.costs[parents[valid_idx]] + sim_params.dt * sst_params.time_to_evolve, num_new
    )
    goal_mask = callables.goal_fn(states_end[valid_idx], goal_vec, sst_params.goal_radius)
    return tree, rng_key, goal_mask, jnp.sum(goal_mask), states_end[valid_idx], start_idx

# @partial(jax.jit, static_argnums=(2, 3, 4))
# def jit_while(tree, goal_vec, sst_params, sim_params, callables, obstacles, i):
#     def body_fn(carry):
#         tree, key, _, _, _, _, iter = carry
#         tree, key, goal_mask, goal_count, states, start_idx = rrt_iteration(
#             tree, key, goal_vec, obstacles, sst_params, sim_params, callables
#         )
#         return (tree, key, goal_mask, goal_count, states, start_idx, iter + 1)

#     def cond_fn(carry):
#         tree, _, _, goal_count, _, _, _ = carry
#         return (goal_count == 0) & (tree.tree_size < MAX_TREE_SIZE - sim_params.batch_size)

#     init_carry = (tree, jax.random.PRNGKey(i), jnp.zeros(sim_params.batch_size, dtype=bool),
#                   jnp.array(0, dtype=jnp.int32), jnp.zeros([sim_params.batch_size, sim_params.dims]),
#                   jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32))

#     return jax.lax.while_loop(cond_fn, body_fn, init_carry)

@partial(jax.jit, static_argnums=(2, 3, 4, 7)) # Added 7 for max_iters
def jit_while(tree, goal_vec, sst_params, sim_params, callables, obstacles, i, max_iters=100000):
    def body_fn(carry):
        tree, key, _, _, _, _, iters = carry
        tree, key, goal_mask, goal_count, states, start_idx = rrt_iteration(
            tree, key, goal_vec, obstacles, sst_params, sim_params, callables
        )
        return (tree, key, goal_mask, goal_count, states, start_idx, iters + 1)

    def cond_fn(carry):
        tree, _, _, goal_count, _, _, iters = carry
        # 1. No goal found yet
        # 2. Tree has room for at least one more full batch
        # 3. We haven't exceeded the iteration cap
        can_continue = (goal_count == 0) & \
                       (tree.tree_size < MAX_TREE_SIZE - sim_params.batch_size) & \
                       (iters < max_iters)
        return can_continue

    init_carry = (tree, jax.random.PRNGKey(i), 
                  jnp.zeros(sim_params.batch_size, dtype=bool),
                  jnp.array(0, dtype=jnp.int32), 
                  jnp.zeros([sim_params.batch_size, sim_params.dims]),
                  jnp.array(0, dtype=jnp.int32), 
                  jnp.array(0, dtype=jnp.int32)) # Iteration counter

    return jax.lax.while_loop(cond_fn, body_fn, init_carry)

def verify_sol(path, actions, obstacles, sst_params, sim_params, callables):
    if path is None: return False
    for i in range(len(actions)-1):
        start = path[i][None, :]
        action = actions[i+1][None, :]
        states_end, valid_mask, _ = propagate.rollout_final_2d(
            start, action, obstacles, sst_params, sim_params, callables
        )
        if not valid_mask: return False
    return True
def extract_sol(tree, goal_mask, start_idx):
    if jnp.sum(goal_mask) == 0:
        print(tree.tree_size)
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
# ------------------------------------------------------------------
# EXECUTION
# ------------------------------------------------------------------

if __name__ == "__main__":
    batch_size = 32768
    # State: [px, py, pz, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
    # Goal from YAML: p=[4, 5, 2], quat=[0, 0, 0, 1], v=0, w=0
    goal_vec = jnp.array([4.0, 5.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    sim_params = MJXparams(
        motion_constraints=MotionConstraints(max_accel=25.0, min_accel=-25.0), # max_acc from yaml
        physics_constants=PhysicsConstants(),
        batch_size=batch_size,
        # Bounds from YAML environment: min [1, 0.5, 1], max [5, 5.5, 3]
        bounds=Bounds(min_x=1.0, max_x=5.0, min_y=0.5, max_y=5.5), 
        dims=13, 
        action_dims=4, 
        dt=0.01, 
        seed=42
    )

    sst_params = SSTparams(
        batch_size=batch_size, 
        δBN=0.1, 
        δs=0.05, 
        decay=0.8,
        start=Position(x=4.0, y=1.0, z=2.0), # Start from YAML
        goal=Position(x=4.0, y=5.0, z=2.0),  # Goal from YAML
        goal_radius=0.5, 
        time_to_evolve=20, # Standard rollout steps for quadcopter in dynobench
    )

    callables = Callables(
        prop_fn=propagate_quadcopter, 
        valid_fn=valid_quadcopter, 
        sample_fn=sample_quadcopter,
        dist_fn=dist_quadcopter, 
        sampact_fn=sample_actions_quadcopter, 
        goal_fn=reached_goal_quadcopter
    )

    SIM_PARAMS_RESERVED, CALLABLES_RESERVED = sim_params, callables
    obstacles = OBSTACLES # Using the hardcoded list from the previous step

    # print("Compiling JAX kernels for Quadcopter...")
    # dummy_tree = rrtree.KinoTree.init(MAX_TREE_SIZE, sim_params.dims, sim_params.action_dims)
    # _ = jit_while(dummy_tree, goal_vec, sst_params, sim_params, callables, obstacles, 0)

    print("Compiling JAX kernels for Quadcopter (this may take 1-3 minutes)...")
    dummy_tree = rrtree.KinoTree.init(MAX_TREE_SIZE, sim_params.dims, sim_params.action_dims)
    
    # NEW: Create a 'dummy' version of sst_params with a tiny MAX_TREE_SIZE just for JIT
    # Or simply wrap the call in a way that terminates quickly
    _ = jit_while(dummy_tree, goal_vec, sst_params, sim_params, callables, obstacles, 0, max_iters=5)
    
    all_durations = []
    all_sizes = []
    all_costs = []
    success_count = 0

    print(f"Starting {100} runs of Quadcopter Window Navigation...")
    
    for i in range(10):
        tree = rrtree.KinoTree.init(MAX_TREE_SIZE, sim_params.dims, sim_params.action_dims)
        # Init state from YAML
        init_state = jnp.array([4.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        tree, _ = rrtree.add_nodes(tree, init_state, jnp.zeros(4), -1, 0.0, 1)
        
        start_t = time.perf_counter()
        tree, key, goal_mask, goal_found, states, start_idx, iters = jax.block_until_ready(
            jit_while(tree, goal_vec, sst_params, sim_params, callables, obstacles, i)
        )
        duration = time.perf_counter() - start_t
        
        if goal_found > 0:
            path, actions = extract_sol(tree, goal_mask, start_idx)
            is_valid = verify_sol(path, actions, obstacles, sst_params, sim_params, callables)
            
            if not is_valid:
                print(f"Run {i}: Found path but verification failed!")
                continue

            goal_node_idx = start_idx + jnp.argmax(goal_mask)
            cost = float(tree.costs[goal_node_idx])
            
            all_durations.append(duration)
            all_sizes.append(int(tree.tree_size))
            all_costs.append(cost)
            success_count += 1
            
            print(f"Run {i:02d}: Time: {duration:.3f}s | Size: {tree.tree_size} | Cost: {cost:.2f}")
        else:
            print(f"Run {i:02d}: Failed to find solution")

    # Final Stats (same as before)
    # ...

    # --------------------------------------------------------------
    # STATISTICS REPORT
    # --------------------------------------------------------------
    print("\n" + "="*50)
    print(f"FINAL STATISTICS ({success_count}/100 Successful)")
    print("="*50)
    
    if success_count > 0:
        stats = {
            "Solve Time (s)": all_durations,
            "Tree Size": all_sizes,
            "Path Cost (s)": all_costs
        }
        
        for label, data in stats.items():
            arr = np.array(data)
            print(f"{label:15} | Mean: {np.mean(arr):10.3f} | Median: {np.median(arr):10.3f}")
    else:
        print("No successful runs to report statistics.")
    print("="*50)