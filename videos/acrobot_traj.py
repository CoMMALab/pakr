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
# ACROBOT DYNAMICS (ALIGNED WITH DYNOBENCH)
# ------------------------------------------------------------------

@jax.jit
def propagate_acrobot(states, actions, dt, constants):
    # Benchmark Parameters (Standard Acrobot)
    l1, l2 = 1.0, 1.0
    m1, m2 = 1.0, 1.0
    lc1, lc2 = 0.5, 0.5
    I1, I2 = 0.33, 0.33
    g = 9.81

    q1, q2, dq1, dq2 = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
    u = actions[:, 0]

    # Benchmark Mass Matrix Determinant
    det = I1*I2 + I2*(l1**2)*m2 - (l1**2)*(lc2**2)*(m2**2)*(jnp.cos(q2)**2)

    # q1_ddot (from Model_acrobot::calcV)
    t1 = -I2 * (g*lc1*m1*jnp.sin(q1) + g*m2*(l1*jnp.sin(q1) + lc2*jnp.sin(q1+q2)) \
         - 2.*l1*lc2*m2*dq1*dq2*jnp.sin(q2) - l1*lc2*m2*(dq2**2)*jnp.sin(q2))
    t2 = (I2 + l1*lc2*m2*jnp.cos(q2)) * (g*lc2*m2*jnp.sin(q1+q2) + l1*lc2*m2*(dq1**2)*jnp.sin(q2) - u)
    q1_ddot = (t1 + t2) / det

    # q2_ddot (from Model_acrobot::calcV)
    t3 = (I2 + l1*lc2*m2*jnp.cos(q2)) * (g*lc1*m1*jnp.sin(q1) + g*m2*(l1*jnp.sin(q1) + lc2*jnp.sin(q1+q2)) \
         - 2.*l1*lc2*m2*dq1*dq2*jnp.sin(q2) - l1*lc2*m2*(dq2**2)*jnp.sin(q2))
    t4 = (g*lc2*m2*jnp.sin(q1+q2) + l1*lc2*m2*(dq1**2)*jnp.sin(q2) - u) * \
         (I1 + I2 + (l1**2)*m2 + 2.*l1*lc2*m2*jnp.cos(q2))
    q2_ddot = (t3 - t4) / det

    new_dq1 = dq1 + q1_ddot * dt
    new_dq2 = dq2 + q2_ddot * dt
    new_q1 = (q1 + new_dq1 * dt + jnp.pi) % (2 * jnp.pi) - jnp.pi
    new_q2 = (q2 + new_dq2 * dt + jnp.pi) % (2 * jnp.pi) - jnp.pi

    return jnp.stack([new_q1, new_q2, new_dq1, new_dq2], axis=-1)

@jax.jit
def dist_acrobot(sim_params, diff):
    # diff: [dq1, dq2, dv1, dv2]
    # 1. Handle Angular wrap-around for the first two elements
    dq = (diff[..., 0:2] + jnp.pi) % (2 * jnp.pi) - jnp.pi
    
    # 2. Extract Velocities
    dv = diff[..., 2:4]
    
    # 3. Apply the 3-weight logic
    w = jnp.array([0.5, 0.5, 0.2])
    
    # dist = w0*|dq1| + w1*|dq2| + w2*sqrt(dv1^2 + dv2^2)
    angular_dist = jnp.abs(dq[..., 0]) * w[0] + jnp.abs(dq[..., 1]) * w[1]
    velocity_dist = jnp.linalg.norm(dv, axis=-1) * w[2]
    
    return (angular_dist + velocity_dist).T

@jax.jit
def reached_goal_acrobot(states, goal_vec, radius):
    # Swing-down Goal check
    diff = states - goal_vec
    diff_angles = (diff[:, 0:2] + jnp.pi) % (2 * jnp.pi) - jnp.pi
    # Re-using weighted Euclidean for the goal threshold
    dist_sq = jnp.sum(diff_angles**2, axis=-1) + 0.1 * jnp.sum(diff[:, 2:4]**2, axis=-1)
    return dist_sq < radius**2

@partial(jax.jit, static_argnums=(0,))
def sample_acrobot(sim_params, key):
    B = sim_params.batch_size
    k1, k2 = jax.random.split(key)
    angles = jax.random.uniform(k1, (B, 2), minval=-jnp.pi, maxval=jnp.pi)
    vels = jax.random.uniform(k2, (B, 2), minval=-8.0, maxval=8.0)
    return jnp.concatenate([angles, vels], axis=-1)

@partial(jax.jit, static_argnums=(0,))
def sample_actions_acrobot(sim_params, key):
    return jax.random.uniform(key, (sim_params.batch_size, 1), minval=-10.0, maxval=10.0)

@partial(jax.jit, static_argnums=(1))
def valid_acrobot(state, params, obstacles):
    return jnp.all(jnp.abs(state[:, 2:4]) < 8.0, axis=-1)

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

@partial(jax.jit, static_argnums=(2, 3, 4))
def jit_while(tree, goal_vec, sst_params, sim_params, callables, obstacles, i):
    def body_fn(carry):
        tree, key, _, _, _, _, iter = carry
        tree, key, goal_mask, goal_count, states, start_idx = rrt_iteration(
            tree, key, goal_vec, obstacles, sst_params, sim_params, callables
        )
        return (tree, key, goal_mask, goal_count, states, start_idx, iter + 1)

    def cond_fn(carry):
        tree, _, _, goal_count, _, _, _ = carry
        return (goal_count == 0) & (tree.tree_size < MAX_TREE_SIZE - sim_params.batch_size)

    init_carry = (tree, jax.random.PRNGKey(i), jnp.zeros(sim_params.batch_size, dtype=bool),
                  jnp.array(0, dtype=jnp.int32), jnp.zeros([sim_params.batch_size, sim_params.dims]),
                  jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32))

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
    goal_vec = jnp.array([jnp.pi, 0.0, 0.0, 0.0])

    sim_params = MJXparams(
        motion_constraints=MotionConstraints(max_accel=10.0, min_accel=-10.0),
        physics_constants=PhysicsConstants(),
        batch_size=batch_size,
        bounds=Bounds(min_x=-jnp.pi, max_x=jnp.pi, min_y=-jnp.pi, max_y=jnp.pi),
        dims=4,
        action_dims=1,
        dt=0.01,
        seed=42,
    )

    sst_params = SSTparams(
        batch_size=batch_size,
        δBN=0.1,
        δs=0.05,
        decay=0.8,
        start=Position(x=jnp.pi, y=0.0, z=0.0),
        goal=Position(x=0.0, y=0.0, z=0.0),
        goal_radius=0.25,
        time_to_evolve=43,
    )

    callables = Callables(
        prop_fn=propagate_acrobot,
        valid_fn=valid_acrobot,
        sample_fn=sample_acrobot,
        dist_fn=dist_acrobot,
        sampact_fn=sample_actions_acrobot,
        goal_fn=reached_goal_acrobot,
    )

    SIM_PARAMS_RESERVED, CALLABLES_RESERVED = sim_params, callables
    obstacles = jnp.array([])

    print("Compiling JAX kernels...")
    dummy_tree = rrtree.KinoTree.init(MAX_TREE_SIZE, sim_params.dims, sim_params.action_dims)
    _ = jit_while(dummy_tree, goal_vec, sst_params, sim_params, callables, obstacles, 0)

    # ---------------------------------------------------------
    # RUN PLANNER ONCE
    # ---------------------------------------------------------

    tree = rrtree.KinoTree.init(MAX_TREE_SIZE, sim_params.dims, sim_params.action_dims)

    init_state = jnp.array([0.0, 0.0, 0.0, 0.0])
    tree, _ = rrtree.add_nodes(tree, init_state, jnp.zeros(1), -1, 0.0, 1)

    print("Running planner...")

    tree, key, goal_mask, goal_found, states, start_idx, iters = jax.block_until_ready(
        jit_while(tree, goal_vec, sst_params, sim_params, callables, obstacles, 0)
    )

    if goal_found == 0:
        print("Planner failed to find a solution.")
        exit()

    path, actions = extract_sol(tree, goal_mask, start_idx)

    print("Solution found.")
    print("Actions:", len(actions))

    # ---------------------------------------------------------
    # REPLAY ACTIONS AND RECORD TRAJECTORY
    # ---------------------------------------------------------

    print("Replaying trajectory...")

    traj = []

    state = jnp.array([0.0, 0.0, 0.0, 0.0])

    traj.append(np.array(state))

    for action in actions[1:]:  # skip root dummy action

        action = action[None, :]

        for step in range(sst_params.time_to_evolve):

            state = propagate_acrobot(state[None, :], action, sim_params.dt, None)[0]

            if step % 1 == 0:
                traj.append(np.array(state))

    traj = np.array(traj)

    # ---------------------------------------------------------
    # SAVE TRAJECTORY
    # ---------------------------------------------------------

    import os
    os.makedirs("videos", exist_ok=True)

    save_path = "videos/acrobot_traj.npy"
    np.save(save_path, traj)

    print("Saved trajectory:", save_path)
    print("Trajectory length:", len(traj))