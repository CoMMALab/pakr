import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
import mujoco
from mujoco import mjx
import rrtree
import helper
import params
import time
import gc
from params import MJXparams, SSTparams, Callables, MotionConstraints, PhysicsConstants, Position, Bounds

# ------------------------------------------------------------------
# WALKER SPECIFIC HELPERS
# ------------------------------------------------------------------

@jax.jit
def valid_walker(states, sim_params):
    """
    Checks if walker is upright and within horizontal bounds.
    states: [batch, 18] -> [rootx, rootz, rooty, joints..., velocities...]
    """
    z_height = states[:, 1]
    ang = states[:, 2]
    
    # Standard Gym-like termination: height > 0.8 and |angle| < 1.0 rad
    is_upright = (z_height > 0.8) & (z_height < 2.0) & (jnp.abs(ang) < 1.0)
    
    # Horizontal bounds
    x_pos = states[:, 0]
    x_valid = (x_pos > sim_params.bounds.min_x) & (x_pos < sim_params.bounds.max_x)
    
    return is_upright & x_valid

@jax.jit
def dist_walker(sim_params, diff):
    # Higher weights on root position (x, z) and torso angle
    # diff: [qpos_diff (9), qvel_diff (9)]
    weights = jnp.array([
        1.0, 1.0, 2.0,  # Root X, Z, Angle
        0.1, 0.1, 0.1,  # Right Leg
        0.1, 0.1, 0.1,  # Left Leg
        0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01 # Velocities
    ])
    return jnp.sum(weights * (diff**2), axis=-1).T

@jax.jit
def reached_goal_walker(states, goal, radius):
    # Focus goal on reaching a target X position while remaining upright
    x_dist = jnp.abs(states[:, 0] - goal.x)
    z_dist = jnp.abs(states[:, 1] - 1.25) # Target nominal height
    return (x_dist < radius) & (z_dist < 0.3)

@partial(jax.jit, static_argnums=(0,))
def sample_walker(sim_params, key):
    k1, k2 = jax.random.split(key)
    # Sample around nominal standing pose
    # qpos: [x, z, ang, r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle]
    pos_min = jnp.array([sim_params.bounds.min_x, 0.8, -0.5, -1.0, -1.5, -0.75, -1.0, -1.5, -0.75])
    pos_max = jnp.array([sim_params.bounds.max_x, 1.5, 0.5, 1.0, 0.0, 0.75, 1.0, 0.0, 0.75])
    
    pos = jax.random.uniform(k1, (sim_params.batch_size, 9), minval=pos_min, maxval=pos_max)
    vel = jax.random.uniform(k2, (sim_params.batch_size, 9), minval=-1.0, maxval=1.0)
    
    return jnp.concatenate([pos, vel], axis=-1)

@partial(jax.jit, static_argnums=(0,))
def sample_actions_walker(sim_params, key):
    # Walker has 6 motors, ctrlrange is [-1, 1]
    return jax.random.uniform(key, (sim_params.batch_size, 6), minval=-1.0, maxval=1.0)

# ------------------------------------------------------------------
# MJX PROPAGATION (Logic remains same as Cartpole template)
# ------------------------------------------------------------------

def make_walker_propagate(mjx_model):
    @partial(jax.jit, static_argnums=(0,))
    def propagate_batch(num_steps, states, actions):
        nq, nv = mjx_model.nq, mjx_model.nv
        qpos, qvel = states[:, :nq], states[:, nq : nq + nv]
        template = mjx.make_data(mjx_model)

        def _make_data(qp, qv, u):
            return template.replace(qpos=qp, qvel=qv, ctrl=u)

        data_batch = jax.vmap(_make_data)(qpos, qvel, actions)
        vmapped_step = jax.vmap(mjx.step, in_axes=(None, 0))

        def step_fn(_, data):
            return vmapped_step(mjx_model, data)

        final_data = jax.lax.fori_loop(0, num_steps, step_fn, data_batch)
        return jnp.concatenate([final_data.qpos, final_data.qvel], axis=-1)

    return propagate_batch

def make_walker_rollout(prop_fn):
    def rollout_final_walker(state0, actions, obstacles, sst_params, sim_params):
        batch = state0.shape[0]
        final_states = prop_fn(sst_params.time_to_evolve, state0, actions)
        valid_mask = valid_walker(final_states, sim_params)
        cost = jnp.full((batch,), 0.01 * sst_params.time_to_evolve) # Time-based cost
        return final_states, valid_mask, cost
    return rollout_final_walker


# ------------------------------------------------------------------
# RRT INFRASTRUCTURE
# ------------------------------------------------------------------

SIM_PARAMS_RESERVED = None
CALLABLES_RESERVED = None
TIERS = [512, 4096, 16384, 64536, 128000]

def nn_tier_factory(size):
    def nn_fn(operands):
        states, tree_size, query = operands
        parents, _ = helper.nearest_neighbor_masked(
            SIM_PARAMS_RESERVED, CALLABLES_RESERVED.dist_fn, 
            states[:size], tree_size, query
        )
        return parents
    return nn_fn

NN_BRANCHES = [nn_tier_factory(t) for t in TIERS]

@partial(jax.jit, static_argnums=(3, 4, 5))
def rrt_iteration(tree, rng_key, obstacles, sst_params, sim_params, callables):
    B = sim_params.batch_size
    K = B // A 

    rng_key, sk1, sk2 = jax.random.split(rng_key, 3)
    seed_pts = callables.sample_fn(sim_params, sk1)[:K]

    branch_idx = jnp.minimum(jnp.digitize(tree.tree_size, jnp.array(TIERS)), len(TIERS) - 1)
    parents_small = jax.lax.switch(branch_idx, NN_BRANCHES, (tree.states, tree.tree_size, seed_pts))

    start_states = jnp.repeat(tree.states[parents_small], A, axis=0)
    parents = jnp.repeat(parents_small, A, axis=0)
    actions = callables.sampact_fn(sim_params, sk2)

    states_end, valid_mask, dist_traveled = callables.prop_fn(
        start_states, actions, obstacles, sst_params, sim_params
    )

    # Filter/Padding logic
    valid_mask = valid_mask.at[-1].set(False)
    num_new = jnp.sum(valid_mask)
    valid_idx = jnp.nonzero(valid_mask, size=B, fill_value=-1)[0]

    new_states = states_end[valid_idx]
    new_costs = tree.costs[parents[valid_idx]] + dist_traveled[valid_idx]

    tree, start_idx = rrtree.add_nodes(tree, new_states, actions[valid_idx], parents[valid_idx], new_costs, num_new)
    goal_mask = callables.goal_fn(new_states, sst_params.goal, sst_params.goal_radius)

    return tree, rng_key, goal_mask, jnp.sum(goal_mask), new_states, start_idx

@partial(jax.jit, static_argnums=(1, 2, 3))
def jit_while(tree, sst_params, sim_params, callables, obstacles, i):
    def body_fn(carry):
        tree, key, goal_mask, goal_count, _, _, iter_cnt = carry
        key, subkey = jax.random.split(key)
        tree, subkey, goal_mask, goal_count, states, start_idx = rrt_iteration(
            tree, subkey, obstacles, sst_params, sim_params, callables
        )
        return (tree, key, goal_mask, goal_count, states, start_idx, iter_cnt + 1)

    def cond_fn(carry):
        tree, _, _, goal_count, _, _, _ = carry
        return (goal_count == 0) & (tree.tree_size < MAX_TREE_SIZE - sim_params.batch_size)

    init_carry = (tree, jax.random.PRNGKey(i), jnp.zeros(sim_params.batch_size, dtype=bool),
                  jnp.array(0, dtype=jnp.int32), jnp.zeros([sim_params.batch_size, sim_params.dims]),
                  jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32))
    
    return jax.lax.while_loop(cond_fn, body_fn, init_carry)

# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------
MAX_TREE_SIZE = 128000

if __name__ == "__main__":
    # Load Walker Model
    model = mujoco.MjModel.from_xml_path("models/walker.xml") 
    mjx_model = mjx.put_model(model)
    
    b = 2048 # Reduced batch size slightly due to higher state dimensionality
    A = 16

    sim_params = MJXparams(
        motion_constraints=MotionConstraints(max_vel=5.0, min_vel=-5.0, max_accel=20.0, min_accel=-20.0),
        physics_constants=PhysicsConstants(),
        batch_size=b,
        bounds=Bounds(min_x=-10.0, max_x=10.0, min_y=-1.0, max_y=1.0),
        dims=18,        # 9 pos + 9 vel
        action_dims=6,  # 6 motors
        dt=0.005        # Matching XML timestep
    )

    sst_params = SSTparams(
        batch_size=b, δBN=0.2, δs=0.1, decay=0.9,
        start=Position(x=0.0, y=1.25, z=0.0), 
        goal=Position(x=5.0, y=1.25, z=0.0), 
        goal_radius=0.5,
        time_to_evolve=10 # 10 MJX steps per edge
    )

    prop = make_walker_propagate(mjx_model)
    callables = Callables(
        prop_fn=make_walker_rollout(prop),
        valid_fn=valid_walker,
        sample_fn=sample_walker,
        dist_fn=dist_walker,
        sampact_fn=sample_actions_walker,
        goal_fn=reached_goal_walker
    )

    SIM_PARAMS_RESERVED = sim_params
    CALLABLES_RESERVED = callables
    obstacles = jnp.array([]) # No obstacles in standard cartpole

    print("Compiling MJX Walker Kernels...")
    tree = rrtree.KinoTree.init(MAX_TREE_SIZE, sim_params.dims, sim_params.action_dims)


    init_qpos = jnp.array([
        0.0,    # rootx
        1.25,   # rootz (height)
        0.0,    # rooty (torso angle)
        -0.2,   # right_hip
        -0.3,   # right_knee
        0.1,    # right_ankle
        -0.2,   # left_hip
        -0.3,   # left_knee
        0.1     # left_ankle
    ])
    init_qvel = jnp.zeros(9)
    init_state = jnp.concatenate([init_qpos, init_qvel])


    tree, _ = rrtree.add_nodes(tree, init_state, jnp.zeros(sim_params.action_dims), -1, 0.0, 1)
    _ = jit_while(tree, sst_params, sim_params, callables, obstacles, 0)
    print("Compilation Complete.")
    times, iters, sizes = [], [], []
    # --- Run Loop ---
    for i in range(10):
        gc.collect()
        tree = rrtree.KinoTree.init(MAX_TREE_SIZE, sim_params.dims, sim_params.action_dims)
        tree, _ = rrtree.add_nodes(tree, init_state, jnp.zeros(1), -1, 0.0, 1)
        
        start_t = time.perf_counter()
        res = jit_while(tree, sst_params, sim_params, callables, obstacles, i)
        tree, _, goal_mask, count, _, start_idx, iter_num = jax.block_until_ready(res)
        duration = time.perf_counter() - start_t

        times.append(duration); iters.append(iter_num); sizes.append(tree.tree_size)
        
        print(f"Run {i} | Goal: {count > 0} | Iters: {iter_num} | Time: {duration:.3f}s | Nodes: {tree.tree_size}")

    # Statistics (Consistent with original script)
    times, iters, sizes = jnp.array(times), jnp.array(iters), jnp.array(sizes)
    print(f"\nAverage time over {len(times)} runs: {jnp.mean(times)*1e3:.3f} ms, {jnp.mean(iters):.2f} iterations, size {jnp.mean(sizes):.2f}")
    print(f"min time: {jnp.min(times)*1e3:.3f} ms, max time: {jnp.max(times)*1e3:.3f} ms")