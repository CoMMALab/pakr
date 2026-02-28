import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
import time
import rrtree
import helper
from vine.pbd_vine import step_vine_batched
from params import Callables, Position
from vine.nns_usage import solve as find_actuator_params, solve_fwd as actuator_params_fwd_
from vine.load_env import load_box_config
# ------------------------------------------------------------------
# VINE DYNAMICS (DOUBLE PENDULUM)
# ------------------------------------------------------------------
find_actuator_params = jax.vmap(find_actuator_params, in_axes=(None, None, 0))
find_actuator_params = jax.jit(find_actuator_params, static_argnames=('predict', 'params'))
forward = jax.jit(step_vine_batched, static_argnames=['params', 'x0_list', 'y0_list', 'heading0_list', 'bend_energy_func'])
actuator_params_fwd = jax.vmap(lambda a, b, c: actuator_params_fwd_(predict, act_params, a, b, c), in_axes=(0, 0, 0))

@partial(jax.jit, static_argnums=(0, 1, 2))
def rollout_jit(sst_params, simparams, batch_size, 
                curr_time, cspace, bodies, bending_control, obstacles):
    """
    JIT-friendly version of rollout using jax.lax.scan.
    """
    steps_to_iter = sst_params.time_to_evolve
    init_x, init_y, init_heading = sst_params.start.x, sst_params.start.y, sst_params.start.z
    # Note: 'record_every' must be a static constant or passed as a static argument
    record_every = int(sst_params.δs // (simparams.grow_rate * simparams.dt))

    def scan_fn(carry, step_idx):
        cspace, bodies, curr_time = carry
        
        # Check which elements have already hit or exceeded the limit
        # We use max_bodies - 1 as the boundary per original logic
        reached_max = bodies >= simparams.max_bodies - 1
        
        # Call the existing JITed forward propagator
        next_cspace, next_bodies = forward(
            simparams, cspace, bodies, bending_control,
            init_x, init_y, init_heading, actuator_params_fwd, obstacles
        )
        
        # Update states only if they haven't reached the limit
        # This effectively "freezes" the vine once it reaches max_bodies
        cspace = jnp.where(reached_max[..., None], cspace, next_cspace)
        bodies = jnp.where(reached_max, bodies, next_bodies)
        curr_time = curr_time + simparams.dt
        
        new_carry = (cspace, bodies, curr_time)
        
        # Logic for recording at specific intervals
        # lax.scan returns the state at every step; we can slice the output later
        return new_carry, new_carry

    # Initial carry state
    init_state = (cspace, bodies, curr_time)
    
    # Run the loop for the fixed number of steps
    _, history = jax.lax.scan(scan_fn, init_state, jnp.arange(steps_to_iter))
    
    # history contains (cspace_record, bodies_record, time_record) for every step
    c_hist, b_hist, t_hist = history
    
    # Subsample the history to match your 'record_every' requirement
    # JAX supports dynamic slicing, but it's often easier to return the full 
    # history or use a specific step-based slice
    cspace_record = c_hist[::record_every]
    bodies_record = b_hist[::record_every]
    time_record = t_hist[::record_every]
    
    return cspace_record, bodies_record, time_record

@partial(jax.jit, static_argnums=(0,))
def cspace_to_tip(sim_params, cspace, n_bodies, x0, y0, h0):
    """
    Computes the (x, y, theta) of the vine tip.
    
    Args:
        sim_params: The VineParams object.
        cspace: Array of shape (max_bodies + 1,) containing [angles..., tip_length].
        n_bodies: Integer/Scalar, number of fully formed segments.
        x0, y0, h0: The anchor pose (start of the vine).
    """
    # 1. Extract the angles and the growing tip length
    angles = cspace[:sim_params.max_bodies]
    tip_len = cspace[sim_params.max_bodies]
    
    # 2. Define the segment-by-segment propagation
    def body_fn(carry, i):
        x, y, h = carry
        angle = angles[i]
        
        # Determine segment length: 
        # If i < n_bodies: it's a full segment (sim_params.body_length)
        # If i == n_bodies: it's the growing tip (tip_len)
        # If i > n_bodies: it's a ghost segment (0 length)
        is_full = i < n_bodies
        is_tip = i == n_bodies
        
        length = jnp.where(is_full, sim_params.body_length, 
                 jnp.where(is_tip, tip_len, 0.0))
        
        # Standard kinematics update
        new_h = h + angle
        new_x = x + length * jnp.cos(new_h)
        new_y = y + length * jnp.sin(new_h)
        
        return (new_x, new_y, new_h), None

    # 3. Run the kinematics chain
    final_pose, _ = jax.lax.scan(
        body_fn, 
        (x0, y0, h0), 
        jnp.arange(sim_params.max_bodies)
    )
    
    return jnp.stack(final_pose) # Returns [x_tip, y_tip, h_tip]

@jax.jit
def dist_vine(sim_params, diff):
    # diff: [dtheta1, dtheta2, ddtheta1, ddtheta2]
    dq = diff[..., 0:2]
    dv = diff[..., 2]
    # Simple weighted Euclidean
    return (jnp.sum(dq**2, axis=-1) + 0.1 * jnp.sum(dv**2, axis=-1)).T

@partial(jax.jit, static_argnums=(0,))
def sample_3D_state(sst_params, key):
    batch_size = sst_params.batch_size
    k1, k2 = jax.random.split(key) # Split keys for independence
    
    pos = jax.random.uniform(k1, (batch_size, 2), 
                             minval=jnp.array([sst_params.min_x, sst_params.min_y]), 
                             maxval=jnp.array([sst_params.max_x, sst_params.max_y]))
    
    theta = jax.random.uniform(k2, (batch_size, 1), 
                               minval=-jnp.pi, maxval=jnp.pi)

    return jnp.concatenate([pos, theta], axis=-1)

@jax.jit
def reached_goal_vine(states, goal, radius):
    return jnp.linalg.norm(states[:, 0:2] - jnp.array([goal.x, goal.y]), axis=1) < radius

# ------------------------------------------------------------------
# PLANNER SETUP
# ------------------------------------------------------------------

MAX_TREE_SIZE = 500000
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
    # K is number of parents, A is actions per parent
    B = sim_params.batch_size
    K = B // A 
    
    rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)
    
    # 1. Selection
    seed_pts = callables.sample_fn(sst_params, subkey1)[:K]
    branch_idx = jnp.minimum(jnp.digitize(tree.tree_size, jnp.array(TIERS)), len(TIERS) - 1)
    parents_small = jax.lax.switch(branch_idx, NN_BRANCHES, (tree.states, tree.tree_size, seed_pts)) 
    
    # Repeat parents for each action
    parents = jnp.repeat(parents_small, A, axis=0) 
    start_states_full = tree.states[parents]
    
    # 2. Extract state components
    cspace_start = start_states_full[:, :sim_params.max_bodies + 1]
    bodies_start = start_states_full[:, sim_params.max_bodies + 1].astype(jnp.int32)
    time_start   = tree.costs[parents]
    
    # 3. Action Sampling
    rng_key, angle_key = jax.random.split(rng_key)
    new_bend_angles = jax.random.uniform(angle_key, (B,), minval=-3.33, maxval=3.33)
    p, l0 = find_actuator_params(predict, act_params, 1.0 / new_bend_angle) 
    
    actions = tree._bending_controls[propagate_origin_idx]
    
    for idx in range(batch_size):
        actions[idx, current_bodies[idx]:, 0] = p[idx]        
        actions[idx, current_bodies[idx]:, 1] = l0[idx]

    # 4. Propagation (Black Box JIT Rollout)
    # We ignore the path records and just take the final state
    c_rec, b_rec, t_rec = rollout_jit(
        sst_params, sim_params, B, 
        time_start, cspace_start, bodies_start, actions, obstacles
    )
    
    cspace_end = c_rec[-1]
    bodies_end = b_rec[-1]
    time_end   = t_rec[-1]
    
    # 5. Cost calculation: Delta Time
    # SST Cost = Parent Cost + Time Elapsed
    edge_costs = time_end

    # 6. Re-assemble flat state
    # Format: [cspace, n_bodies, total_time]
    states_end = jnp.concatenate([
        cspace_end, 
        bodies_end[:, None].astype(jnp.float32), 
    ], axis=-1)

    # 7. Direct Addition to Tree
    # We no longer mask or filter; all B elements are added
    tree, start_idx = rrtree.add_nodes(
        tree, 
        states_end, 
        actions, 
        parents, 
        total_costs, 
        B # Adding the full batch
    )
    
    # 8. Goal check
    tips = cspace_to_tip_jit(
        sim_params, B, cspace_end, bodies_end, 
        sst_params.start.x, sst_params.start.y, sst_params.start.z
    )
    goal_mask = callables.goal_fn(tips, sst_params.goal, sst_params.goal_radius)
    
    return tree, rng_key, goal_mask, jnp.sum(goal_mask), states_end, start_idx

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


from dataclasses import dataclass
import jax.numpy as jnp

@dataclass(frozen=True)
class VineParams:
    batch_size: int
    max_bodies: int
    dims: int
    action_dims: int
    body_length: float
    radius: float
    dt: float
    grow_rate: float
    grow_force: float
    stiffness: float
    damping: float
    substeps: int
    alpha: float

@dataclass(frozen=True)
class SSTparams:
    batch_size: int
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    start: Position
    goal: Position
    goal_radius: float
    time_to_evolve: int = 100

if __name__ == "__main__":
    cfg = load_box_config('vine/envs/env_live.txt')

    batch_size = 1024
    A = 32
    max_bodies = 30
    sim_params = VineParams(
        batch_size=batch_size,
        max_bodies=max_bodies,
        dims=max_bodies + 2, # cspace + n_bodies + time
        action_dims=max_bodies, # bending control for each body
        body_length=68.0, # 25.0 mm
        radius=50, # 16.0,
        dt=1/10,
        grow_rate=20.0,
        grow_force=15.0,
        stiffness=50.0,
        damping=50.0,
        # Curiously, decreasing substeps helps prevent penetration bugs. But it doesn't fix the root problem
        substeps=15, # FIXME THIS NUMBER CAN BE MUCH SMALLER IF WE DO LANGRANGE PROPERRLY
        alpha=1e-2,
    )
    
    obstacles=cfg['obstacles'],
    # SST params
    sst_params = SSTparams(
        batch_size=1024,
        min_x=0.0,
        max_x=float(cfg['bound_x']),
        min_y=0.0,
        max_y=float(cfg['bound_y']),
        # USE TUPLES HERE. Lists [x, y, z] are unhashable.
        start=Position(float(cfg['start'][0]), float(cfg['start'][1]), float(cfg['start'][2])),
        goal=Position(float(cfg['goal'][0]), float(cfg['goal'][1]), float(cfg['goal'][2])),
        goal_radius=float(cfg['goal_radius']),
    )

    callables = Callables(
        prop_fn=None, valid_fn=None, sample_fn=sample_3D_state,
        dist_fn=dist_vine, sampact_fn=None, goal_fn=reached_goal_vine
    )

    SIM_PARAMS_RESERVED, CALLABLES_RESERVED = sim_params, callables
    obstacles = jnp.array([])



    init_state = jnp.zeros(sim_params.max_bodies + 2) # cspace + n_bodies + time
    init_state = init_state.at[0].set(sst_params.start.x) # n_bodies = 0
    init_state = init_state.at[1].set(sst_params.start.y)
    init_state = init_state.at[2].set(sst_params.start.z)


    dummy_tree = rrtree.KinoTree.init(MAX_TREE_SIZE, sim_params.dims, sim_params.action_dims)
    tree, _ = rrtree.add_nodes(dummy_tree, init_state, jnp.zeros(sim_params.action_dims), -1, 0.0, 1)
    print("\nStarting Vine RRT - Pre-compiling...")

    _ = jit_while(dummy_tree, sst_params, sim_params, callables, obstacles, 0)
    print("Compilation complete.\n")

    times, iters, sizes, costs = [], [], [], []

    for i in range(100):

        tree = rrtree.KinoTree.init(MAX_TREE_SIZE, sim_params.dims, sim_params.action_dims)
        tree, _ = rrtree.add_nodes(tree, init_state, jnp.zeros(sim_params.action_dims), -1, 0.0, 1)
        
        start_p = time.perf_counter()
        result = jit_while(tree, sst_params, sim_params, callables, obstacles, i)
        tree, key, goal_mask, goal_found, states, start_idx, iter_val, size = jax.block_until_ready(result)
        timer = time.perf_counter() - start_p


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