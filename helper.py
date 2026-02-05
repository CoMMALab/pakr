import jax
import jax.numpy as jnp
import pandas as pd
from functools import partial

##########################
# math functions regarding rotations
##########################

@jax.jit
def wrap_rotation(diff):
    """
    Wrap the angle into [-pi, pi).
    """
    return (diff + jnp.pi) % (2 * jnp.pi) - jnp.pi

@jax.jit
def shortest_rotation(angle1, angle2):
    diff = angle2 - angle1
    return wrap_rotation(diff)

###################################
# State Validation for different motions
###################################
@jax.jit
def collision_check(points, obstacles):
    """
    Batched collision check: points vs axis-aligned 3D boxes.

    Args:
        points: shape (B, 3)  -> [x, y, z] for each point
        obstacles: shape (O, 6) -> [x1, y1, z1, x2, y2, z2] per obstacle

    Returns:
        valid_mask: shape (B,) bool array, True if point is NOT in any obstacle
    """
    # Ensure x1 <= x2, y1 <= y2, z1 <= z2
    mins = jnp.minimum(obstacles[:, :3], obstacles[:, 3:6])  # (O,3)
    maxs = jnp.maximum(obstacles[:, :3], obstacles[:, 3:6])  # (O,3)

    # points shape: (B,1,3), mins/maxs shape: (1,O,3) for broadcasting
    points_exp = points[:, None, :]   # (B,1,3)
    mins_exp = mins[None, :, :]       # (1,O,3)
    maxs_exp = maxs[None, :, :]       # (1,O,3)

    # Check if points are inside each box along all axes
    inside = jnp.all((points_exp >= mins_exp) & (points_exp <= maxs_exp), axis=-1)  # (B,O)

    # A point is invalid if it is inside **any** box
    invalid_mask = jnp.any(inside, axis=-1)  # (B,)

    # Return valid mask (True if NOT colliding)
    valid_mask = ~invalid_mask
    return valid_mask

@partial(jax.jit, static_argnums=(1))
def valid_DI(state, params, obstacles):
    """
    double integrator
    state[x, y, z, vx, vy, vz]
    """
    x, y, z, vx, vy, vz = state.T
    within_bounds = (x >= params.bounds.min_x) & (x <= params.bounds.max_x) & \
                    (y >= params.bounds.min_y) & (y <= params.bounds.max_y) & \
                    (z >= params.bounds.min_z) & (z <= params.bounds.max_z) & \
                    (vx >= params.motion_constraints.min_vel) & (vx <= params.motion_constraints.max_vel) & \
                    (vy >= params.motion_constraints.min_vel) & (vy <= params.motion_constraints.max_vel) & \
                    (vz >= params.motion_constraints.min_vel) & (vz <= params.motion_constraints.max_vel)
    collision_free = collision_check(state[:, :3], obstacles)
    return within_bounds & collision_free
   
@partial(jax.jit, static_argnums=(1))
def valid_DA(state, params, obstacles):
    """
    dubins airplane
    state[x, y, z, yaw, pitch, v]
    """
    x, y, z, yaw, pitch, v = state.T
    within_bounds = (x >= params.bounds.min_x) & (x <= params.bounds.max_x) & \
                    (y >= params.bounds.min_y) & (y <= params.bounds.max_y) & \
                    (z >= params.bounds.min_z) & (z <= params.bounds.max_z) & \
                    (pitch >= params.motion_constraints.min_pitch) & (pitch <= params.motion_constraints.max_pitch) & \
                    (v >= params.motion_constraints.min_vel) & (v <= params.motion_constraints.max_vel)
    collision_free = collision_check(state[:, :3], obstacles)
    return within_bounds & collision_free
  
@partial(jax.jit, static_argnums=(1))
def valid_QC(state, params, obstacles):
    """
    quadcopter
    state[x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
    """
    x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz = state.T
    within_bounds = (x >= params.bounds.min_x) & (x <= params.bounds.max_x) & \
                    (y >= params.bounds.min_y) & (y <= params.bounds.max_y) & \
                    (z >= params.bounds.min_z) & (z <= params.bounds.max_z) & \
                    (vx >= params.motion_constraints.min_vel) & (vx <= params.motion_constraints.max_vel) & \
                    (vy >= params.motion_constraints.min_vel) & (vy <= params.motion_constraints.max_vel) & \
                    (vz >= params.motion_constraints.min_vel) & (vz <= params.motion_constraints.max_vel) & \
                    (wx >= params.motion_constraints.min_angle_vel) & (wx <= params.motion_constraints.max_angle_vel) & \
                    (wy >= params.motion_constraints.min_angle_vel) & (wy <= params.motion_constraints.max_angle_vel) & \
                    (wz >= params.motion_constraints.min_angle_vel) & (wz <= params.motion_constraints.max_angle_vel)
    collision_free = collision_check(state[:, :3], obstacles)
    return within_bounds & collision_free
    
###################################
# Sampling functions for different motions
# samplers for states and actions
###################################

@partial(jax.jit, static_argnums=(0))
def sample_DI(sim_params, key):
    """
    double integrator
    state[x, y, z, vx, vy, vz]
    """
    num_samples = sim_params.batch_size
    key_x, key_y, key_z, key_vx, key_vy, key_vz = jax.random.split(key, 6)
    x = jax.random.uniform(key_x, (num_samples, 1), minval=sim_params.bounds.min_x, maxval=sim_params.bounds.max_x)
    y = jax.random.uniform(key_y, (num_samples, 1), minval=sim_params.bounds.min_y, maxval=sim_params.bounds.max_y)
    z = jax.random.uniform(key_z, (num_samples, 1), minval=sim_params.bounds.min_z, maxval=sim_params.bounds.max_z)
    vx = jax.random.uniform(key_vx, (num_samples, 1), minval=sim_params.motion_constraints.min_vel, maxval=sim_params.motion_constraints.max_vel)
    vy = jax.random.uniform(key_vy, (num_samples, 1), minval=sim_params.motion_constraints.min_vel, maxval=sim_params.motion_constraints.max_vel)
    vz = jax.random.uniform(key_vz, (num_samples, 1), minval=sim_params.motion_constraints.min_vel, maxval=sim_params.motion_constraints.max_vel)
    samples = jnp.concatenate([x, y, z, vx, vy, vz], axis=-1)
    return samples

@partial(jax.jit, static_argnums=(0))
def sample_DA(sim_params, key):
    """
    dubins airplane
    state[x, y, z, yaw, pitch, v]
    """
    num_samples = sim_params.batch_size
    key_x, key_y, key_z, key_yaw, key_pitch, key_v = jax.random.split(key, 6)
    x = jax.random.uniform(key_x, (num_samples, 1), minval=sim_params.bounds.min_x, maxval=sim_params.bounds.max_x)
    y = jax.random.uniform(key_y, (num_samples, 1), minval=sim_params.bounds.min_y, maxval=sim_params.bounds.max_y)
    z = jax.random.uniform(key_z, (num_samples, 1), mminval=sim_params.bounds.min_z, maxval=sim_params.bounds.max_z)
    yaw = jax.random.uniform(key_yaw, (num_samples, 1), minval=sim_params.motion_constraints.min_yaw, maxval=sim_params.motion_constraints.max_yaw)
    pitch = jax.random.uniform(key_pitch, (num_samples, 1), minval=sim_params.motion_constraints.min_pitch, maxval=sim_params.motion_constraints.max_pitch)
    v = jax.random.uniform(key_v, (num_samples, 1), minval=sim_params.motion_constraints.min_vel, maxval=sim_params.motion_constraints.max_vel)
    samples = jnp.concatenate([x, y, z, yaw, pitch, v], axis=-1)
    return samples

@partial(jax.jit, static_argnums=(0))
def sample_QC(sim_params, key):
    """
    quadcopter
    state[x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
    """
    num_samples = sim_params.batch_size
    keys = jax.random.split(key, 13)
    x = jax.random.uniform(keys[0], (num_samples, 1), minval=sim_params.bounds.min_x, maxval=sim_params.bounds.max_x)
    y = jax.random.uniform(keys[1], (num_samples, 1), minval=sim_params.bounds.min_y, maxval=sim_params.bounds.max_y)
    z = jax.random.uniform(keys[2], (num_samples, 1), minval=sim_params.bounds.min_z, maxval=sim_params.bounds.max_z)
    vx = jax.random.uniform(keys[3], (num_samples, 1), minval=sim_params.motion_constraints.min_vel, maxval=sim_params.motion_constraints.max_vel)
    vy = jax.random.uniform(keys[4], (num_samples, 1), minval=sim_params.motion_constraints.min_vel, maxval=sim_params.motion_constraints.max_vel)
    vz = jax.random.uniform(keys[5], (num_samples, 1), minval=sim_params.motion_constraints.min_vel, maxval=sim_params.motion_constraints.max_vel)
    roll = jax.random.uniform(keys[6], (num_samples, 1), minval=sim_params.motion_constraints.min_roll, maxval=sim_params.motion_constraints.max_roll)
    pitch = jax.random.uniform(keys[7], (num_samples, 1), minval=sim_params.motion_constraints.min_pitch, maxval=sim_params.motion_constraints.max_pitch)
    yaw = jax.random.uniform(keys[8], (num_samples, 1), minval=sim_params.motion_constraints.min_yaw, maxval=sim_params.motion_constraints.max_yaw)
    wx = jax.random.uniform(keys[9], (num_samples, 1), minval=sim_params.motion_constraints.min_angle_vel, maxval=sim_params.motion_constraints.max_angle_vel)
    wy = jax.random.uniform(keys[10], (num_samples, 1), minval=sim_params.motion_constraints.min_angle_vel, maxval=sim_params.motion_constraints.max_angle_vel)
    wz = jax.random.uniform(keys[11], (num_samples, 1), minval=sim_params.motion_constraints.min_angle_vel, maxval=sim_params.motion_constraints.max_angle_vel)
    samples = jnp.concatenate([x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz], axis=-1)
    return samples

@partial(jax.jit, static_argnums=(0))
def sample_actions_DI(sim_params, key):
    """
    double integrator
    action[ax, ay, az]
    """
    num_samples = sim_params.batch_size
    key_ax, key_ay, key_az = jax.random.split(key, 3)
    ax = jax.random.uniform(key_ax, (num_samples, 1), minval=sim_params.motion_constraints.min_accel, maxval=sim_params.motion_constraints.max_accel)
    ay = jax.random.uniform(key_ay, (num_samples, 1), minval=sim_params.motion_constraints.min_accel, maxval=sim_params.motion_constraints.max_accel)
    az = jax.random.uniform(key_az, (num_samples, 1), minval=sim_params.motion_constraints.min_accel, maxval=sim_params.motion_constraints.max_accel)
    actions = jnp.concatenate([ax, ay, az], axis=-1)
    return actions

@partial(jax.jit, static_argnums=(0))
def sample_actions_DA(sim_params, key):
    """
    dubins airplane
    action[a, yaw_rate, pitch_rate]
    """
    num_samples = sim_params.batch_size
    key_a, key_yaw_rate, key_pitch_rate = jax.random.split(key, 3)
    a = jax.random.uniform(key_a, (num_samples, 1), minval=sim_params.motion_constraints.min_accel, maxval=sim_params.motion_constraints.max_accel)
    yaw_rate = jax.random.uniform(key_yaw_rate, (num_samples, 1), minval=sim_params.motion_constraints.min_yaw_rate, maxval=sim_params.motion_constraints.max_yaw_rate)
    pitch_rate = jax.random.uniform(key_pitch_rate, (num_samples, 1), minval=sim_params.motion_constraints.min_pitch_rate, maxval=sim_params.motion_constraints.max_pitch_rate)
    actions = jnp.concatenate([a, yaw_rate, pitch_rate], axis=-1)
    return actions

@partial(jax.jit, static_argnums=(0))
def sample_actions_QC(sim_params, key):
    """
    quadcopter
    action[thrust, roll_rate, pitch_rate, yaw_rate]
    """
    num_samples = sim_params.batch_size
    keys = jax.random.split(key, 5)
    thrust = jax.random.uniform(keys[0], (num_samples, 1), minval=sim_params.motion_constraints.min_thrust, maxval=sim_params.motion_constraints.max_thrust)
    roll_rate = jax.random.uniform(keys[1], (num_samples, 1), minval=sim_params.motion_constraints.min_torque, maxval=sim_params.motion_constraints.max_torque)
    pitch_rate = jax.random.uniform(keys[2], (num_samples, 1), minval=sim_params.motion_constraints.min_torque, maxval=sim_params.motion_constraints.max_torque)
    yaw_rate = jax.random.uniform(keys[3], (num_samples, 1), minval=sim_params.motion_constraints.min_torque, maxval=sim_params.motion_constraints.max_torque)
    actions = jnp.concatenate([thrust, roll_rate, pitch_rate, yaw_rate], axis=-1)
    return actions

###################################
# Difference functions for different motions
###################################

@jax.jit
def normalize(lower, upper, diff):
    # normalizes to bounds
    range = upper - lower
    return diff / range

@partial(jax.jit, static_argnums=(0,))
def dist_DI(sim_params, diff):
    """
    double integrator
    state[x, y, z, vx, vy, vz]
    """
    dx, dy, dz, dvx, dvy, dvz = diff.T
    dx = normalize(sim_params.bounds.min_x, sim_params.bounds.max_x, dx)
    dy = normalize(sim_params.bounds.min_y, sim_params.bounds.max_y, dy)
    dz = normalize(sim_params.bounds.min_z, sim_params.bounds.max_z, dz)
    dvx = normalize(sim_params.motion_constraints.min_vel, sim_params.motion_constraints.max_vel, dvx)
    dvy = normalize(sim_params.motion_constraints.min_vel, sim_params.motion_constraints.max_vel, dvy)
    dvz = normalize(sim_params.motion_constraints.min_vel, sim_params.motion_constraints.max_vel, dvz)
    dist2 = dx**2 + dy**2 + dz**2 + dvx**2 + dvy**2 + dvz**2
    return dist2 / 6

@partial(jax.jit, static_argnums=(0))
def dist_DA(sim_params, diff):
    """
    dubins airplane
    state[x, y, z, yaw, pitch, v]
    """
    # Compute shortest rotation for the yaw component [-pi, pi)
    yaw_delta = wrap_rotation(diff[..., 3])
    pitch_delta = wrap_rotation(diff[..., 4])
    
    # Normalize other components to [0, 1]
    dx = normalize(sim_params.bounds.min_x, sim_params.bounds.max_x, diff[..., 0])
    dy = normalize(sim_params.bounds.min_y, sim_params.bounds.max_y, diff[..., 1])
    dz = normalize(sim_params.bounds.min_z, sim_params.bounds.max_z, diff[..., 2])
    dv = normalize(sim_params.motion_constraints.min_vel, sim_params.motion_constraints.max_vel, diff[..., 5])
    yaw_delta = normalize(sim_params.motion_constraints.min_yaw, sim_params.motion_constraints.max_yaw, yaw_delta)
    pitch_delta = normalize(sim_params.motion_constraints.min_pitch, sim_params.motion_constraints.max_pitch, pitch_delta)
    
    # Compute pairwise distances
    dist2 = dx**2 + dy**2 + dz**2 + yaw_delta**2 + pitch_delta**2 + dv**2
    return dist2 / 6

@partial(jax.jit, static_argnums=(0))
def dist_QC(sim_params, diff):
    """
    quadcopter
    state[x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
    """
    dx, dy, dz, dvx, dvy, dvz, droll, dpitch, dyaw, dwx, dwy, dwz = diff.T
    dx = normalize(sim_params.bounds.min_x, sim_params.bounds.max_x, dx)
    dy = normalize(sim_params.bounds.min_y, sim_params.bounds.max_y, dy)
    dz = normalize(sim_params.bounds.min_z, sim_params.bounds.max_z, dz)
    dvx = normalize(sim_params.motion_constraints.min_vel, sim_params.motion_constraints.max_vel, dvx)
    dvy = normalize(sim_params.motion_constraints.min_vel, sim_params.motion_constraints.max_vel, dvy)
    dvz = normalize(sim_params.motion_constraints.min_vel, sim_params.motion_constraints.max_vel, dvz)
    droll = wrap_rotation(droll)
    dpitch = wrap_rotation(dpitch)
    dyaw = wrap_rotation(dyaw)
    droll = normalize(sim_params.motion_constraints.min_roll, sim_params.motion_constraints.max_roll, droll)
    dpitch = normalize(sim_params.motion_constraints.min_pitch, sim_params.motion_constraints.max_pitch, dpitch)
    dyaw = normalize(sim_params.motion_constraints.min_yaw, sim_params.motion_constraints.max_yaw, dyaw)
    dwx = normalize(sim_params.motion_constraints.min_angle_vel, sim_params.motion_constraints.max_angle_vel, dwx)
    dwy = normalize(sim_params.motion_constraints.min_angle_vel, sim_params.motion_constraints.max_angle_vel, dwy)
    dwz = normalize(sim_params.motion_constraints.min_angle_vel, sim_params.motion_constraints.max_angle_vel, dwz)
    dist2 = dx**2 + dy**2 + dz**2 + dvx**2 + dvy**2 + dvz**2 + droll**2 + dpitch**2 + dyaw**2 + dwx**2 + dwy**2 + dwz**2
    return dist2 / 12
    
@partial(jax.jit, static_argnums=(0, 1))
def nearest_neighbor(sim_params, dist_fn, ref_points: jnp.ndarray, query_points: jnp.ndarray):
    """
    Returns single nearest neighbor per batch
    """
    assert ref_points.shape[1] == query_points.shape[1], f"ref_points shape: {ref_points.shape}, query_points shape: {query_points.shape}"
    diff = query_points[:, None] - ref_points
    dist2 = dist_fn(sim_params, diff)
    dist2 = dist2.T
    indices = jnp.argmin(dist2, axis=1)
    return indices, jnp.take_along_axis(dist2, indices[:, None], axis=1).flatten()

@partial(jax.jit, static_argnums=(0, 1))
def nearest_neighbor_masked(
    sim_params,
    dist_fn,
    ref_points: jnp.ndarray,     # (MAX_TREE_SIZE, dims)
    tree_size: jnp.ndarray,      # scalar int
    query_points: jnp.ndarray,   # (batch, dims)
):
    """
    Masked nearest neighbor over padded tree.

    Returns:
        indices : (batch,)
        dists   : (batch,)
    """
    # ref_points: (Nmax, D)
    # query_points: (B, D)

    B = query_points.shape[0]
    Nmax = ref_points.shape[0]

    # Compute pairwise diffs → (B, Nmax, D)
    diff = query_points[:, None, :] - ref_points[None, :, :]

    # Distance → (B, Nmax)
    dist2 = dist_fn(sim_params, diff)
    dist2 = dist2.T
    # Mask out invalid tree entries
    valid_mask = jnp.arange(Nmax) < tree_size   # (Nmax,)
    dist2 = jnp.where(valid_mask[None, :], dist2, jnp.inf)

    # Nearest neighbor
    indices = jnp.argmin(dist2, axis=1)          # (B,)
    dists = jnp.take_along_axis(
        dist2, indices[:, None], axis=1
    ).squeeze(1)

    return indices, dists

@partial(jax.jit, static_argnums=(0, 1))
def nearest_neighbor_mjx( # for some fuckin reason this doesnt need the .T idk
    sim_params,
    dist_fn,
    ref_points: jnp.ndarray,     # (MAX_TREE_SIZE, dims)
    tree_size: jnp.ndarray,      # scalar int
    query_points: jnp.ndarray,   # (batch, dims)
):
    """
    Masked nearest neighbor over padded tree.

    Returns:
        indices : (batch,)
        dists   : (batch,)
    """
    B = query_points.shape[0]
    Nmax = ref_points.shape[0]

    # Compute pairwise diffs → (B, Nmax, D)
    diff = query_points[:, None, :] - ref_points[None, :, :]  # (B, Nmax, D)

    # Compute distances → (B, Nmax)
    dist2 = dist_fn(sim_params, diff)  # should return shape (B, Nmax)

    # Mask out invalid tree entries
    valid_mask = jnp.arange(Nmax) < tree_size  # (Nmax,)
    dist2 = jnp.where(valid_mask[None, :], dist2, jnp.inf)  # broadcasts over B

    # Nearest neighbor
    indices = jnp.argmin(dist2, axis=1)  # (B,)
    dists = jnp.take_along_axis(dist2, indices[:, None], axis=1).squeeze(1)  # (B,)

    return indices, dists


@partial(jax.jit, static_argnums=(0, 1))
def dist_all(sim_params, dist_fn, ref_points: jnp.ndarray, query_points: jnp.ndarray):
    """
    Returns all distances to all reference points
    """
    assert ref_points.shape[1] == query_points.shape[1], f"ref_points shape: {ref_points.shape}, query_points shape: {query_points.shape}"
    diff = query_points[:, None] - ref_points
    dist2 = dist_fn(sim_params, diff)
    return dist2


@jax.jit
def dist2goal(nodes, goal):
    """
    Returns all distances to all reference points
    """
    diff = nodes[:, 3:] - goal
    dist2 = jnp.sum(diff ** 2, axis=-1)
    return dist2

def dist_all_no_jax(sim_params, dist_fn, ref_points, query_points):
    """
    Returns all distances to all reference points
    """
    assert ref_points.shape[1] == query_points.shape[1], f"ref_points shape: {ref_points.shape}, query_points shape: {query_points.shape}"
    diff = query_points[:, None] - ref_points
    dist2 = dist_fn(sim_params, diff)
    return dist2

@jax.jit
def reached_goal(states, goal, goal_radius):
    diff2 = jnp.sum((states[:, :3] - jnp.asarray([goal.x, goal.y, goal.z], dtype=jnp.float32)) ** 2, axis=-1)
    return diff2 < goal_radius ** 2

def get_obs(filename):
    df = pd.read_csv(filename)  # columns: x1,y1,z1,x2,y2,z2
    obs = []
    for idx, row in df.iterrows():
        obs.append(jnp.asarray(row))
    return jnp.asarray(obs)

def find_solution_path(tree, solutions):
    if len(solutions) == 0:
        print("No solutions found.")
        return []
    
    # Get the best solution (lowest cost)
    current_idx = solutions[0][2].data['parent_idx']  # (cost, counter, sol)
    
    controls = []
    states = []

    controls.append(solutions[0][2].data['action'])
    states.append(solutions[0][2].data['cspace'])
    while current_idx != -1:
        ctrl = tree._controls[current_idx]
        node = tree._c_spaces[current_idx]
        controls.append(ctrl)
        states.append(node)
        current_idx = tree._parent_idxs[current_idx]
    
    controls.pop()
    controls.reverse()  # reverse to get from start to goal
    states.reverse() 
    return jnp.asarray(controls), jnp.asarray(states)


def recreate_trajectory(state0, actions, sim_params, prop_fn):
    """
    Recreate full trajectory given compressed actions.

    Args:
        state0:   (dims,) initial state vector
        actions:  (T, act_dim + 2) where last 2 columns = [timesteps, dist_traveled]
        sim_params: struct with simulation params (e.g. dt, physics_constants)
        prop_fn:  propagation function to integrate dynamics

    Returns:
        positions: (total_steps + 1, 3) positions at each timestep (includes start)
        states:    (total_steps + 1, dims) full state sequence
    """
    dt = sim_params.dt
    state_dim = state0.shape[0]

    # Split actions into real controls and timesteps
    controls = actions[:, :-2]        # (T, act_dim)
    timesteps = actions[:, -2].astype(int)  # (T,)

    # Expand controls: repeat each control for its corresponding timesteps
    expanded_actions = jnp.repeat(controls, timesteps, axis=0)  # (total_steps, act_dim)
    total_steps = expanded_actions.shape[0]

    # Allocate storage for state trajectory
    states = jnp.zeros((total_steps + 1, state_dim))
    states = states.at[0].set(state0)

    def step_fn(state, action):
        next_state = prop_fn(state[None, :], action[None, :], dt, sim_params.physics_constants)[0]
        return next_state, next_state

    # Use lax.scan for unrolled integration
    _, next_states = jax.lax.scan(step_fn, state0, expanded_actions)

    # Combine initial + propagated states
    states = states.at[1:].set(next_states)

    # Extract positions only
    positions = states[:, :3]

    return positions, states

###########################
# MJX helpers
###########################

# -----------------------------
# Sample initial states
# -----------------------------
@partial(jax.jit, static_argnums=(0,))
def sample_FRB(sim_params, key):
    B = sim_params.batch_size
    keys = jax.random.split(key, 4)

    # ----------------------------
    # Arm joints (7 DOF)
    # ----------------------------
    q_nom = jnp.full((7,), (sim_params.motion_constraints.max_vel + sim_params.motion_constraints.min_vel)/2)
    q_range = jnp.full((7,), (sim_params.motion_constraints.max_vel - sim_params.motion_constraints.min_vel)/2)
    q = q_nom + jax.random.uniform(keys[0], (B, 7), minval=-q_range, maxval=q_range)
    dq = jnp.zeros((B, 7))  # could sample small random velocities if desired

    # ----------------------------
    # Block pose
    # ----------------------------
    xyz = jax.random.uniform(
        keys[1], (B, 3),
        minval=jnp.array([sim_params.bounds.min_x, sim_params.bounds.min_y, sim_params.bounds.min_z]),
        maxval=jnp.array([sim_params.bounds.max_x, sim_params.bounds.max_y, sim_params.bounds.max_z]),
    )

    # Correct: use quaternion (w, x, y, z) for orientation
    block_quat = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (B, 1))

    # ----------------------------
    # Block velocities
    # ----------------------------
    dxyz = jnp.zeros((B, 3))
    omega = jnp.zeros((B, 3))  # angular velocity

    # ----------------------------
    # Concatenate into 27D state
    # ----------------------------
    return jnp.concatenate([q, dq, xyz, block_quat, dxyz, omega], axis=-1)


# -----------------------------
# Sample actions (torques)
# -----------------------------
@partial(jax.jit, static_argnums=(0,))
def sample_actions_FRB(sim_params, key):
    B = sim_params.batch_size
    tau = jax.random.uniform(
        key, (B, 7),
        minval=sim_params.motion_constraints.min_torque,
        maxval=sim_params.motion_constraints.max_torque
    )
    return tau

# -----------------------------
# Distance function
# -----------------------------
@partial(jax.jit, static_argnums=(0,))
def dist_FRB(sim_params, diff):
    # diff: (..., 27)
    dq = diff[..., 0:7]
    dxyz = diff[..., 14:17]
    drpy = diff[..., 17:20]

    q_cost = jnp.sum(dq**2, axis=-1)
    xyz_cost = jnp.sum(dxyz**2, axis=-1)
    rpy_cost = jnp.sum(drpy**2, axis=-1)

    w_q, w_xyz, w_rpy = 1.0, 1.0, 0.1
    return w_q*q_cost + w_xyz*xyz_cost + w_rpy*rpy_cost

# -----------------------------
# Goal check
# -----------------------------
@jax.jit
def reached_goal_FRB(states, goal, radius):
    block_xyz = states[:, 14:17]
    diff2 = jnp.sum((block_xyz - jnp.asarray([goal.x, goal.y, goal.z]))**2, axis=-1)
    return diff2 < radius**2

# -----------------------------
# Validity check
# -----------------------------
@partial(jax.jit, static_argnums=(1,))
def valid_FRB(states, params, obstacles):
    q = states[:, :7]
    block_xyz = states[:, 14:17]

    joint_ok = jnp.all((q >= params.motion_constraints.min_vel) & (q <= params.motion_constraints.max_vel), axis=-1)
    table_ok = block_xyz[:, 2] >= 0.0

    return joint_ok & table_ok


