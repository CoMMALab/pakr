import jax
import jax.numpy as jnp
import mujoco.mjx as mjx
from functools import partial


def propagate_double_integrator(states, actions, dt, constants):
    """
    Batched dynamics for 3D double integrator.
    states:  (batch, 6) = [x, y, z, vx, vy, vz]
    actions: (batch, 3) = [ax, ay, az]
    """
    # Split state into pos and vel blocks
    pos = states[:, :3]     # (batch, 3)
    vel = states[:, 3:]     # (batch, 3)
    
    # Integrate
    new_pos = pos + vel * dt + 0.5 * actions * dt**2
    new_vel = vel + actions * dt
    
    return jnp.concatenate([new_pos, new_vel], axis=-1)


def propagate_dubins_airplane(states, actions, dt, constants):
    """
    Batched RK4 dynamics for Dubins airplane.
    Directly returns next states instead of derivatives.

    Args:
        states:  (batch, 6) [x, y, z, yaw, pitch, v]
        actions: (batch, 3) [a, yaw_rate, pitch_rate]
        dt:      scalar timestep

    Returns:
        next_states: (batch, 6)
    """

    def dynamics(state, action):
        x, y, z, yaw, pitch, v = state
        a, yaw_rate, pitch_rate = action

        dx = v * jnp.cos(pitch) * jnp.cos(yaw)
        dy = v * jnp.cos(pitch) * jnp.sin(yaw)
        dz = v * jnp.sin(pitch)
        dyaw = yaw_rate
        dpitch = pitch_rate
        dv = a

        return jnp.stack([dx, dy, dz, dyaw, dpitch, dv], axis=-1)

    # RK4 integration — directly computes next state
    k1 = dynamics(states, actions)
    k2 = dynamics(states + 0.5 * dt * k1, actions)
    k3 = dynamics(states + 0.5 * dt * k2, actions)
    k4 = dynamics(states + dt * k3, actions)

    next_states = states + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return next_states


def propagate_quadcopter(states, actions, dt, constants):
    """
    Batched dynamics for quadcopter using RK4 integration.
    Matches CUDA implementation more closely.
    """
    def dynamics(states, actions, constants):
        # Unpack states
        x, y, z      = states[:, 0], states[:, 1], states[:, 2]
        phi, theta, psi = states[:, 3], states[:, 4], states[:, 5]
        u, v, w      = states[:, 6], states[:, 7], states[:, 8]
        p, q, r      = states[:, 9], states[:, 10], states[:, 11]

        # Unpack actions
        thrust = actions[:, 0]
        tau = actions[:, 1:]  # (batch, 3)

        # Precompute trig
        cphi, sphi = jnp.cos(phi), jnp.sin(phi)
        ctheta, stheta = jnp.cos(theta), jnp.sin(theta)
        cpsi, spsi = jnp.cos(psi), jnp.sin(psi)

        # Position derivatives (world frame)
        xdot = ctheta*cpsi*u + (sphi*stheta*cpsi - cphi*spsi)*v + (cphi*stheta*cpsi + sphi*spsi)*w
        ydot = ctheta*spsi*u + (sphi*stheta*spsi + cphi*cpsi)*v + (cphi*stheta*spsi - sphi*cpsi)*w
        zdot = -stheta*u + sphi*ctheta*v + cphi*ctheta*w

        # Euler angle rates (nonlinear)
        phidot   = p + (q*sphi + r*cphi) * jnp.tan(theta)
        thetadot = q*cphi - r*sphi
        psidot   = (q*sphi + r*cphi) / ctheta

        # Aerodynamic drag
        XYZ = -constants.NU * jnp.sqrt(u*u + v*v + w*w)
        LMN = -constants.MU * jnp.sqrt(p*p + q*q + r*r)

        # Linear accelerations (body frame)
        udot = (r*v - q*w) - constants.g * jnp.sin(theta) + constants.MASS_INV * XYZ * u
        vdot = (p*w - r*u) + constants.g * jnp.cos(theta) * jnp.sin(phi) + constants.MASS_INV * XYZ * v
        wdot = (q*u - p*v) + constants.g * jnp.cos(theta) * jnp.cos(phi) + constants.MASS_INV * XYZ * w + constants.MASS_INV * thrust

        # Angular accelerations
        pdot = (constants.IY - constants.IZ)/constants.IX * q*r + LMN/ constants.IX * p + tau[:, 0] / constants.IX
        qdot = (constants.IZ - constants.IX)/constants.IY * p*r + LMN/ constants.IY * q + tau[:, 1] / constants.IY
        rdot = (constants.IX - constants.IY)/constants.IZ * p*q + LMN/ constants.IZ * r + tau[:, 2] / constants.IZ

        # Stack results → shape (batch, 12)
        return jnp.concatenate(
            [xdot, ydot, zdot,
            phidot, thetadot, psidot,
            udot, vdot, wdot,
            pdot, qdot, rdot],
            axis=-1
        )

    # This works batched directly — no need for vmap


    # RK4 integration
    k1 = dynamics(states, actions, constants)
    k2 = dynamics(states + 0.5 * dt * k1, actions, constants)
    k3 = dynamics(states + 0.5 * dt * k2, actions, constants)
    k4 = dynamics(states + dt * k3, actions, constants)

    next_states = states + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return next_states


@partial(jax.jit, static_argnums=(3, 4, 5))
def rollout(state0, actions, obstacles, sst_params, sim_params, callables):
    """
    Rollout trajectories given actions.

    Returns:
        states_seq     : (T, batch, dims) - propagated states at each timestep
        kill           : (batch,) - first timestep each trajectory becomes invalid
        dist_traveled  : (T, batch) - per-step distance traveled for each trajectory
    """
    steps = sst_params.time_to_evolve
    dt = sim_params.dt
    prop_fn = callables.prop_fn
    valid_fn = callables.valid_fn

    batch = state0.shape[0]
    kill = jnp.full((batch,), steps, dtype=jnp.int32)
    dist_traveled = jnp.zeros((steps, batch), dtype=jnp.float32)
    actions = jnp.tile(actions[None, :, :], (steps, 1, 1))  # (steps, batch, action_dims)

    def step_fn(carry, actions_t):
        states, kill, t, dist_traveled = carry

        # 1. Propagate forward one step (batched)
        next_states = prop_fn(states, actions_t, dt, sim_params.physics_constants)

        # 2. Compute batched distance between states
        dist_t = jnp.linalg.norm(next_states - states, axis=-1)  # (batch,)
        dist_traveled = dist_traveled.at[t].set(dist_t)

        # 3. Check which trajectories become invalid here
        is_invalid = ~valid_fn(next_states, sim_params, obstacles)

        # 4. Record first invalid timestep
        kill = jnp.where((is_invalid) & (kill == steps), t, kill)

        return (next_states, kill, t + 1, dist_traveled), (next_states)

    # Run scan over timesteps
    (_, kill, _, dist_traveled), (states_seq) = jax.lax.scan(
        step_fn,
        init=(state0, kill, 0, dist_traveled),
        xs=actions
    )

    # states_seq: (T, batch, dims)
    # dist_traveled: (T, batch)
    # kill: (batch,)
    return states_seq, kill, dist_traveled



#####################
# MJX rollout
#####################

def sample(params, key): # batched
    """
    Samples random direction, converts to unit vector dir, samples random speed
    Returns (batch_size, 3) = [x, y, vel].
    """
    key_angle, key_vel = jax.random.split(key)
    angles = jax.random.uniform(key_angle, (params.batch_size, 1), minval=0.0, maxval=1.0)
    angles = angles * 2.0 * jnp.pi
    x = jnp.cos(angles)
    y = jnp.sin(angles)
    vel = jax.random.uniform(key_vel, (params.batch_size, 1), minval=1e-8, maxval=5.0)
    
    return jnp.hstack([x, y, vel])

def single_rollout(model, data0, action, act_x, act_y, horizon=100):
    direction = jnp.array([action[0], action[1]])
    norm = jnp.linalg.norm(direction) + 1e-8
    unit_dir = direction / norm
    velocity = unit_dir * action[2]
    ctrl = jnp.zeros(model.nu).at[act_x].set(velocity[0]).at[act_y].set(velocity[1])
    def body_fn(data, _):
        data_next = mjx.step(model, data, ctrl)
        return data_next, data_next  # carry, output
    _, traj = jax.lax.scan(body_fn, data0, None, length=horizon)
    return traj

batched_rollout = jax.vmap(single_rollout, in_axes=(None, None, 0, None, None))