import jax
import jax.numpy as jnp
import mujoco.mjx as mjx
from functools import partial
import helper

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
        x, y, z, yaw, pitch, v = state.T
        a, yaw_rate, pitch_rate = action.T

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
        udot = (r*v - q*w) - constants.g * jnp.sin(theta) + 1/constants.m * XYZ * u
        vdot = (p*w - r*u) + constants.g * jnp.cos(theta) * jnp.sin(phi) + 1/constants.m * XYZ * v
        wdot = (q*u - p*v) + constants.g * jnp.cos(theta) * jnp.cos(phi) + 1/constants.m * XYZ * w + 1/constants.m * thrust

        # Angular accelerations
        pdot = (constants.IY - constants.IZ)/constants.IX * q*r + LMN/ constants.IX * p + tau[:, 0] / constants.IX
        qdot = (constants.IZ - constants.IX)/constants.IY * p*r + LMN/ constants.IY * q + tau[:, 1] / constants.IY
        rdot = (constants.IX - constants.IY)/constants.IZ * p*q + LMN/ constants.IZ * r + tau[:, 2] / constants.IZ

        # Stack results → shape (batch, 12)
        return jnp.stack(
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


@partial(jax.jit, static_argnums=(3, 4, 5))
def rollout_final(state0, actions, obstacles, sst_params, sim_params, callables):
    """
    Rollout trajectories given actions, returning only the final state, a valid mask, 
    and the total distance traveled per trajectory.

    Args:
        state0      : (batch, dims) - initial states
        actions     : (batch, action_dims) - per-trajectory actions
        obstacles   : environment obstacles
        sst_params  : parameters like time_to_evolve
        sim_params  : physics parameters including dt
        callables   : object with prop_fn and valid_fn

    Returns:
        final_states: (batch, dims) - final state of each trajectory
        valid_mask  : (batch,) - True if trajectory remained valid throughout
        dist_traveled: (batch,) - total distance traveled along valid steps
    """
    steps = sst_params.time_to_evolve
    dt = sim_params.dt
    prop_fn = callables.prop_fn
    valid_fn = callables.valid_fn

    batch = state0.shape[0]
    states = state0 # (batch, dims)
    valid_mask = jnp.ones((batch,), dtype=jnp.bool_)
    dist_traveled = jnp.zeros((batch,), dtype=jnp.float32)

    # Tile actions for all steps
    actions_seq = jnp.tile(actions[None, :, :], (steps, 1, 1))  # (steps, batch, action_dims)

    def step_fn(carry, actions_t):
        states, valid_mask, dist_traveled = carry

        # Propagate all trajectories
        next_states = prop_fn(states, actions_t, dt, sim_params.physics_constants)

        step_dist = jnp.sum((next_states[..., :3] - states[..., :3]) ** 2, axis=-1) ** 0.5

        dist_traveled = dist_traveled + step_dist

        # Update validity mask
        is_valid = valid_fn(next_states, sim_params, obstacles)
        valid_mask = valid_mask & is_valid

        return (next_states, valid_mask, dist_traveled), None

    (final_states, valid_mask, dist_traveled), _ = jax.lax.scan(
        step_fn,
        init=(states, valid_mask, dist_traveled),
        xs=actions_seq
    )

    return final_states, valid_mask, dist_traveled

@partial(jax.jit, static_argnums=(3, 4))
def rollout_final_mjx(
    data0,
    actions,
    obstacles,
    sst_params,
    sim_params,
):
    """
    MJX rollout with validity checks and distance accumulation.

    Args:
        data0    : mjx.Data (batched)
        actions  : (batch, nu)
    """

    steps = sst_params.time_to_evolve
    model = sim_params.mjx_model

    # Set control once (piecewise constant)
    data0 = data0.replace(ctrl=actions)

    batch = actions.shape[0]

    valid_mask = jnp.ones((batch,), dtype=jnp.bool_)
    dist_traveled = jnp.zeros((batch,), dtype=jnp.float32)

    def step_fn(carry, _):
        data, valid_mask, dist_traveled = carry

        prev_xpos = data.xpos[..., :3]  # (batch, bodies, 3)

        # Step physics
        data = mjx.step(model, data)

        new_xpos = data.xpos[..., :3]

        # Example: track base body displacement
        step_dist = jnp.linalg.norm(
            new_xpos[:, 0] - prev_xpos[:, 0],
            axis=-1
        )

        dist_traveled = dist_traveled + step_dist

        # Validity check
        is_valid = valid_fn_mjx(data, obstacles, sim_params)
        valid_mask = valid_mask & is_valid

        return (data, valid_mask, dist_traveled), None

    (final_data, valid_mask, dist_traveled), _ = jax.lax.scan(
        step_fn,
        init=(data0, valid_mask, dist_traveled),
        xs=None,
        length=steps,
    )

    return final_data, valid_mask, dist_traveled

import jax.numpy as jnp

def make_frb_rollout(prop_fn):
    """
    Adapter: makes MJX propagate_batch look like rollout_final.
    """

    def rollout_final_frb(state0, actions, obstacles, sst_params, sim_params):
        """
        state0:   (batch, 27)
        actions:  (batch, 7)
        returns:
            final_states: (batch, 27)
            valid_mask:  (batch,)  -> all True
            cost:        (batch,)  -> zeros
        """
        batch = state0.shape[0]
        final_states = prop_fn(
            sst_params.time_to_evolve,
            state0,
            actions,
        )

        valid_mask = helper.check_valid_eeb(final_states, sim_params)
        cost = jnp.zeros((batch,), dtype=jnp.float32)

        return final_states, valid_mask, cost

    return rollout_final_frb
