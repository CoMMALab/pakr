import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
import os
from flax import struct
from vine.pbd_vine import step_vine_batched
from params import Callables, Position
from vine.nns_usage import solve as find_actuator_params, solve_fwd as actuator_params_fwd_, params as act_params
from vine.load_env import load_box_config
from vine.nns import get_or_train_model, get_prediction_function

# ------------------------------------------------------------------
# VINE DYNAMICS SETUP
# ------------------------------------------------------------------

trained_state, scaling_info, model = get_or_train_model(act_params)
predict = get_prediction_function(trained_state, scaling_info, model)
find_actuator_params = jax.vmap(find_actuator_params, in_axes=(None, None, 0))
find_actuator_params = jax.jit(find_actuator_params, static_argnames=('predict', 'params'))
forward = jax.jit(step_vine_batched, static_argnames=['params', 'x0_list', 'y0_list', 'heading0_list', 'bend_energy_func'])
actuator_params_fwd = jax.vmap(lambda a, b, c: actuator_params_fwd_(predict, act_params, a, b, c), in_axes=(0, 0, 0))

# ------------------------------------------------------------------
# CORE FUNCTIONS
# ------------------------------------------------------------------

@partial(jax.jit, static_argnums=(0, 1, 2))
def rollout_jit_full(sst_params, simparams, batch_size, 
                    curr_time, cspace, bodies, bending_control, obstacles):
    """
    Returns full history for the given sst_params.time_to_evolve horizon.
    """
    steps_to_iter = sst_params.time_to_evolve
    init_x, init_y, init_heading = sst_params.start.x, sst_params.start.y, sst_params.start.z

    def scan_fn(carry, step_idx):
        cspace, bodies, curr_time = carry
        reached_max = bodies >= simparams.max_bodies - 1
        
        next_cspace, next_bodies = forward(
            simparams, cspace, bodies, bending_control,
            init_x, init_y, init_heading, actuator_params_fwd, obstacles
        )
        
        cspace = jnp.where(reached_max[..., None], cspace, next_cspace)
        bodies = jnp.where(reached_max, bodies, next_bodies)
        curr_time = curr_time + simparams.dt
        
        new_carry = (cspace, bodies, curr_time)
        return new_carry, new_carry

    init_state = (cspace, bodies, curr_time)
    final_carry, history = jax.lax.scan(scan_fn, init_state, jnp.arange(steps_to_iter))
    
    # history: (c_hist, b_hist, t_hist)
    return history, final_carry

def cspace_to_tip_single(sim_params, cspace, n_bodies, x0, y0, h0):
    angles = cspace[:sim_params.max_bodies]
    tip_len = cspace[sim_params.max_bodies]
    
    def body_fn(carry, i):
        x, y, h = carry
        angle = angles[i]
        length = jnp.where(i < n_bodies, sim_params.body_length, 
                           jnp.where(i == n_bodies, tip_len, 0.0))
        new_h = h + angle
        new_x = x + length * jnp.cos(new_h)
        new_y = y + length * jnp.sin(new_h)
        return (new_x, new_y, new_h), None

    (xf, yf, hf), _ = jax.lax.scan(body_fn, (x0, y0, h0), jnp.arange(sim_params.max_bodies))
    return jnp.array([xf, yf, hf])

@partial(jax.jit, static_argnums=(0,))
def cspace_to_tip_batched(sim_params, cspace, n_bodies, x0, y0, h0):
    return jax.vmap(cspace_to_tip_single, in_axes=(None, 0, 0, None, None, None))(
        sim_params, cspace, n_bodies, x0, y0, h0
    )

@jax.jit
def reached_goal_vine(states, goal, radius):
    # Expects states to be the (B, 3) tip array
    return jnp.linalg.norm(states[:, 0:2] - jnp.array([goal.x, goal.y]), axis=1) < radius

# ------------------------------------------------------------------
# TRAJECTORY EXTRACTION LOGIC
# ------------------------------------------------------------------

def save_full_trajectory(solution_path, sst_params, sim_params, obstacles):
    actions = jnp.load(solution_path)
    actions_batched = actions[None, ...] 
    
    # Initial state
    cspace = jnp.zeros((1, sim_params.max_bodies + 1))
    bodies = jnp.zeros((1,), dtype=jnp.int32)
    curr_time = jnp.array([0.0])
    
    all_c, all_b, all_tips = [], [], []
    
    max_loops = 20 # Safety cap to prevent infinite loops
    goal_reached = False
    
    print(f"Propagating solution from {solution_path}...")
    
    for i in range(max_loops):
        print(i)
        # 1. Run the 70-step JITed rollout
        history, (cspace, bodies, curr_time) = rollout_jit_full(
            sst_params, sim_params, 1, curr_time, cspace, bodies, actions_batched, obstacles
        )
        c_hist, b_hist, _ = history
        
        # 2. Compute tips for this segment
        compute_tips = jax.vmap(lambda c, b: cspace_to_tip_batched(
            sim_params, c, b, sst_params.start.x, sst_params.start.y, sst_params.start.z
        ))
        tip_hist = compute_tips(c_hist, b_hist)
        
        # 3. Store results
        # 3. Store results
        all_c.append(c_hist[:, 0, :])
        # FIX: b_hist is (70, 1), so use [:, 0] to get (70,)
        # We then use [:, None] to keep it as (70, 1) for later concatenation
        all_b.append(b_hist[:, 0, None]) 
        all_tips.append(tip_hist[:, 0, :])
        
        # 4. Check if we reached the goal in this segment
        # Check the last tip in the current history segment
        if reached_goal_vine(tip_hist[-1, :, :], sst_params.goal, sst_params.goal_radius)[0]:
            print(f"Goal reached at loop iteration {i+1}!")
            goal_reached = True
            break
            
    # Combine all segments
    full_c = jnp.concatenate(all_c, axis=0)
    full_b = jnp.concatenate(all_b, axis=0)
    full_tips = jnp.concatenate(all_tips, axis=0)

    # Final Shape: (Total_Steps, Tips(3) + CSpace(31) + Bodies(1))
    traj_final = jnp.concatenate([
        full_tips, 
        full_c, 
        full_b.astype(jnp.float32)
    ], axis=-1)

    os.makedirs("videos", exist_ok=True)
    save_path = "vine/vine_traj.npy"
    np.save(save_path, np.array(traj_final))
    print(f"Trajectory saved to {save_path}. Total steps: {traj_final.shape[0]}")


@struct.dataclass
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

@struct.dataclass
class SSTparams:
    batch_size: int
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    start: Position
    goal: Position
    goal_radius: float
    time_to_evolve: int = 70

if __name__ == "__main__":
    cfg = load_box_config('vine/envs/env_live.txt')

    batch_size = 128
    A = 2
    max_bodies = 30
    sim_params = VineParams(
        batch_size=batch_size,
        max_bodies=max_bodies,
        dims=max_bodies + 5, # tip + cspace + tip + n_bodies
        action_dims=max_bodies, # bending control for each body
        body_length=68.0, # 25.0 mm
        radius=50, # 16.0,
        dt=1.0,
        grow_rate=20.0,
        grow_force=15.0,
        stiffness=50.0,
        damping=50.0,
        # Curiously, decreasing substeps helps prevent penetration bugs. But it doesn't fix the root problem
        substeps=15, # FIXME THIS NUMBER CAN BE MUCH SMALLER IF WE DO LANGRANGE PROPERRLY
        alpha=1e-2,
    )
    
    obstacles=cfg['obstacles']

    print(obstacles.shape)
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

    save_full_trajectory('vine/results/solution_iter_8.npy', sst_params, sim_params, obstacles)

    