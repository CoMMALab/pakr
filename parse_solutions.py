import numpy as np
import jax
from jax import numpy as jnp
import mujoco
from mujoco import mjx

# 1. Setup
model_path = "models/eeonly.xml"
mj_model = mujoco.MjModel.from_xml_path(model_path)
mjx_model = mjx.put_model(mj_model)

# 2. Load Planner Data
actions = jnp.array(np.load("solution_actions.npy"))
initial_state_arr = np.load("solution_states.npy")[0]

T = actions.shape[0]
SUBSTEPS = 5

@jax.jit
def high_freq_rollout(model, initial_state, action_seq):
    def set_initial_mjx(state_arr):
        # ball_x, ball_y, ball_dx, ball_dy, block_x, block_y, block_theta...
        qpos = jnp.array([state_arr[0], state_arr[1], state_arr[4], state_arr[5], state_arr[6]])
        qvel = jnp.array([state_arr[2], state_arr[3], state_arr[7], state_arr[8], state_arr[9]])
        return qpos, qvel

    qpos, qvel = set_initial_mjx(initial_state)
    data = mjx.make_data(model)
    data = data.replace(qpos=qpos, qvel=qvel)
    data = mjx.forward(model, data)

    def control_step(carry_data, action):
        def physics_step(inner_data, _):
            d = inner_data.replace(ctrl=action)
            d = mjx.step(model, d)
            # Concatenate qpos and qvel into one state vector for storage
            state_vector = jnp.concatenate([d.qpos, d.qvel])
            return d, state_vector

        carry_data, substep_traj = jax.lax.scan(physics_step, carry_data, None, length=SUBSTEPS)
        return carry_data, substep_traj

    _, trajectory = jax.lax.scan(control_step, data, action_seq)
    return trajectory

print("Generating high-frequency MJX trajectory...")
# Trajectory shape: (T, SUBSTEPS, nq + nv)
full_trajectory = high_freq_rollout(mjx_model, initial_state_arr, actions)

# Flatten to (T * SUBSTEPS, nq + nv)
flattened_trajectory = full_trajectory.reshape(-1, mj_model.nq + mj_model.nv)

# Save to disk
np.save("mjx_full_trajectory.npy", np.array(flattened_trajectory))
print(f"Saved {len(flattened_trajectory)} states to mjx_full_trajectory.npy")