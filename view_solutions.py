import time
import numpy as np
import mujoco
import mujoco.viewer

# -----------------------------
# Load model
# -----------------------------
model = mujoco.MjModel.from_xml_path("models/eeonly.xml")
data = mujoco.MjData(model)

# -----------------------------
# Load saved solution
# -----------------------------
actions = np.load("solution_actions.npy")
states = np.load("solution_states.npy")

T = actions.shape[0]

# -----------------------------
# Helper: set state
# State format:
# [ ball_x, ball_y,
#   ball_dx, ball_dy,
#   block_x, block_y, block_theta,
#   block_dx, block_dy, block_dtheta ]
# -----------------------------
def set_state_from_array(data, state):
    # qpos layout:
    # [ball_x, ball_y, block_x, block_y, block_theta]
    data.qpos[:] = [
        state[0],  # ball_x
        state[1],  # ball_y
        state[4],  # block_x
        state[5],  # block_y
        state[6],  # block_theta
    ]

    # qvel layout:
    # [ball_dx, ball_dy, block_dx, block_dy, block_dtheta]
    data.qvel[:] = [
        state[2],  # ball_dx
        state[3],  # ball_dy
        state[7],  # block_dx
        state[8],  # block_dy
        state[9],  # block_dtheta
    ]

    mujoco.mj_forward(model, data)


# -----------------------------
# Initialize to first state
# -----------------------------
SUBSTEPS = 50
set_state_from_array(data, states[0])

# -----------------------------
# Launch viewer
# -----------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:

    for t in range(T-1):
        for _ in range(SUBSTEPS):
            data.ctrl[:] = actions[t+1]
            mujoco.mj_step(model, data)

            viewer.sync()
            time.sleep(model.opt.timestep)

    print("Playback finished.")
