import time
import numpy as np
import mujoco
import mujoco.viewer

# 1. Load Model and Data
# model = mujoco.MjModel.from_xml_path("models/block_push_nice.xml")
model = mujoco.MjModel.from_xml_path("models/acrobot_nice.xml")
#model = mujoco.MjModel.from_xml_path("models/cartpole_nice.xml")
data = mujoco.MjData(model)
        
# 2. Load Pre-computed MJX Trajectory
# Expected shape: (Total Steps, nq + nv)
#trajectory = np.load("visuals/mjx_full_trajectory.npy")
trajectory = np.load("videos/acrobot_traj.npy")
#trajectory = np.load("videos/cartpole_traj.npy")
nq = model.nq
state0 = trajectory[0]  # Initial state for reference
data.qpos[:] = state0[:nq]

print(f"Loaded trajectory with {len(trajectory)} frames.")

# 3. Playback Loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 6.0 
    time.sleep(0.7)
    # Optional: Reset camera or settings here

    
    for state in trajectory:
        if not viewer.is_running():
            break
            
        # Split state back into qpos and qvel
        data.qpos[:] = state[:nq]
        data.qvel[:] = state[nq:]
        
        # Compute forward kinematics for visualization
        mujoco.mj_forward(model, data)
        
        viewer.sync()
        
        # Match the simulation timestep for real-time speed
        time.sleep(model.opt.timestep)

    time.sleep(1.0)  # Pause at the end of the trajectory

print("Playback finished.")