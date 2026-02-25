import time
import numpy as np
import mujoco
import mujoco.viewer

# 1. Load Model and Data
model = mujoco.MjModel.from_xml_path("models/eeonly.xml")
data = mujoco.MjData(model)

# 2. Load Pre-computed MJX Trajectory
# Expected shape: (Total Steps, nq + nv)
trajectory = np.load("mjx_full_trajectory.npy")
nq = model.nq

print(f"Loaded trajectory with {len(trajectory)} frames.")

# 3. Playback Loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Optional: Reset camera or settings here
    viewer.cam.distance = 3.0 
    
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

print("Playback finished.")