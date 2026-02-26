import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("obstacles.xml")
data = mujoco.MjData(model)

# EE joint and actuator indices
ee_x_idx = model.joint("ee_x").qposadr
ee_y_idx = model.joint("ee_y").qposadr
act_x = model.actuator("ee_x").id
act_y = model.actuator("ee_y").id
print(ee_x_idx, ee_y_idx, act_x, act_y)
# Square corners (relative to origin)
square_path = [
    [0,  0.2],
    [-0.4,  0.2],
    [-0.4, -0.2],
    [0, -0.2],
]


def move_to(target_xy, speed=0.5, tol=0.005):
    while True:
        pos_x = data.qpos[ee_x_idx]
        pos_y = data.qpos[ee_y_idx]

        err_x = target_xy[0] - pos_x
        err_y = target_xy[1] - pos_y
        dist = np.hypot(err_x, err_y)

        if dist < tol:
            break

        direction = np.array([err_x, err_y]) / dist
        velocity = direction * speed

        data.ctrl[act_x] = velocity[0]
        data.ctrl[act_y] = velocity[1]

        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(0.01)


with mujoco.viewer.launch_passive(model, data) as viewer:
    while True:
        for corner in square_path:
            print(data.qpos[ee_x_idx], data.qpos[ee_y_idx])
            move_to(corner)
