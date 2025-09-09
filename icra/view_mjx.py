import os
import time
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import mujoco
import mujoco.viewer
import jax.numpy as jnp
import helper

def to_xml(filename):
    """
    Generates a MuJoCo XML model:
    - Loads obstacles from CSV
    - Adds a controllable ball with actuators
    - Disables gravity so the ball follows waypoints smoothly
    """
    def indent(elem, level=0):
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            for child in elem:
                indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

    # Load obstacles CSV
    df = pd.read_csv(filename)  # columns: x1,y1,z1,x2,y2,z2

    # Root element
    mujoco_root = ET.Element("mujoco", model="waypoints_demo")
    ET.SubElement(mujoco_root, "compiler", angle="degree", coordinate="local")
    # Set gravity to zero so ball doesn't fall
    ET.SubElement(mujoco_root, "option", timestep="0.002", gravity="0 0 0")
    worldbody = ET.SubElement(mujoco_root, "worldbody")

    # Add obstacles
    for idx, row in df.iterrows():
        x1, y1, z1, x2, y2, z2 = row
        center = [(x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2]
        size = [abs(x2 - x1) / 2, abs(y2 - y1) / 2, abs(z2 - z1) / 2]

        body = ET.SubElement(worldbody, "body", name=f"box{idx}", pos=f"{center[0]} {center[1]} {center[2]}")
        ET.SubElement(body, "geom",
                      type="box",
                      size=f"{size[0]} {size[1]} {size[2]}",
                      rgba="0.5 0.5 0.8 0.3",
                      mass="0")

    # Add ball body at origin
    ball_body = ET.SubElement(worldbody, "body", name="ball", pos="0.1 0.08 0.05")
    ET.SubElement(ball_body, "geom",
                type="sphere",
                size="0.02",        # Keep radius for visualization
                rgba="1 0 0 1",
                mass="1",
                contype="0",
                conaffinity="0")
    
    goal_body = ET.SubElement(worldbody, "body", name="goal", pos="0.8 0.95 0.9")
    ET.SubElement(goal_body, "geom",
                  type="sphere",
                  size="0.05",       # Smaller ball size
                  rgba="0 1 0 1",
                  mass="0")

    # Add free joint for ball so it can move freely
    ET.SubElement(ball_body, "joint", name="ball_free", type="free")

    # Add actuators to control ball position directly (x, y, z)
    actuator = ET.SubElement(mujoco_root, "actuator")
    ET.SubElement(actuator, "motor", joint="ball_free", ctrllimited="true",
                  ctrlrange="-10 10", gear="1 0 0 0 0 0")  # X-axis
    ET.SubElement(actuator, "motor", joint="ball_free", ctrllimited="true",
                  ctrlrange="-10 10", gear="0 1 0 0 0 0")  # Y-axis
    ET.SubElement(actuator, "motor", joint="ball_free", ctrllimited="true",
                  ctrlrange="-10 10", gear="0 0 1 0 0 0")  # Z-axis

    # Prettify XML
    indent(mujoco_root)

    # Return XML string
    xml_str = ET.tostring(mujoco_root, encoding="utf-8")
    return xml_str


def run_mjx_with_waypoints(csv_filename, waypoints_npy):
    # Load waypoints
    waypoints = jnp.load(waypoints_npy)  # shape (T, 3)
    obstacles = helper.get_obs(csv_filename)
    in_collision = helper.collision_check(waypoints, obstacles)
    T = waypoints.shape[0]
    print(waypoints)

    # Generate model XML
    xml_str = to_xml(csv_filename)
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)

    # Use MuJoCo passive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        t = 0
        while viewer.is_running():
            # Current target waypoint
            target = waypoints[t]

            # Get current ball position
            ball_pos = data.qpos[0:3]
            error = target - ball_pos

            # PD control for smooth movement
            kp = 50.0
            kv = 10.0
            data.ctrl[0:3] = kp * error - kv * data.qvel[0:3]

            # Step simulation forward
            mujoco.mj_step(model, data)

            # Move to next waypoint if close enough
            if np.linalg.norm(error) < 0.02 and t < T - 1:
                t += 1

            viewer.sync()
            time.sleep(0.002)  # match timestep


if __name__ == '__main__':
    run_mjx_with_waypoints(
        csv_filename='envs/tree.csv',
        waypoints_npy='cache/waypoints.npy'
    )
