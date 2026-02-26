import jax
jax.config.update("jax_disable_jit", True)

import mujoco
import mujoco.viewer
from mujoco import mjx
import jax.numpy as jnp
import time

XML_PATH = "models/eeonly.xml"

# Load model
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

print("nq:", model.nq, "nv:", model.nv, "nu:", model.nu)
print("ctrlrange:", model.actuator_ctrlrange)

# MJX setup
mjx_model = mjx.put_model(model)
mjx_data = mjx.put_data(model, data)

mjx_data = mjx.forward(mjx_model, mjx_data)

def step_random(m, d, key):
    key, subkey = jax.random.split(key)
    ctrl = jax.random.uniform(subkey, (m.nu,), minval=-10.0, maxval=10.0)
    d = d.replace(ctrl=ctrl)
    d = mjx.step(m, d)
    return d, key

rng = jax.random.PRNGKey(0)

# 1. Initialize MJX data correctly
mjx_data = mjx.put_data(model, data)

# 2. IMPORTANT: Run a forward pass to settle the physics state in MJX
mjx_data = mjx.forward(mjx_model, mjx_data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Run physics steps
        for _ in range(5):
            # Pass the result back to mjx_data!
            mjx_data, rng = step_random(mjx_model, mjx_data, rng)

        # 3. Explicitly sync back to the CPU 'data' object for the viewer
        mjx.get_data_into(data, mjx_model, mjx_data)
        
        viewer.sync()
        time.sleep(0.01)
