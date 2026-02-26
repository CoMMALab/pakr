import jax
import jax.numpy as jnp
from functools import partial
import mujoco
import mujoco.mjx as mjx
import time

# ---------------------------------------------
# 1. Function Factory Definition
# ---------------------------------------------
def make_propagate_fn(mjx_model):
    """
    Creates a JIT-compiled propagation function that CLOSES over the mjx_model.
    This avoids passing mjx_model as a hashable JIT argument.
    """
    
    @partial(jax.jit, static_argnums=(0,)) 
    def mjx_propagate_batch_jit(num_steps, initial_states, actions):
        """
        The actual dynamics function. It uses mjx_model captured from the outer scope.
        """
        # State mapping: [qpos_x, qpos_theta, qvel_x, qvel_theta]
        qpos = initial_states[:, :2]
        qvel = initial_states[:, 2:]
        
        # Create a single template
        template = mjx.make_data(mjx_model)
        
        # Use vmap to properly batch the data structure
        def _make_data(q, v, u):
            return template.replace(qpos=q, qvel=v, ctrl=u)
        
        # vmap over the batch dimension to create batched data
        data_batch = jax.vmap(_make_data)(qpos, qvel, actions)
        
        # Create vmapped step function - this is the KEY to performance!
        vmapped_step = jax.vmap(mjx.step, in_axes=(None, 0))
        
        # Run simulation using fori_loop with vmapped step
        def scan_step(i, data):
            return vmapped_step(mjx_model, data)
        
        final_data = jax.lax.fori_loop(0, num_steps, scan_step, data_batch)
        
        # Extract the final state batch
        final_states = jnp.concatenate([final_data.qpos, final_data.qvel], axis=-1)
        
        return final_states
    
    return mjx_propagate_batch_jit

# --- Setup and Test Execution ---

# 1. Load MuJoCo Model and compile to MJX
XML_PATH = 'models/cartpole2d.xml'

try:
    model = mujoco.MjModel.from_xml_path(XML_PATH)
except Exception as e:
    print(f"Error loading model from {XML_PATH}. Ensure the file exists.")
    print(e)
    exit()

# Compile the model for MJX
mjx_model = mjx.put_model(model)

# 2. Define Test Parameters
BATCH_SIZE = 10000
NUM_MJX_STEPS = 1  # Number of simulation steps

# 3. Generate Initial States and Actions
key = jax.random.PRNGKey(42)
key_q, key_v, key_a = jax.random.split(key, 3)

initial_qpos = jax.random.uniform(key_q, (BATCH_SIZE, 2), minval=-0.01, maxval=0.01)
initial_qvel = jax.random.uniform(key_v, (BATCH_SIZE, 2), minval=-0.01, maxval=0.01)
initial_states = jnp.concatenate([initial_qpos, initial_qvel], axis=-1)

actions = jax.random.uniform(key_a, (BATCH_SIZE, 1), minval=-5.0, maxval=5.0)

# 4. Create the JIT-compiled propagation function
print("1. Compiling JAX/MJX function...")
print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

mjx_propagate_fn = make_propagate_fn(mjx_model)

# JIT Compilation Run
start_jit = time.perf_counter()
print("nq, nv, nu:", mjx_model.nq, mjx_model.nv, mjx_model.nu)

_ = mjx_propagate_fn(NUM_MJX_STEPS, initial_states[:1], actions[:1]).block_until_ready()

jit_time = time.perf_counter() - start_jit
print(f"JIT Compilation Time: {jit_time:.3f} s")

# 5. Main Execution Run
print("\n2. Executing batched propagation...")

# Warmup run
print("Warmup run...")
_ = mjx_propagate_fn(NUM_MJX_STEPS, initial_states, actions).block_until_ready()

print("Timed run...")
start_exec = time.perf_counter()
final_states = mjx_propagate_fn(NUM_MJX_STEPS, initial_states, actions).block_until_ready()
exec_time = time.perf_counter() - start_exec

# 6. Output Results
print(f"Execution Time: {exec_time:.4f} s")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Num Steps: {NUM_MJX_STEPS}")
print(f"Final States Shape: {final_states.shape}")
print(f"\nExample Final State (First): {final_states[0]}")