import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
from functools import partial
import time

def make_franka_propagate_fn(mjx_model):
    """
    Batched MJX propagation for Franka (7 DOF) + free block
    """

    @partial(jax.jit, static_argnums=(0,))
    def propagate_batch(num_steps, states, actions):
        """
        states: (batch, 27)
          [ q_arm(7),
            dq_arm(7),
            block_xyz(3),
            block_quat(4),
            block_linvel(3),
            block_angvel(3) ]
        actions: (batch, 7) joint torques
        """

        batch = states.shape[0]

        # -------------------------
        # Split state
        # -------------------------
        q_arm = states[:, 0:7]
        dq_arm = states[:, 7:14]

        block_xyz = states[:, 14:17]
        block_quat = states[:, 17:21]

        block_linvel = states[:, 21:24]
        block_angvel = states[:, 24:27]

        # -------------------------
        # Assemble qpos / qvel
        # -------------------------
        qpos = jnp.concatenate(
            [q_arm, block_xyz, block_quat], axis=-1
        )  # (batch, 14)

        qvel = jnp.concatenate(
            [dq_arm, block_linvel, block_angvel], axis=-1
        )  # (batch, 13)

        # -------------------------
        # MJX data creation
        # -------------------------
        template = mjx.make_data(mjx_model)

        def _make_data(qp, qv, u):
            return template.replace(
                qpos=qp,
                qvel=qv,
                ctrl=u,
            )

        data_batch = jax.vmap(_make_data)(qpos, qvel, actions)

        # -------------------------
        # MJX stepping
        # -------------------------
        vmapped_step = jax.vmap(mjx.step, in_axes=(None, 0))

        def step_fn(i, data):
            return vmapped_step(mjx_model, data)

        final_data = jax.lax.fori_loop(
            0, num_steps, step_fn, data_batch
        )

        # -------------------------
        # Extract final state
        # -------------------------
        final_states = jnp.concatenate(
            [
                final_data.qpos[:, 0:7],        # arm q
                final_data.qvel[:, 0:7],        # arm dq
                final_data.qpos[:, 7:10],       # block xyz
                final_data.qpos[:, 10:14],      # block quat
                final_data.qvel[:, 7:10],       # block lin vel
                final_data.qvel[:, 10:13],      # block ang vel
            ],
            axis=-1,
        )

        return final_states

    return propagate_batch

from functools import partial
import jax
import jax.numpy as jnp
import mujoco.mjx as mjx


def make_ball_block_propagate_fn(mjx_model):
    """
    Batched MJX propagation for:
      - 2D force-controlled ball
      - 2D block with rotation (nonprehensile)
    """

    @partial(jax.jit, static_argnums=(0,))
    def propagate_batch(num_steps, states, actions):
        """
        states: (batch, 10)
          [ ball_x, ball_y,
            ball_dx, ball_dy,
            block_x, block_y, block_theta,
            block_dx, block_dy, block_dtheta ]

        actions: (batch, 2)
          [ Fx, Fy ]
        """

        # -------------------------
        # Split state
        # -------------------------
        ball_xy     = states[:, 0:2]
        ball_dxy    = states[:, 2:4]

        block_xyth  = states[:, 4:7]
        block_dxyth = states[:, 7:10]

        # -------------------------
        # Assemble qpos / qvel
        # MJX model layout:
        # qpos = [ ball_x, ball_y, block_x, block_y, block_theta ]
        # qvel = [ ball_dx, ball_dy, block_dx, block_dy, block_dtheta ]
        # -------------------------
        qpos = jnp.concatenate(
            [ball_xy, block_xyth], axis=-1
        )  # (batch, 5)

        qvel = jnp.concatenate(
            [ball_dxy, block_dxyth], axis=-1
        )  # (batch, 5)

        # -------------------------
        # MJX data creation
        # -------------------------
        template = mjx.make_data(mjx_model)

        def _make_data(qp, qv, u):
            return template.replace(
                qpos=qp,
                qvel=qv,
                ctrl=u,
            )

        data_batch = jax.vmap(_make_data)(qpos, qvel, actions)

        # -------------------------
        # MJX stepping (batched)
        # -------------------------
        vmapped_step = jax.vmap(mjx.step, in_axes=(None, 0))

        def step_fn(_, data):
            return vmapped_step(mjx_model, data)

        final_data = jax.lax.fori_loop(
            0, num_steps, step_fn, data_batch
        )

        # -------------------------
        # Extract final state
        # -------------------------
        final_states = jnp.concatenate(
            [
                final_data.qpos[:, 0:2],   # ball x,y
                final_data.qvel[:, 0:2],   # ball dx,dy
                final_data.qpos[:, 2:5],   # block x,y,theta
                final_data.qvel[:, 2:5],   # block dx,dy,dtheta
            ],
            axis=-1,
        )

        return final_states

    return propagate_batch



if __name__ == "__main__":
    # XML_PATH = "models/franka_block.xml"  # <-- your MJCF
    XML_PATH = "models/eeonly.xml"  # <-- your MJCF

    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
    except Exception as e:
        print(f"Error loading model from {XML_PATH}")
        raise e

    mjx_model = mjx.put_model(model)

    print("Model loaded")
    print("nq, nv, nu:", mjx_model.nq, mjx_model.nv, mjx_model.nu)


    # -------------------------------------------------
    # 2. Test parameters
    # -------------------------------------------------
    BATCH_SIZE = 10_000
    NUM_MJX_STEPS = 1  # try 1, 5, 10 later

    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 6)


    # -------------------------------------------------
    # 3. Generate initial states
    # -------------------------------------------------
    # Arm
    q_arm = jax.random.uniform(
        keys[0], (BATCH_SIZE, 7), minval=-0.1, maxval=0.1
    )
    q_arm = jnp.zeros((BATCH_SIZE, 7))
    dq_arm = jnp.zeros((BATCH_SIZE, 7))

    # Block
    block_xyz = jax.random.uniform(
        keys[1], (BATCH_SIZE, 3),
        minval=jnp.array([-0.2, -0.2, 0.05]),
        maxval=jnp.array([ 0.2,  0.2, 0.05]),
    )
    block_quat = jnp.tile(
    jnp.array([1.0, 0.0, 0.0, 0.0]), (BATCH_SIZE, 1))

    block_vel = jnp.zeros((BATCH_SIZE, 3))
    block_omega = jnp.zeros((BATCH_SIZE, 3))

    initial_states = jnp.concatenate(
        [
            q_arm,
            dq_arm,
            block_xyz,
            block_quat,
            block_vel,
            block_omega,
        ],
        axis=-1,
    )

    assert initial_states.shape == (BATCH_SIZE, 27)


    # -------------------------------------------------
    # 4. Actions (joint torques)
    # -------------------------------------------------
    actions = jax.random.uniform(
        keys[2], (BATCH_SIZE, 7), minval=-5.0, maxval=5.0
    )


    # -------------------------------------------------
    # 5. Create propagation function
    # -------------------------------------------------
    print("\n1. Compiling MJX Franka propagation...")
    print(f"JAX devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")

    #franka_propagate_fn = make_franka_propagate_fn(mjx_model)
    franka_propagate_fn = make_ball_block_propagate_fn(mjx_model)


    # -------------------------------------------------
    # 6. JIT compilation timing
    # -------------------------------------------------
    start_jit = time.perf_counter()

    _ = franka_propagate_fn(
        NUM_MJX_STEPS,
        initial_states[:1],
        actions[:1],
    ).block_until_ready()

    jit_time = time.perf_counter() - start_jit
    print(f"JIT compilation time: {jit_time:.3f} s")


    # -------------------------------------------------
    # 7. Warmup run
    # -------------------------------------------------
    print("\n2. Warmup run...")
    _ = franka_propagate_fn(
        NUM_MJX_STEPS,
        initial_states,
        actions,
    ).block_until_ready()


    # -------------------------------------------------
    # 8. Timed execution
    # -------------------------------------------------
    print("Timed run...")
    start_exec = time.perf_counter()

    final_states = franka_propagate_fn(
        NUM_MJX_STEPS,
        initial_states,
        actions,
    ).block_until_ready()

    exec_time = time.perf_counter() - start_exec


    # -------------------------------------------------
    # 9. Output results
    # -------------------------------------------------
    print("\n===== RESULTS =====")
    print(f"Execution Time: {exec_time:.4f} s")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Num MJX Steps: {NUM_MJX_STEPS}")
    print(f"Final States Shape: {final_states.shape}")
    print("Example final state (first):")
    print(final_states[0])