import jax
import jax.numpy as jnp
import mjx
import mujoco
import mujoco.viewer
import time

# batched sim per sample in sst. B1 = batch size for model param variation, B2 = sample batch size
def simulate_until_reached(
    models: mjx.Model,               # shape (B1,)
    initial_states: mjx.State,       # shape (B1, B2)
    ee_idx: tuple[int, int],         # indices in qpos for (x, y)
    ee_target: jnp.ndarray,          # shape (B1, B2, 2)
    speed: float = 0.2,
    tol: float = 0.005,
    max_steps: int = 500,
) -> mjx.State:
    B1, B2 = models.batch_size, initial_states.batch_size
    x_idx, y_idx = ee_idx
    states = initial_states

    def cond(loop_carry):
        step, states = loop_carry
        qpos = mjx.get_qpos(states)
        ee_pos = jnp.stack([qpos[..., x_idx], qpos[..., y_idx]], axis=-1)
        dist = jnp.linalg.norm(ee_target - ee_pos, axis=-1)
        return jnp.logical_and(step < max_steps, jnp.any(dist > tol))

    def body(loop_carry):
        step, states = loop_carry
        qpos = mjx.get_qpos(states)
        ee_pos = jnp.stack([qpos[..., x_idx], qpos[..., y_idx]], axis=-1)
        delta = ee_target - ee_pos
        norm = jnp.linalg.norm(delta, axis=-1, keepdims=True)
        velocity = jnp.where(norm > 1e-6, delta / norm * speed, jnp.zeros_like(delta))

        ctrl = jnp.zeros((B1, B2, models.nu))
        ctrl = ctrl.at[..., 0].set(velocity[..., 0])  # EE x actuator
        ctrl = ctrl.at[..., 1].set(velocity[..., 1])  # EE y actuator

        states = mjx.set_control(models, states, ctrl)
        states = mjx.step(models, states)
        return step + 1, states

    _, final_states = jax.lax.while_loop(cond, body, (0, states))
    return final_states

# Render final path with vcxsrv
def render_sol(path, model):
    def move_to(target_xy, speed=0.5, tol=0.005):
        while True:
            pos_x = data.qpos[0]
            pos_y = data.qpos[1]

            err_x = target_xy[0] - pos_x
            err_y = target_xy[1] - pos_y
            dist = jnp.hypot(err_x, err_y)

            if dist < tol:
                break

            direction = jnp.array([err_x, err_y]) / dist
            velocity = direction * speed

            data.ctrl[0] = velocity[0]
            data.ctrl[1] = velocity[1]

            mujoco.mj_step(model, data)
            viewer.sync()

            time.sleep(0.01)

    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while True:
            for waypoint in path:
                move_to(waypoint)