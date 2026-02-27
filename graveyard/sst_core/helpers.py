import jax.numpy as jnp
import jax
import mjx


# abstract states since mjx uses mjx.state
class StateWrapper:
    def to_vector(self) -> jnp.ndarray:
        raise NotImplementedError
    
class MJXStateWrapper(StateWrapper):
    def __init__(self, state: mjx.State):
        self.state = state

    def to_vector(self) -> jnp.ndarray:
        qpos = self.state.qpos
        return jnp.concatenate([qpos[0:2], qpos[7:].reshape(-1)])

    def get_mjx_state(self):
        return self.state
    
class VineStateWrapper(StateWrapper):
    def __init__(self, pose: jnp.ndarray):
        self.pose = pose

    def to_vector(self) -> jnp.ndarray:
        return self.pose
    
def wrap_rotation(diff):
    # Wrap diff into (-pi, pi]
    return (diff + jnp.pi) % (2 * jnp.pi) - jnp.pi

def distance(sst_params, diff):
    rot_delta = wrap_rotation(diff[..., 2])
    # Scale rotation based on angle weight
    rot_delta *= sst_params.rotation_metric_scale
    # Compute pairwise distance squared
    dist2 = diff[..., 0] ** 2 + diff[..., 1] ** 2 + rot_delta ** 2
    return dist2    
    
# finds distance squared between query points and all reference points
def nearest_neighbor_all(sst_params, ref_points: jnp.ndarray, query_points: jnp.ndarray):
    diff = query_points[:, None] - ref_points 
    dist2 = distance(sst_params, diff)
    return dist2