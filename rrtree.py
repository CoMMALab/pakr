from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
import time

@struct.dataclass
class KinoTree:
    states: jnp.ndarray       # shape (MAX_TREE_SIZE, state_dim)
    actions: jnp.ndarray      # shape (MAX_TREE_SIZE, action_dim)
    parents: jnp.ndarray      # shape (MAX_TREE_SIZE,)
    costs: jnp.ndarray        # shape (MAX_TREE_SIZE,)
    tree_size: int            # current number of nodes

    @staticmethod
    def init(max_size: int, state_dim: int, action_dim: int):
        return KinoTree(
            states=jnp.zeros((max_size, state_dim), dtype=jnp.float32),
            actions=jnp.zeros((max_size, action_dim), dtype=jnp.float32),
            parents=jnp.full((max_size,), -1, dtype=jnp.int32),
            costs=jnp.full((max_size,), jnp.inf, dtype=jnp.float32),
            tree_size=0,
        )


@jax.jit
def add_nodes(tree, new_states, new_actions, parent_idxs, new_costs, N):
    """
    Add new nodes to the tree. Returns a new KinoTree (functional update).
    
    Args:
        tree        : KinoTree
        new_states  : (N, state_dim)
        new_actions : (N, action_dim)
        parent_idxs : (N,)
        new_costs   : (N,)
    """
    new_states = jnp.atleast_2d(new_states)
    new_actions = jnp.atleast_2d(new_actions)
    new_costs = jnp.atleast_1d(new_costs)
    parent_idxs = jnp.atleast_1d(parent_idxs)
    
    start = tree.tree_size

    states = lax.dynamic_update_slice(tree.states, new_states, (start, 0))
    actions = lax.dynamic_update_slice(tree.actions, new_actions, (start, 0))
    parents = lax.dynamic_update_slice(tree.parents, parent_idxs, (start,))
    costs = lax.dynamic_update_slice(tree.costs, new_costs, (start,))
    
    tree_size = start + N

    return tree.replace(
        states=states,
        actions=actions,
        parents=parents,
        costs=costs,
        tree_size=tree_size
    ), start
