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
    def init(max_size: int, max_bodies: int):
        return KinoTree(
            states=jnp.zeros((max_size, max_bodies + 5), dtype=jnp.float32),
            actions=jnp.zeros((max_size, max_bodies, 2), dtype=jnp.float32),
            parents=jnp.full((max_size,), -1, dtype=jnp.int32),
            costs=jnp.full((max_size,), jnp.inf, dtype=jnp.float32),
            tree_size=0,
        )


# In /workspace/vine/rrtree.py

def add_nodes(tree, states, actions, parents, costs, n):
    start = tree.tree_size
    
    # Ensure states and actions have a leading batch dimension
    # If adding a single node, states is (D,) -> (1, D)
    # actions is (max_bodies, 2) -> (1, max_bodies, 2)
    new_states = jnp.atleast_2d(states)
    
    # Logic to ensure actions is Rank 3 (Batch, Bodies, 2)
    if actions.ndim == 2:
        new_actions = actions[None, ...] # Add leading batch dim
    else:
        new_actions = actions

    # Update states (Rank 2: (MAX, D))
    updated_states = jax.lax.dynamic_update_slice(
        tree.states, new_states, (start, 0)
    )
    
    # Update actions (Rank 3: (MAX, Bodies, 2))
    # Provide 3 indices to match the (500000, 30, 2) shape
    updated_actions = jax.lax.dynamic_update_slice(
        tree.actions, new_actions, (start, 0, 0)
    )

    # Update parents and costs (Rank 1)
    new_parents = jnp.atleast_1d(parents)
    new_costs = jnp.atleast_1d(costs)
    
    updated_parents = jax.lax.dynamic_update_slice(tree.parents, new_parents, (start,))
    updated_costs = jax.lax.dynamic_update_slice(tree.costs, new_costs, (start,))

    return tree.replace(
        states=updated_states,
        actions=updated_actions,
        parents=updated_parents,
        costs=updated_costs,
        tree_size=start + n
    ), start