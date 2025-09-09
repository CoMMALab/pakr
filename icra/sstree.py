from flax import struct
import jax.numpy as jnp
import jax

@struct.dataclass
class SSTree:
    num_states: int
    num_witnesses: int

    # static fields
    dims: int = struct.field(pytree_node=False) 
    action_dims: int = struct.field(pytree_node=False)

    # -------- States --------
    _isactive: jnp.ndarray
    _kil: jnp.ndarray 
    _c_spaces: jnp.ndarray
    _controls: jnp.ndarray
    _cost_to_come: jnp.ndarray
    _cost_total: jnp.ndarray
    _parent_idxs: jnp.ndarray
    _num_children: jnp.ndarray    

    # -------- Witnesses --------
    _witness_positions: jnp.ndarray
    _rep_idxs: jnp.ndarray


def init_tree(dims, action_dims, init_size=32768):
    return SSTree(
        num_states=0,
        num_witnesses=0,
        dims=dims,
        action_dims=action_dims,
        _isactive=jnp.zeros(init_size, dtype=bool),
        _kil=jnp.zeros(init_size, dtype=bool),
        _c_spaces=jnp.zeros((init_size, dims), dtype=jnp.float32),
        _controls=jnp.zeros((init_size, action_dims + 2), dtype=jnp.float32),
        _cost_to_come=jnp.zeros(init_size, dtype=jnp.float32),
        _cost_total=jnp.zeros(init_size, dtype=jnp.float32),
        _parent_idxs=jnp.zeros(init_size, dtype=jnp.int32),
        _num_children=jnp.zeros(init_size, dtype=jnp.int32),

        _witness_positions=jnp.zeros((init_size, dims), dtype=jnp.float32),
        _rep_idxs=jnp.full(init_size, -1, dtype=jnp.int32),
    )

from jax import lax

@jax.jit
def add_states(tree: SSTree, isactive, c_space, controls,
               cost_to_come, cost_total, parent_idx):
    c_space = jnp.atleast_2d(c_space)
    controls = jnp.atleast_2d(controls)
    cost_to_come = jnp.atleast_1d(cost_to_come)
    cost_total = jnp.atleast_1d(cost_total)
    parent_idx = jnp.atleast_1d(parent_idx)
    isactive = jnp.atleast_1d(isactive)

    n_new = c_space.shape[0]
    start = tree.num_states

    # Use lax.dynamic_update_slice for dynamic insertion
    _isactive = lax.dynamic_update_slice(tree._isactive, isactive, (start,))
    _c_spaces = lax.dynamic_update_slice(tree._c_spaces, c_space, (start, 0))
    _controls = lax.dynamic_update_slice(tree._controls, controls, (start, 0))
    _cost_to_come = lax.dynamic_update_slice(tree._cost_to_come, cost_to_come, (start,))
    _cost_total = lax.dynamic_update_slice(tree._cost_total, cost_total, (start,))
    _parent_idxs = lax.dynamic_update_slice(tree._parent_idxs, parent_idx, (start,))
    _num_children = lax.dynamic_update_slice(tree._num_children,
                                             jnp.zeros_like(parent_idx), (start,))

    tree = tree.replace(
        _isactive=_isactive,
        _c_spaces=_c_spaces,
        _controls=_controls,
        _cost_to_come=_cost_to_come,
        _cost_total=_cost_total,
        _parent_idxs=_parent_idxs,
        _num_children=_num_children,
        num_states=start + n_new,
    )

    # Increment children counts for valid parents
    updates = jnp.ones_like(parent_idx, dtype=tree._num_children.dtype)
    tree = tree.replace(
        _num_children=tree._num_children.at[parent_idx].add(updates)
    )

    return tree

@jax.jit
def add_witnesses(
    tree: SSTree,
    witness_positions: jnp.ndarray,
    rep_idxs: jnp.ndarray,
) -> tuple[SSTree, jnp.ndarray]:
    """
    Add witness positions and record their representative state indices (JIT-safe).

    Args:
        tree: SSTree
        witness_positions: (K, dims) array of witness positions (or shape (dims,) for one)
        rep_idxs: (K,) int array of representative state indices (use -1 for no rep)

    Returns:
        (new_tree, witness_idx_range)
    """
    # Ensure shapes / dtypes
    witness_positions = jnp.atleast_2d(witness_positions)
    rep_idxs = jnp.atleast_1d(rep_idxs).astype(jnp.int32)
    n_new = witness_positions.shape[0]

    # Dynamic start index
    start = tree.num_witnesses

    # Use lax.dynamic_update_slice for JIT-safe updates
    new_witness_positions = lax.dynamic_update_slice(
        tree._witness_positions,
        witness_positions,
        (start, 0)
    )

    new_rep_idxs = lax.dynamic_update_slice(
        tree._rep_idxs,
        rep_idxs,
        (start,)
    )

    # Update the tree immutably
    tree = tree.replace(
        _witness_positions=new_witness_positions,
        _rep_idxs=new_rep_idxs,
        num_witnesses=start + n_new,
    )

    return tree

@jax.jit
def batch_prune(tree: SSTree) -> SSTree:
    def cond_fn(tree):
        # Find candidates: inactive, not already pruned, no children
        prune_mask = (~tree._isactive) & (~tree._kil) & (tree._num_children == 0)
        return jnp.any(prune_mask)

    def body_fn(tree):
        prune_mask = (~tree._isactive) & (~tree._kil) & (tree._num_children == 0)

        # Mark these nodes as pruned
        tree = tree.replace(
            _kil=jnp.where(prune_mask, True, tree._kil)
        )

        # Get parents of newly pruned nodes (dummy -1 if not pruned)
        parents = jnp.where(prune_mask, tree._parent_idxs, -1)

        # Count how many children are lost per parent
        child_loss_counts = jnp.bincount(
            jnp.clip(parents, 0),  # -1 -> 0 but ignored below
            length=tree._num_children.shape[0]
        )

        # Subtract children counts
        tree = tree.replace(
            _num_children=tree._num_children - child_loss_counts
        )

        return tree

    return jax.lax.while_loop(cond_fn, body_fn, tree)


def clean_states(tree: SSTree) -> SSTree:
    """Removes fully pruned states and compacts arrays."""
    kept_idxs = jnp.nonzero(tree._kil, size=tree.num_states, fill_value=-1)[0]

    tree = tree.replace(
        _isactive=tree._isactive[kept_idxs],
        _kil=tree._kil[kept_idxs],
        _c_spaces=tree._c_spaces[kept_idxs],
        _controls=tree._controls[kept_idxs],
        _cost_to_come=tree._cost_to_come[kept_idxs],
        _cost_total=tree._cost_total[kept_idxs],
        _parent_idxs=tree._parent_idxs[kept_idxs],
        _num_children=tree._num_children[kept_idxs],
        num_states=kept_idxs.shape[0],
    )

    return tree

def process_new_masks(
    tree: SSTree,
    fresh_mask: jnp.ndarray,
    dominating_states_mask: jnp.ndarray,
    propagate_origin_idx: jnp.ndarray,
    dominating_states_idx: jnp.ndarray,
    dominating_states_witness_idx: jnp.ndarray,
) -> tuple[SSTree, jnp.ndarray]:
    """
    Processes masks for new and dominating states:
    - Updates parent child counts
    - Updates witnesses and representative indices
    """

    # 2. Increment child counters for parents of fresh and dominating states
    tree = tree.replace(
        _num_children=tree._num_children.at[propagate_origin_idx[fresh_mask]].add(1)
    )
    tree = tree.replace(
        _num_children=tree._num_children.at[propagate_origin_idx[dominating_states_mask]].add(1)
    )

    # 4. Overthrow old reps for dominating states
    dominating_states_witness_idx = dominating_states_witness_idx[dominating_states_mask]
    old_rep_idx = tree._rep_idxs[dominating_states_witness_idx]

    tree = tree.replace(_isactive=tree._isactive.at[old_rep_idx].set(False))
    tree = tree.replace(_rep_idxs=tree._rep_idxs.at[dominating_states_witness_idx].set(dominating_states_idx))

    return tree


def prune_too_expensive(tree: SSTree, min_cost: int) -> SSTree:
    """
    Batched pruning for nodes that are too expensive:
    - Marks any node with cost_to_come > min_cost + 1 as inactive
    - Sets num_children of these nodes to 0 directly
    - Avoids the while loop in batch_prune
    """

    too_expensive_mask = tree._cost_to_come > min_cost + 1
    tree = tree.replace(
        _isactive=tree._isactive.at[too_expensive_mask].set(False),
        _num_children=tree._num_children.at[too_expensive_mask].set(0)
    )

    return tree
