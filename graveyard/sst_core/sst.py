from dataclasses import dataclass
import jax.numpy as jnp
from queue import PriorityQueue
from sst_core.helpers import *

class SSTparams:
    # Contains δBN, δs, min/max 2D sampling
    def __init__(self, 
                 batch_size: int, 
                 dBN: float, 
                 ds: float, 
                 min_x: float, max_x: float, min_y: float, max_y: float, 
                 start, dist_goal,
                 heuristic,
                 heuristic_weight = 0.2
                ):

        # Batch size for sampling
        self.batch_size = batch_size
        
        # Scale for how many distance units are equal to one radian of rotation
        # Increase to make SST more strict about needing similar angles
        # Used as metric in nearest neighbors lookup.
        self.rotation_metric_scale = 10
                
        # Radius for best-nearest search, for selecting which node to expand
        self.dBN = dBN 
        # Radius for local best search, for pruning dominated nodes
        self.ds = ds 
        
        # δBN + 2 * δs will be the clearance radius of the plan
        
        # Bounding box for the 2D space
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        
        self.start = start
        self.dist_goal = dist_goal
        self.heuristic = heuristic
        
        
        # Weightage for the cost-to-go from the geometric solution, used
        # in finding total heuristic cost
        self.heuristic_weight = heuristic_weight
        
        # Store final states in goal
        # Each entry is a dict {cspace, bodies, bending_control, cost_to_come, cost_total, tip}
        # Yeah params isn't the right place for this but it's good enough
        self.solutions = PriorityQueue()
        

@dataclass
class Node:
    state: StateWrapper
    parent: int  # index of parent node
    cost: float
    depth: int
    action: jnp.ndarray

@dataclass
class Witness:
    state: jnp.ndarray
    rep: int  # index of representative node

def sample_3D_state(params, batch_size):
    x = jnp.random.uniform(params.min_x, params.max_x, batch_size)
    y = jnp.random.uniform(params.min_y, params.max_y, batch_size)
    theta = jnp.random.uniform(-jnp.pi, jnp.pi, batch_size)
    
    return jnp.stack([x, y, theta], axis=1)

# Best-nearest selection
# Find rep with lowest cost within dBN otherwise nearest
def bn_selection(
        params: SSTparams,
        active: jnp.ndarray,
        active_costs: jnp.ndarray,
        batch_size: int
    ):
    xrand = sample_3D_state(params, batch_size)
    dist2 = nearest_neighbor_all(params, active, xrand)
    inside_mask = dist2 <= (params.δBN ** 2) # Shape (B, K)
    
    big_val = 1e15
    
    # Get the inside point with the lowest cost
    # If there are no inside points, behave unpredictably
    masked_costs = jnp.where(inside_mask, active_costs[indices], big_val)
    # Get the index of the active tip (per xrand) that has the minimum cost
    min_idx = masked_costs.argmin(axis=1)
    # Convert the indices to the actual tip positions
    cheapest_inside_point_idx = indices[min_idx]
    
    # Get the closest active state to the random state
    nearest_idx = jnp.argmin(dist2, axis=1)
    nearest_tip_idx = indices[nearest_idx]
    
    # If at least one neighbor is within δBN, return the nearest active state,
    # else return the active state with the minimum cost
    any_inside = inside_mask.any(axis=1)
    result = np.where(any_inside, cheapest_inside_point_idx, nearest_tip_idx)
    
    assert result.shape == (batch_size,)
    
    return result, xrand




def step_sst_batched(
    tree: SSTree,
    simulator,
    sampler,
    heuristic,
    validity_checker,
    distance_fn,
    selection_radius,
    pruning_radius,
    max_steps
):
    for _ in range(max_steps):
        x_rand = sampler.sample()

        idx, _ = tree.find_nearest_node(x_rand, distance_fn)
        x_near = tree.nodes[idx].state

        u = sampler.sample_action()
        x_new, cost = simulator.simulate(x_near, u)

        if not validity_checker.is_valid(x_new):
            continue

        witness_idx, w_dist = tree.find_witness(x_new, distance_fn)
        witness = tree.witnesses[witness_idx]

        if witness.rep is None or cost < tree.nodes[witness.rep].cost:
            new_node = Node(state=x_new, parent=idx, cost=cost, depth=tree.nodes[idx].depth + 1)
            tree.add_node(new_node)
            witness.rep = len(tree.nodes) - 1

    return tree
