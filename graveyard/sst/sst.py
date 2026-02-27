import jax
from jax import jit
from functools import partial
import jax.numpy as jnp
import time
import helper
import propagate
from queue import PriorityQueue
from max_cover import max_cover
import params
import heapq
import argparse
import itertools
import sstree


@partial(jit, static_argnums=(0, 1, 5))
def best_first_selection(
        sst_params: params.SSTparams,
        sim_params: params.MJXparams,
        active: jnp.ndarray,
        active_costs: jnp.ndarray,
        epoch: int,
        callables: params.Callables,
        key,
    ):
    """
    Decides the closest active state to start growing from, returns:
        - The nearest active state if at least one neighbor is within δBN
        - The active state with the minimum cost otherwise
    """    
    # 1) xrand ← Sample_State(X);
    batch_size = sim_params.batch_size
    xrand = callables.sample_fn(sim_params, key)
    
    # 2. Xnear ← Near(V, xrand, δBN);
    # Compute the distance from xrand to all active states
    dist2 = helper.dist_all(sim_params, callables.dist_fn, active, xrand)
    
    # Find the (B, K) indices of the neighbors within δBN
    δBN = sst_params.δBN * sst_params.decay ** epoch
    # δBN mask over (K, B)
    inside_mask = dist2 <= (δBN ** 2)                      # (K, B)

    # Broadcast costs to (K, B)
    big_val = 1e15
    costs_grid = jnp.broadcast_to(active_costs[:, None], dist2.shape)  # (K, B)
    masked_costs = jnp.where(inside_mask, costs_grid, big_val)         # (K, B)

    # Cheapest inside per *column* (per sample)
    cheapest_row_per_col = jnp.argmin(masked_costs, axis=0)            # (B,)

    # Nearest (by distance) per *column*
    nearest_row_per_col = jnp.argmin(dist2, axis=0)                    # (B,)

    # Any inside per *column*
    any_inside_per_col = jnp.any(inside_mask, axis=0)                  # (B,)

    # Choose cheapest-inside if any, else nearest
    result = jnp.where(any_inside_per_col, cheapest_row_per_col, nearest_row_per_col)  # (B,)

    
    assert result.shape == (batch_size,)
    
    return result, xrand

@partial(jit, static_argnums=(0, 1, 2))
def check_dominating_nodes(
    params: params.SSTparams,
    sim_params: params.MJXparams,
    dist_fn: callable,
    new_states: jnp.ndarray,     # Shape (B, 3)
    new_costs: jnp.ndarray,    # Shape (B,)
    witnesses: jnp.ndarray,  # Shape (M, 3)
    rep_costs: jnp.ndarray,     # Shape (M,)
    epoch: int
    ):
    """
    Given some new states to add, determine if:
      - They are in a new cell (distance > δs), and are added automatically. We later make a witness in its place too.
      - They are in the cell of a witness, and are added if they are better than the witness's current rep
    """
    
    assert new_states.shape[0] == new_costs.shape[0], f"new_states shape: {new_states.shape}, xnew_costs shape: {new_costs.shape}"
    assert witnesses.shape[0] == rep_costs.shape[0], f"witnesses shape: {witnesses.shape}, rep_costs shape: {rep_costs.shape}"
    
    indices, dist2 = helper.nearest_neighbor(sim_params, dist_fn, witnesses, new_states) # Shape (B)
    assert indices.ndim == 1 and dist2.ndim == 1
    
    # Find the mask of states that are within δs
    δs = params.δs * params.decay ** epoch

    within_cell = dist2 <= (δs) ** 2
    
    # 2) If distance > δs, we automatically add x_new
    fresh_mask = ~within_cell
        
    # 3) If distance < δs, only add x_new if x_new_cost < cost of that nearest rep.]
    dominating_states_mask = within_cell & (new_costs <= rep_costs[indices])
        
    dominating_states_witness_idx = jnp.where(dominating_states_mask, indices, -1)

    
    assert dominating_states_mask.shape == dominating_states_mask.shape
    assert fresh_mask.shape == (new_states.shape[0],), f"to_add_fresh_mask shape: {fresh_mask.shape}, new_states shape: {new_states.shape}"
    
    assert dominating_states_mask.shape == (new_states.shape[0],)
    
    # Fresh states are disjoint from dominating states
    return fresh_mask, dominating_states_mask, dominating_states_witness_idx

class Solution():
    def __init__(self, cost, data):
        self.cost = cost
        self.data = data

@partial(jax.jit, static_argnums=(0,))
def refactor_results(
    n: int,
    actions: jnp.ndarray,             # (batch, act_dim)
    new_states: jnp.ndarray,          # (T, batch, dims)
    dist_traveled: jnp.ndarray,       # (T, batch)
    kill: jnp.ndarray,                # (batch,)
    costs: jnp.ndarray,               # (batch,)
    propagate_origin_idx: jnp.ndarray # (batch,)
):
    """
    Flattens new_states, actions, distances, and costs into per-valid-state arrays.
    Optionally skips every n states if n > 0.
    """
    T, batch, dims = new_states.shape

    # 1. Build mask of valid states: (T, batch)
    mask = jnp.arange(T)[:, None] < kill[None, :]
    # 2. Flatten everything
    states_flat = new_states.transpose(1, 0, 2).reshape(-1, dims)               # (T*batch, dims)
    dist_flat = dist_traveled.T.reshape(-1)                     # (T*batch,)
    actions_flat = jnp.repeat(actions, T, axis=0)             # (batch*T, act_dim)
    costs_flat = jnp.repeat(costs, T)                         # (batch*T,)
    origin_idx_flat = jnp.repeat(propagate_origin_idx, T)      # (batch*T,)
    timesteps_flat = jnp.tile(jnp.arange(T), batch)           # (T*batch,)

    # 3. Keep only valid states using indices
    valid_indices = jnp.nonzero(mask.T.reshape(-1), size=states_flat.shape[0])[0]
    states_valid = states_flat[valid_indices]
    dist_valid = dist_flat[valid_indices]
    actions_valid = actions_flat[valid_indices]
    timesteps_valid = timesteps_flat[valid_indices]
    costs_valid = costs_flat[valid_indices]
    origin_idx_valid = origin_idx_flat[valid_indices]

    # 5. Update costs: add per-state distance traveled
    new_costs = costs_valid + dist_valid

    # 6. Build final actions_out with timestep + distance appended
    actions_out = jnp.concatenate(
        [actions_valid, timesteps_valid[:, None] + 1, dist_valid[:, None]],
        axis=1
    )

    n_valid = jnp.sum(mask)
    return states_valid, actions_out, new_costs, origin_idx_valid, n_valid


def sst(sst_params: params.SSTparams, sim_params: params.MJXparams, tree, iters, epoch, callables, obstacles,):
    # init tree if first iter of sst*
    if not tree:
        init = jnp.concatenate([jnp.asarray([sst_params.start.x, sst_params.start.y, sst_params.start.z]), jnp.zeros(sim_params.dims - 3, dtype=jnp.float32)], axis=0)
        controls = jnp.zeros(sim_params.action_dims+2)
        tree = sstree.init_tree(sim_params.dims, sim_params.action_dims)
        cost_to_go = 0.0
        tree = sstree.add_states(tree,
                    isactive=True,
                    c_space=init,
                    controls=controls,
                    # Heuristic stuff
                    cost_to_come=0.0,
                    cost_total= cost_to_go,
                    parent_idx=-1,)
        
        tree = sstree.add_witnesses(tree, init, 0)

    for iter in range(iters):
        # ------------ Sample random tip positions and their closest active states ------------
        key = jax.random.PRNGKey(sim_params.seed + epoch * iters + iter)
        active_states_mask = tree._isactive
        active_states_idx, xrand = best_first_selection(sst_params, sim_params, tree._c_spaces[active_states_mask], tree._cost_to_come[active_states_mask], epoch, callables, key)
        propagate_origin_idx = jnp.where(active_states_mask)[0][active_states_idx] 
        
        actions = callables.sampact_fn(sim_params, key)
        
        print('Starting rollout')
        start_time = time.time()
        new_states, kill, dists = propagate.rollout(
            state0=tree._c_spaces[propagate_origin_idx],
            actions=actions,
            obstacles=obstacles,
            sst_params=sst_params,
            sim_params=sim_params,
            callables=callables,
        )
        print('Rollout time:', time.time() - start_time)
        start_time = time.time()
        new_states, current_controls, new_costs, propagate_origin_idx, n_valid = refactor_results(
            n=sst_params.sparsity,
            actions=actions,
            new_states=new_states,
            dist_traveled=dists,
            kill=kill,
            costs=tree._cost_to_come[propagate_origin_idx],
            propagate_origin_idx=propagate_origin_idx
        )
        new_states = new_states[:n_valid]
        new_costs = new_costs[:n_valid]
        propagate_origin_idx = propagate_origin_idx[:n_valid]
        current_controls = current_controls[:n_valid]
        print('Refactor time:', time.time() - start_time, 'with', n_valid, 'valid states')
        # --------- Find non-overlapping subset of states ---------
        if sst_params.do_set_cover:
            start_time = time.time()
            non_overlapping_mask = max_cover(sst_params, sim_params, new_states, epoch, callables.dist_fn) # FIXME jaxify, handle no-copy
            print('max_cover time:', time.time() - start_time, 'ended with', non_overlapping_mask.sum(), 'states')
            
            new_states = new_states[non_overlapping_mask]
            new_costs = new_costs[non_overlapping_mask]
            current_controls = current_controls[non_overlapping_mask]
            propagate_origin_idx = propagate_origin_idx[non_overlapping_mask]
            
        if sst_params.do_cost_to_go:
            pass
        
        #draw_dead_state(sim_params, xnew_cspaces, xnew_bodies, init_x, init_y, init_heading)
        
        # If any states falls in the goal region, add to the solutions
        # Append all info: bodies, cspaces, costs, bending controls

        start_time = time.time()
        in_goal_mask = helper.reached_goal(new_states, sst_params.goal, sst_params.goal_radius)
        solution_costs = new_costs[in_goal_mask]
        solution_states = new_states[in_goal_mask]
        solution_controls = current_controls[in_goal_mask]
        solution_parent_idx = propagate_origin_idx[in_goal_mask]
        solutions = (solution_costs, solution_states, solution_controls, solution_parent_idx)
        print('Goal check time:', time.time() - start_time, 'found', len(solution_costs), 'new solutions')
        # --------- Add new states to tree ---------
        # Get cost for each witness's rep, or -np.inf if has no rep
        start_time = time.time()
        witness_rep_costs = jnp.where(tree._rep_idxs[:tree.num_witnesses] > 0, tree._cost_total[tree._rep_idxs[:tree.num_witnesses]], -jnp.inf)
        
        # Fresh states don't touch any existing witness (will make new witnesses for them)
        # Dominating states fall inside an existing witness and has better cost than the witness's rep
        # (will replace the old rep with them)
        # All these masks index into xnew_*
        fresh_mask, dominating_states_mask, dominating_states_witness_idx = \
                            check_dominating_nodes(sst_params,
                            sim_params,
                            callables.dist_fn,
                            new_states, 
                            new_costs, 
                            tree._witness_positions[:tree.num_witnesses], 
                            witness_rep_costs,
                            epoch)
        print('Check dominating time:', time.time() - start_time, 'found', fresh_mask.sum() + dominating_states_mask.sum(), 'to add')
        start_time = time.time()
        # Add new witness-creating states to the tree, and record their indexes      
        tree = sstree.add_states(tree, isactive=True,
                                        c_space=new_states[fresh_mask],
                                        controls=current_controls[fresh_mask],
                                        # Heuristic stuff
                                        cost_to_come=new_costs[fresh_mask],
                                        cost_total=new_costs[fresh_mask],
                                        # Tree stuff
                                        parent_idx=propagate_origin_idx[fresh_mask],)
        xnew_fresh_idx = jnp.arange(tree.num_states, tree.num_states + jnp.sum(fresh_mask), dtype=jnp.int32)
        # Add new dominating states to the tree, and record their indexes
        tree = sstree.add_states(tree, isactive=True,
                                        c_space=new_states[dominating_states_mask],
                                        controls=current_controls[dominating_states_mask],
                                        # Heuristic stuff
                                        cost_to_come=new_costs[dominating_states_mask],
                                        cost_total=new_costs[dominating_states_mask],
                                        # Tree stuff
                                        parent_idx=propagate_origin_idx[dominating_states_mask],)
        # print(tree.num_states)
        # print(tree._num_children[:tree.num_states], tree._parent_idxs[:tree.num_states])
        dominating_states_idx = jnp.arange(tree.num_states, tree.num_states + jnp.sum(dominating_states_mask), dtype=jnp.int32)
        tree = sstree.add_witnesses(tree, new_states[fresh_mask], xnew_fresh_idx)
        print('Add states time:', time.time() - start_time, 'added', fresh_mask.sum() + dominating_states_mask.sum(), 'states')
        tree = sstree.process_new_masks(
            tree,
            fresh_mask,
            dominating_states_mask,
            propagate_origin_idx,
            dominating_states_idx,
            dominating_states_witness_idx
        )
        start_time = time.time()
        # Set states to inactive if they have 1 more cost than the best so far
        min_cost = solution_costs[0] if len(solution_costs) > 0 else 9999
        
        tree = sstree.prune_too_expensive(tree, min_cost)
        tree = sstree.batch_prune(tree)
        print('Prune time:', time.time() - start_time)

        if (solution_costs.shape[0] > 0):
            return tree, solutions
        
        print(tree.num_states)

    return tree, solutions


def sst_star(sst_params: params.SSTparams, sim_params: params.MJXparams, callables: params.Callables, obstacles: jnp.ndarray):
    solutions = []
    heapq.heapify(solutions)

    sst_iter_0 = 7
    sst_iter = sst_iter_0
    epoch = 0
    
    # Dimension of the state space
    d = sim_params.dims
    # Dimension of the control space
    l = sim_params.action_dims
    
    # Initial tree is None, sst will create it
    tree = None
    
    while True:
        # Clear the screen
        #clear_all_surfaces()
        
        tree, (solution_costs, solution_states, solution_controls, solution_parent_idx) = sst(sst_params, sim_params, tree, int(sst_iter), epoch, callables, obstacles)
        for i in range(len(solution_costs)):
            cost = float(solution_costs[i])  # convert to Python float
            sol = Solution(
                cost=cost,
                data={
                    'cspace': solution_states[i],
                    'action': solution_controls[i],
                    'cost_to_come': solution_costs[i],
                    'parent_idx': solution_parent_idx[i],
                }
            )

            heapq.heappush(solutions, (cost, next(counter), sol))
            
        if len(solutions) > 0:
            print(f'\033[92mFound {len(solutions)} solutions, best cost: {solutions[0][0]:.2f} \033[0m')
            return tree, solutions

        epoch += 1
        sst_iter = sst_iter_0 * (1 + jnp.log2(epoch)) * sst_params.decay ** (-1 * (d + l + 1) * epoch) 
        δs = sst_params.δs * (sst_params.decay ** epoch)
        # Clear the screen
        # clear_all_surfaces()
        
        # Look for reps that no longer fall within δs
        # And deactivate them, also search for prune candidates in their ancestors
        witnesses = tree._witness_positions[:tree.num_witnesses]
        witness_rep_idx_ = tree._rep_idxs[:tree.num_witnesses]
        valid_witness_mask = witness_rep_idx_ >= 0
        print(valid_witness_mask.sum(), 'valid witnesses out of', tree.num_witnesses)
        
        # Get the witnesses with valid reps
        witnesses = witnesses[valid_witness_mask]
        witness_rep_idx = witness_rep_idx_[valid_witness_mask]
        rep = tree._c_spaces[witness_rep_idx]
        
        # Find the ones with distance > δs
        distance_from_witness_to_rep2 = callables.dist_fn(sim_params, witnesses - rep)
        print(distance_from_witness_to_rep2, δs ** 2, sst_params.δs ** 2)
        to_remove_mask = distance_from_witness_to_rep2 > δs ** 2
        print(to_remove_mask.sum(), 'reps to remove')
        
        tree = sstree.batch_prune(tree)
            
        
        # TODO prune_path_at may introduce more invalid nodes
        # we should repeat until there are none
        
        # TODO consider taking the min possible segments K so far,
        # and removing all states with >= k segments (since we know for sure)
        # we can do better
        
        # tree = sstree.clean_states(tree)

profile = True
if __name__ == "__main__":
    counter = itertools.count()  # global counter
    parser = argparse.ArgumentParser(description='Run the SST planner.')
    parser.add_argument('--env', type=str, default='envs/tree.csv', help='Path to the environment config file.')
    parser.add_argument('--motion', type=str, default='di', help='Define motion type: Double Integrator (di), Dubins Airplane (da), Quadcopter (qc)')
    args = parser.parse_args()

    match args.motion:
        case 'di':
            sst_params = params.sst_params_DI
            sim_params = params.sim_params_DI
            callables = params.Callables()
        case 'da':
            sst_params = params.sst_params_DA
            sim_params = params.sim_params_DA
            callables = params.callables_DA
        case 'qc':
            sst_params = params.sst_params_QC
            sim_params = params.sim_params_QC
            callables = params.callables_QC
        case _:
            print("invalid motion type")
            exit

    obstacles = helper.get_obs(args.env)
    
    # jit partial for statics
    #best_first_selection_jit = jit(partial(best_first_selection, sim_params, callables))
    start = time.time()
    tree, solutions = sst_star(sst_params, sim_params, callables, obstacles)
    print('Total time:', time.time() - start)
    controls, states = helper.find_solution_path(tree, solutions)
    init = jnp.concatenate([jnp.asarray([sst_params.start.x, sst_params.start.y, sst_params.start.z]), jnp.zeros(sim_params.dims - 3, dtype=jnp.float32)], axis=0)
    print(controls)
    print(states)
    waypoints, states = helper.recreate_trajectory(init, controls, sim_params, callables.prop_fn)
    jnp.save('cache/waypoints.npy', waypoints)