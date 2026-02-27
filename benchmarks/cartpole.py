import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
import mujoco
from mujoco import mjx
import rrtree
import helper
import propagate
import params
import time
import gc
import argparse
from params import MJXparams, SSTparams, Callables, MotionConstraints, PhysicsConstants, Position, Bounds

# ------------------------------------------------------------------
# CARTPOLE SPECIFIC HELPERS
# ------------------------------------------------------------------

@jax.jit
def dist_cartpole(sim_params, diff):
    # diff: [d_cart_x, d_pole_theta, d_cart_v, d_pole_v]
    # Weights prioritize pole angle and cart position
    weights = jnp.array([1.0, 5.0, 0.1, 0.1])
    return jnp.sum(weights * (diff**2), axis=-1)

@jax.jit
def reached_goal_cartpole(states, goal_state, radius):
    # Check cart position and pole angle
    # state: [x, theta, v, omega]
    diff = states[:, :2] - goal_state[:2]
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1))
    return dist < radius

@partial(jax.jit, static_argnums=(0,))
def sample_cartpole(sim_params, key):
    # state: [x, theta, v, omega]
    k1, k2 = jax.random.split(key)
    # Range: x [-2.4, 2.4], theta [-0.2, 0.2] for stability
    pos = jax.random.uniform(k1, (sim_params.batch_size, 2), 
                             minval=jnp.array([-2.4, -0.2]), 
                             maxval=jnp.array([2.4, 0.2]))
    vel = jax.random.uniform(k2, (sim_params.batch_size, 2), 
                             minval=-1.0, maxval=1.0)
    return jnp.concatenate([pos, vel], axis=-1)

@partial(jax.jit, static_argnums=(0,))
def sample_actions_cartpole(sim_params, key):
    return jax.random.uniform(key, (sim_params.batch_size, 1), 
                              minval=-10.0, maxval=10.0)

# ------------------------------------------------------------------
# MJX PROPAGATION
# ------------------------------------------------------------------

def make_cartpole_propagate(mjx_model):
    @jax.jit
    def prop_fn(start_states, actions, obstacles, sst_params, sim_params):
        # start_states: (B, 4) -> [q, dq]
        def integration_step(carry, unused):
            d, ctrl = carry
            # Set state and control
            d = d.replace(qpos=d.qpos.at[:].set(d.qpos), qvel=d.qvel.at[:].set(d.qvel))
            d = d.replace(ctrl=ctrl)
            d = mjx.step(mjx_model, d)
            return (d, ctrl), None

        # Vectorized rollout using vmap
        def single_rollout(s, u):
            data = mjx.make_data(mjx_model)
            # Cartpole qpos: [x, theta], qvel: [v, omega]
            data = data.replace(qpos=s[:2], qvel=s[2:])
            
            # Sub-stepping for 'time_to_evolve'
            steps = int(sst_params.time_to_evolve)
            (data_final, _), _ = jax.lax.scan(integration_step, (data, u), None, length=steps)
            
            state_end = jnp.concatenate([data_final.qpos, data_final.qvel])
            # Simple boundary check for valid mask
            is_valid = (jnp.abs(state_end[0]) < 4.8) & (jnp.abs(state_end[1]) < 1.0)
            dist = jnp.linalg.norm(state_end[:2] - s[:2])
            return state_end, is_valid, dist

        return jax.vmap(single_rollout)(start_states, actions)
    return prop_fn

# ------------------------------------------------------------------
# RRT INFRASTRUCTURE
# ------------------------------------------------------------------

SIM_PARAMS_RESERVED = None
CALLABLES_RESERVED = None
TIERS = [512, 4096, 16384, 64536, 128000]

def nn_tier_factory(size):
    def nn_fn(operands):
        states, tree_size, query = operands
        parents, _ = helper.nearest_neighbor_masked(
            SIM_PARAMS_RESERVED, CALLABLES_RESERVED.dist_fn, 
            states[:size], tree_size, query
        )
        return parents
    return nn_fn

NN_BRANCHES = [nn_tier_factory(t) for t in TIERS]

@partial(jax.jit, static_argnums=(3, 4, 5))
def rrt_iteration(tree, rng_key, obstacles, sst_params, sim_params, callables):
    B = sim_params.batch_size
    A = 16 
    K = B // A 

    rng_key, sk1, sk2 = jax.random.split(rng_key, 3)
    seed_pts = callables.sample_fn(sim_params, sk1)[:K]

    branch_idx = jnp.minimum(jnp.digitize(tree.tree_size, jnp.array(TIERS)), len(TIERS) - 1)
    parents_small = jax.lax.switch(branch_idx, NN_BRANCHES, (tree.states, tree.tree_size, seed_pts))

    start_states = jnp.repeat(tree.states[parents_small], A, axis=0)
    parents = jnp.repeat(parents_small, A, axis=0)
    actions = callables.sampact_fn(sim_params, sk2)

    states_end, valid_mask, dist_traveled = callables.prop_fn(
        start_states, actions, obstacles, sst_params, sim_params
    )

    # Filter/Padding logic
    valid_mask = valid_mask.at[-1].set(False)
    num_new = jnp.sum(valid_mask)
    valid_idx = jnp.nonzero(valid_mask, size=B, fill_value=-1)[0]

    new_states = states_end[valid_idx]
    new_costs = tree.costs[parents[valid_idx]] + dist_traveled[valid_idx]

    tree, start_idx = rrtree.add_nodes(tree, new_states, actions[valid_idx], parents[valid_idx], new_costs, num_new)
    goal_mask = callables.goal_fn(new_states, sst_params.goal_vec, sst_params.goal_radius)

    return tree, rng_key, goal_mask, jnp.sum(goal_mask), new_states, start_idx

@partial(jax.jit, static_argnums=(1, 2, 3))
def jit_while(tree, sst_params, sim_params, callables, obstacles, i):
    def body_fn(carry):
        tree, key, goal_mask, goal_count, _, _, iter_cnt = carry
        key, subkey = jax.random.split(key)
        tree, subkey, goal_mask, goal_count, states, start_idx = rrt_iteration(
            tree, subkey, obstacles, sst_params, sim_params, callables
        )
        return (tree, key, goal_mask, goal_count, states, start_idx, iter_cnt + 1)

    def cond_fn(carry):
        tree, _, _, goal_count, _, _, _ = carry
        return (goal_count == 0) & (tree.tree_size < MAX_TREE_SIZE - sim_params.batch_size)

    init_carry = (tree, jax.random.PRNGKey(i), jnp.zeros(sim_params.batch_size, dtype=bool),
                  jnp.array(0, dtype=jnp.int32), jnp.zeros([sim_params.batch_size, sim_params.dims]),
                  jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32))
    
    return jax.lax.while_loop(cond_fn, body_fn, init_carry)

# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------

MAX_TREE_SIZE = 128000

if __name__ == "__main__":
    # --- MuJoCo Setup ---
    # Load the XML string provided in the prompt or from file
    model = mujoco.MjModel.from_xml_path("cartpole.xml") 
    mjx_model = mjx.put_model(model)
    
    sim_params = MJXparams(
        motion_constraints=MotionConstraints(max_vel=2.0, min_vel=-2.0, max_accel=10.0, min_accel=-10.0),
        physics_constants=PhysicsConstants(),
        batch_size=4096,
        bounds=Bounds(min_x=-4.8, max_x=4.8, min_y=-1.0, max_y=1.0),
        dims=4,         # [x, theta, v, omega]
        action_dims=1,  # [force]
        dt=0.02
    )

    sst_params = SSTparams(
        batch_size=4096,
        goal=Position(x=0.0, y=0.0, z=0.0), # y is theta here
        goal_radius=0.1,
        time_to_evolve=5 # 5 MJX steps per edge
    )
    # Helper to treat goal as a vector for JIT
    sst_params.goal_vec = jnp.array([0.0, 0.0, 0.0, 0.0]) 

    callables = Callables(
        prop_fn=make_cartpole_propagate(mjx_model),
        valid_fn=None, # Handled inside prop_fn for MJX
        sample_fn=sample_cartpole,
        dist_fn=dist_cartpole,
        sampact_fn=sample_actions_cartpole,
        goal_fn=reached_goal_cartpole
    )

    SIM_PARAMS_RESERVED = sim_params
    CALLABLES_RESERVED = callables
    obstacles = jnp.array([]) # No obstacles in standard cartpole

    # --- Warmup ---
    print("Compiling MJX Cartpole Kernels...")
    tree = rrtree.KinoTree.init(MAX_TREE_SIZE, sim_params.dims, sim_params.action_dims)
    init_state = jnp.array([0.0, 0.8, 0.0, 0.0]) # Start with pole leaned over
    tree, _ = rrtree.add_nodes(tree, init_state, jnp.zeros(1), -1, 0.0, 1)
    _ = jit_while(tree, sst_params, sim_params, callables, obstacles, 0)
    print("Compilation Complete.")

    # --- Run Loop ---
    for i in range(10):
        gc.collect()
        tree = rrtree.KinoTree.init(MAX_TREE_SIZE, sim_params.dims, sim_params.action_dims)
        tree, _ = rrtree.add_nodes(tree, init_state, jnp.zeros(1), -1, 0.0, 1)
        
        start_t = time.perf_counter()
        res = jit_while(tree, sst_params, sim_params, callables, obstacles, i)
        tree, _, goal_mask, count, _, start_idx, iters = jax.block_until_ready(res)
        duration = time.perf_counter() - start_t
        
        print(f"Run {i} | Goal: {count > 0} | Iters: {iters} | Time: {duration:.3f}s | Nodes: {tree.tree_size}")