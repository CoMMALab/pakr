import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
import time
import gc
import rrtree
import propagate
import helper
from params import Bounds, Position, MotionConstraints, PhysicsConstants, MJXparams, SSTparams, Callables

# ------------------------------------------------------------------
# ACROBOT DYNAMICS (UNCHANGED)
# ------------------------------------------------------------------

@jax.jit
def propagate_acrobot(states, actions, dt, constants):
    l1, l2 = 1.0, 1.0
    m1, m2 = 1.0, 1.0
    lc1, lc2 = 0.5, 0.5
    I1, I2 = 0.33, 0.33
    g = 9.81

    q1, q2, dq1, dq2 = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
    u = actions[:, 0]

    det = I1*I2 + I2*(l1**2)*m2 - (l1**2)*(lc2**2)*(m2**2)*(jnp.cos(q2)**2)

    t1 = -I2 * (g*lc1*m1*jnp.sin(q1) + g*m2*(l1*jnp.sin(q1) + lc2*jnp.sin(q1+q2))
         - 2.*l1*lc2*m2*dq1*dq2*jnp.sin(q2) - l1*lc2*m2*(dq2**2)*jnp.sin(q2))
    t2 = (I2 + l1*lc2*m2*jnp.cos(q2)) * (g*lc2*m2*jnp.sin(q1+q2)
         + l1*lc2*m2*(dq1**2)*jnp.sin(q2) - u)
    q1_ddot = (t1 + t2) / det

    t3 = (I2 + l1*lc2*m2*jnp.cos(q2)) * (g*lc1*m1*jnp.sin(q1)
         + g*m2*(l1*jnp.sin(q1) + lc2*jnp.sin(q1+q2))
         - 2.*l1*lc2*m2*dq1*dq2*jnp.sin(q2)
         - l1*lc2*m2*(dq2**2)*jnp.sin(q2))
    t4 = (g*lc2*m2*jnp.sin(q1+q2)
         + l1*lc2*m2*(dq1**2)*jnp.sin(q2) - u) * \
         (I1 + I2 + (l1**2)*m2 + 2.*l1*lc2*m2*jnp.cos(q2))
    q2_ddot = (t3 - t4) / det

    new_dq1 = dq1 + q1_ddot * dt
    new_dq2 = dq2 + q2_ddot * dt
    new_q1 = (q1 + new_dq1 * dt + jnp.pi) % (2 * jnp.pi) - jnp.pi
    new_q2 = (q2 + new_dq2 * dt + jnp.pi) % (2 * jnp.pi) - jnp.pi

    return jnp.stack([new_q1, new_q2, new_dq1, new_dq2], axis=-1)


@jax.jit
def dist_acrobot(sim_params, diff):
    dq = (diff[..., 0:2] + jnp.pi) % (2 * jnp.pi) - jnp.pi
    dv = diff[..., 2:4]
    w = jnp.array([0.5, 0.5, 0.2])
    angular = jnp.abs(dq[..., 0]) * w[0] + jnp.abs(dq[..., 1]) * w[1]
    velocity = jnp.linalg.norm(dv, axis=-1) * w[2]
    return (angular + velocity).T


@jax.jit
def reached_goal_acrobot(states, goal_vec, radius):
    diff = states - goal_vec
    diff_angles = (diff[:, 0:2] + jnp.pi) % (2 * jnp.pi) - jnp.pi
    dist_sq = jnp.sum(diff_angles**2, axis=-1) + 0.1 * jnp.sum(diff[:, 2:4]**2, axis=-1)
    return dist_sq < radius**2


@partial(jax.jit, static_argnums=(0,))
def sample_acrobot(sim_params, key):
    B = sim_params.batch_size
    k1, k2 = jax.random.split(key)
    angles = jax.random.uniform(k1, (B, 2), minval=-jnp.pi, maxval=jnp.pi)
    vels = jax.random.uniform(k2, (B, 2), minval=-8.0, maxval=8.0)
    return jnp.concatenate([angles, vels], axis=-1)


@partial(jax.jit, static_argnums=(0,))
def sample_actions_acrobot(sim_params, key):
    return jax.random.uniform(key, (sim_params.batch_size, 1),
                              minval=-10.0, maxval=10.0)


@partial(jax.jit, static_argnums=(1))
def valid_acrobot(state, params, obstacles):
    return jnp.all(jnp.abs(state[:, 2:4]) < 8.0, axis=-1)


# ------------------------------------------------------------------
# AO-X CORE
# ------------------------------------------------------------------

MAX_TREE_SIZE = 2_000_000
TIERS = [512, 1024, 4096, 16384, 32768, 65536,
         131072, 262144, 524288, 1_000_000, 2_000_000]

SIM_PARAMS_RESERVED = None
CALLABLES_RESERVED = None


def nn_tier_factory(size):
    def nn_fn(operands):
        states, tree_size, query = operands
        return helper.nearest_neighbor_masked(
            SIM_PARAMS_RESERVED,
            CALLABLES_RESERVED.dist_fn,
            states[:size], tree_size, query
        )[0]
    return nn_fn


NN_BRANCHES = [nn_tier_factory(t) for t in TIERS]


@partial(jax.jit, static_argnums=(3, 4, 5))
def aox_iteration(tree, rng_key, goal_vec,
                  sst_params, sim_params, callables,
                  best_cost):

    B, A = sim_params.batch_size, 64
    K = B // A

    rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)

    seed_pts = callables.sample_fn(sim_params, subkey1)[:K]
    branch_idx = jnp.minimum(
        jnp.digitize(tree.tree_size, jnp.array(TIERS)),
        len(TIERS) - 1
    )

    parents_small = jax.lax.switch(
        branch_idx, NN_BRANCHES,
        (tree.states, tree.tree_size, seed_pts)
    )

    start_states = jnp.repeat(tree.states[parents_small], A, axis=0)
    parents = jnp.repeat(parents_small, A, axis=0)
    actions = callables.sampact_fn(sim_params, subkey2)

    states_end, valid_mask, _ = propagate.rollout_final_2d(
        start_states, actions, jnp.array([]),
        sst_params, sim_params, callables
    )

    valid_mask = valid_mask.at[-1].set(False)

    step_cost = sim_params.dt * sst_params.time_to_evolve
    new_costs = tree.costs[parents] + step_cost

    diff = states_end[:, 0:2] - goal_vec[0:2]
    diff = (diff + jnp.pi) % (2 * jnp.pi) - jnp.pi
    h = jnp.linalg.norm(diff, axis=-1)

    ao_mask = (new_costs + h <= best_cost) & valid_mask

    num_new = jnp.sum(ao_mask)
    ao_idx = jnp.nonzero(ao_mask, size=B, fill_value=-1)[0]

    tree, start_idx = rrtree.add_nodes(
        tree,
        states_end[ao_idx],
        actions[ao_idx],
        parents[ao_idx],
        new_costs[ao_idx],
        num_new
    )

    goal_mask = callables.goal_fn(
        states_end[ao_idx], goal_vec,
        sst_params.goal_radius
    )

    return tree, rng_key, goal_mask, jnp.sum(goal_mask), start_idx


@partial(jax.jit, static_argnums=(1, 2, 3))
def jit_while(tree, sst_params, sim_params,
              callables, best_cost, seed, goal_vec):

    def body_fn(carry):
        tree, key, goal_mask, goal, start_idx = carry
        key, subkey = jax.random.split(key)
        tree, subkey, goal_mask, goal, start_idx = aox_iteration(
            tree, subkey, goal_vec,
            sst_params, sim_params,
            callables, best_cost
        )
        return (tree, key, goal_mask, goal, start_idx)

    def cond_fn(carry):
        tree, key, goal_mask, goal, start_idx = carry
        return (goal == 0) & \
               (tree.tree_size < MAX_TREE_SIZE - sim_params.batch_size)

    init = (
        tree,
        jax.random.PRNGKey(seed),
        jnp.zeros(sim_params.batch_size, dtype=bool),
        jnp.array(0),
        jnp.array(0)
    )

    return jax.lax.while_loop(cond_fn, body_fn, init)


# ------------------------------------------------------------------
# EXECUTION WITH AGGREGATED STATS
# ------------------------------------------------------------------

if __name__ == "__main__":

    batch_size = 32768
    goal_vec = jnp.array([jnp.pi, 0.0, 0.0, 0.0])

    sim_params = MJXparams(
        motion_constraints=MotionConstraints(max_accel=10.0, min_accel=-10.0),
        physics_constants=PhysicsConstants(),
        batch_size=batch_size,
        bounds=Bounds(min_x=-jnp.pi, max_x=jnp.pi,
                      min_y=-jnp.pi, max_y=jnp.pi),
        dims=4, action_dims=1, dt=0.01, seed=42
    )

    sst_params = SSTparams(
        batch_size=batch_size,
        δBN=0.1, δs=0.05, decay=0.8,
        start=Position(x=jnp.pi, y=0.0, z=0.0),
        goal=Position(x=0.0, y=0.0, z=0.0),
        goal_radius=0.25,
        time_to_evolve=23,
    )

    callables = Callables(
        prop_fn=propagate_acrobot,
        valid_fn=valid_acrobot,
        sample_fn=sample_acrobot,
        dist_fn=dist_acrobot,
        sampact_fn=sample_actions_acrobot,
        goal_fn=reached_goal_acrobot
    )

    SIM_PARAMS_RESERVED, CALLABLES_RESERVED = sim_params, callables

    print("Pre-compiling...")
    dummy_tree = rrtree.KinoTree.init(MAX_TREE_SIZE,
                                      sim_params.dims,
                                      sim_params.action_dims)
    _ = jit_while(dummy_tree, sst_params, sim_params,
                  callables, jnp.inf, 0, goal_vec)
    print("Compilation complete.\n")

    # ---------------- OUTER BENCHMARK LOOP ----------------

    N_RUNS = 100
    MAX_TIME = 5.0
    COST_THRESHOLD = 4.5

    run_costs = []
    run_times = []
    run_iters = []

    for run in range(N_RUNS):

        best_cost = float('inf')
        t0 = time.perf_counter()
        ao_iter = 0

        while True:
            gc.collect()

            tree = rrtree.KinoTree.init(
                MAX_TREE_SIZE,
                sim_params.dims,
                sim_params.action_dims
            )

            init_state = jnp.array([0.0, 0.0, 0.0, 0.0])
            tree, _ = rrtree.add_nodes(
                tree, init_state, jnp.zeros(1), -1, 0.0, 1
            )

            seed = np.random.randint(0, 2**31 - 1)

            tree, key, goal_mask, goal_found, start_idx = \
                jax.block_until_ready(
                    jit_while(tree, sst_params, sim_params,
                              callables,
                              jnp.array(best_cost),
                              seed,
                              goal_vec)
                )

            if goal_found > 0:
                sol_cost = float(tree.costs[start_idx +
                                  jnp.argmax(goal_mask)])
                best_cost = min(best_cost, sol_cost)

            ao_iter += 1
            elapsed = time.perf_counter() - t0

            if elapsed >= MAX_TIME or best_cost <= COST_THRESHOLD:
                break

        run_costs.append(best_cost)
        run_times.append(elapsed)
        run_iters.append(ao_iter)

        print(f"Run {run:02d}: Cost={best_cost:.3f} | "
              f"Time={elapsed:.2f}s | AO iters={ao_iter}")

    # ---------------- AGGREGATED STATISTICS ----------------

    costs = np.array(run_costs)
    times = np.array(run_times)

    successes = costs < float('inf')

    print("\n" + "="*50)
    print(f"SUCCESS RATE: {np.sum(successes)}/{N_RUNS}")
    print("="*50)

    if np.any(successes):
        print(f"Mean Cost:   {np.mean(costs[successes]):.3f}")
        print(f"Median Cost: {np.median(costs[successes]):.3f}")
        print(f"Mean Time:   {np.mean(times[successes]):.3f}")
        print(f"Median Time: {np.median(times[successes]):.3f}")
        print(f"Mean AO Iters: {np.mean(run_iters):.2f}")

    print("="*50)