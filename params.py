import jax
from flax import struct
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
#from mujoco import mjx
from queue import PriorityQueue
import propagate
import helper

@struct.dataclass
class MotionConstraints: #default is for quadcopter
    # general
    max_vel: jnp.float32 = 30.0
    min_vel: jnp.float32 = -30.0
    max_accel: jnp.float32 = 30.0
    min_accel: jnp.float32 = -30.0

    # dubins airplane
    max_yaw_rate: jnp.float32 = jnp.pi / 4.0
    min_yaw_rate: jnp.float32 = -jnp.pi / 4.0
    max_pitch_rate: jnp.float32 = jnp.pi / 6.0
    min_pitch_rate: jnp.float32 = -jnp.pi / 6.0

    max_yaw: jnp.float32 = jnp.pi
    min_yaw: jnp.float32 = -jnp.pi
    max_pitch: jnp.float32 = jnp.pi
    min_pitch: jnp.float32 = -jnp.pi

    # quadcopter
    max_thrust: jnp.float32 = 15.0
    min_thrust: jnp.float32 = 5.0
    max_torque: jnp.float32 = jnp.pi
    min_torque: jnp.float32 = -jnp.pi
    max_roll: jnp.float32 = jnp.pi
    min_roll: jnp.float32 = -jnp.pi
    max_angle_vel: jnp.float32 = 30.0
    min_angle_vel: jnp.float32 = -30.0

@struct.dataclass
class PhysicsConstants:
    g: jnp.float32 = 9.81
    m: jnp.float32 = 1.0
    IX: jnp.float32 = 1.0
    IY: jnp.float32 = 1.0
    IZ: jnp.float32 = 2.0
    NU: jnp.float32 = 10e-3
    MU: jnp.float32 = 2e-6

@struct.dataclass
class Callables:
    prop_fn: callable = propagate.propagate_double_integrator
    valid_fn: callable = helper.valid_DI
    sample_fn: callable = helper.sample_DI
    dist_fn: callable = helper.dist_DI
    sampact_fn: callable = helper.sample_actions_DI

@struct.dataclass
class Bounds:
    min_x: jnp.float32 = 0.0
    max_x: jnp.float32 = 1.0
    min_y: jnp.float32 = 0.0
    max_y: jnp.float32 = 1.0
    min_z: jnp.float32 = 0.0
    max_z: jnp.float32 = 1.0

@struct.dataclass
class MJXparams:
    motion_constraints: MotionConstraints
    physics_constants: PhysicsConstants
    batch_size: jnp.int32
    bounds: Bounds
    dims: jnp.int32
    action_dims: jnp.int32
    dt: jnp.float32 = 0.1
    seed: jnp.int32 = 0
    # model: any = None
    # ee_x_idx: jnp.int32 = -1
    # ee_y_idx: jnp.int32 = -1
    # act_x: jnp.int32 = -1
    # act_y: jnp.int32 = -1

    # def __post_init__(self):
    #     # Only initialize MJX model if a model object is provided
    #     if self.model is not None:
    #         mj_model = mjx.put_model(mujoco.MjModel.from_xml_path("my_model.xml"))
    #         object.__setattr__(self, "model", mj_model)
    #         object.__setattr__(self, "ee_x_idx", self.model.joint("ee_x").qposadr)
    #         object.__setattr__(self, "ee_y_idx", self.model.joint("ee_y").qposadr)
    #         object.__setattr__(self, "act_x", self.model.actuator("ee_x").id)
    #         object.__setattr__(self, "act_y", self.model.actuator("ee_y").id)

@struct.dataclass
class Position:
    x: jnp.float32 = 0.10
    y: jnp.float32 = 0.08
    z: jnp.float32 = 0.05

@struct.dataclass
class SSTparams:
    batch_size: jnp.int32
    # params needs to be immutable so we update these two elsewhere
    δBN: jnp.float32 # Initial radius for best-nearest search, for selecting which node to expand
    δs: jnp.float32 # Initial radius for local best search, for pruning dominated nodes, δBN + 2 * δs will be the clearance radius of the plan
    decay: jnp.int32
    start: Position
    goal: Position
    goal_radius: jnp.float32

    geo_cost_to_go_weight: jnp.float32 = 0.2
    do_cost_to_go: jnp.bool_ = True

    do_maximal: jnp.bool_ = True # Rollout as far as feasible, then use all timesteps as possible new candidates, else pick random timestep
    do_set_cover: jnp.bool_ = True # Do set cover for batched rollout results before adding witnesses
    time_to_evolve: jnp.int32 = 100
    sparsity: jnp.int32 = 0 # if > 0, only keep every n-th state in the trajectory

################################
# preset params for motion types
################################

cartpole_motion_constraints = MotionConstraints(
    max_vel = 5.0,
    min_vel = -5.0,
    max_accel = 2.0,
    min_accel = -2.0,
    max_angle_vel = jnp.pi,
    min_angle_vel = -jnp.pi
)

dubins_motion_constraints = MotionConstraints(
    max_vel = 0.3,
    min_vel = 0.0,
    max_accel = 0.3,
    min_accel = -0.3,
    max_pitch = jnp.pi / 3.0,
    min_pitch = -jnp.pi / 3.0)

double_integrator_motion_constraints = MotionConstraints(
    max_vel = 0.3,
    min_vel = -0.3,
    max_accel = 0.2,
    min_accel = -0.2)

bounds_qc = Bounds(
    min_x = 0.0,
    max_x = 100.0,
    min_y = 0.0,
    max_y = 100.0,
    min_z = 0.0,
    max_z = 100.0
)

start_simple = Position(
    x = 0.10,
    y = 0.08,
    z = 0.05
)

goal_simple = Position(
    x = 0.80,
    y = 0.95,
    z = 0.90
)

start_qc = Position(
    x = 10.0,
    y = 8.0,
    z = 5.0
)

goal_qc = Position(
    x = 80.0,
    y = 95.0,
    z = 90.0
)

#batch_size = 16384
batch_size = 32768
#batch_size = 131072
#batch_size = 65536
#batch_size = 4096
seed = 0
time_to_evolve = 10
# Safe params

# callables_MCP = Callables(
#     prop_fn=None,
#     valid_fn=helper.valid_MCP,
#     dist_fn=helper.dist_MCP,
#     sampact_fn=helper.sample_actions_MCP,
# )

callables_DA = Callables(
    prop_fn=propagate.propagate_dubins_airplane,
    valid_fn=helper.valid_DA,
    sample_fn=helper.sample_DA,
    dist_fn=helper.dist_DA,
    sampact_fn=helper.sample_actions_DA,
)

callables_QC = Callables(
    prop_fn=propagate.propagate_quadcopter,
    valid_fn=helper.valid_QC,
    sample_fn=helper.sample_QC,
    dist_fn=helper.dist_QC,
    sampact_fn=helper.sample_actions_QC,
)

sim_params_DI = MJXparams(
    motion_constraints=double_integrator_motion_constraints,
    physics_constants=PhysicsConstants(),
    batch_size=batch_size,
    bounds = Bounds(),
    dims=6,
    action_dims=3,
    dt = 0.2,
    seed = seed
)

sim_params_DA = MJXparams(
    motion_constraints=dubins_motion_constraints,
    physics_constants=PhysicsConstants(),
    batch_size=batch_size,
    bounds = Bounds(),
    dims=6,
    action_dims=3,
    dt = 0.1,
    seed = seed
)

sim_params_QC = MJXparams(
    motion_constraints=MotionConstraints(),
    physics_constants=PhysicsConstants(),
    batch_size=batch_size,
    bounds = bounds_qc,
    dims=12,
    action_dims=4,
    dt = 0.1,
    seed = seed
)

# SST params
sst_params_DI = SSTparams(
    batch_size=batch_size,
    δBN=0.06 * 1,
    δs=0.045 * 1,
    decay=0.8,
    start=start_simple,
    goal=goal_simple,
    goal_radius=0.05,
    geo_cost_to_go_weight=0.2,
    do_cost_to_go=True,
    do_maximal= True,
    do_set_cover= True,
    time_to_evolve= time_to_evolve,
    sparsity = 0,
)

sst_params_DA = SSTparams(
    batch_size=batch_size,
    δBN=0.06 * 1,
    δs=0.045 * 1,
    decay=0.8,
    start=start_simple,
    goal=goal_simple,
    goal_radius=0.05,
    geo_cost_to_go_weight=0.2,
    do_cost_to_go=True,
    do_maximal= True,
    do_set_cover= True,
    time_to_evolve= time_to_evolve,
    sparsity = 0,
)

sst_params_QC = SSTparams(
    batch_size=batch_size,
    δBN=0.06 * 1,
    δs=0.045 * 1,
    decay=0.8,
    start=start_qc,
    goal=goal_qc,
    goal_radius=5,
    geo_cost_to_go_weight=0.2,
    do_cost_to_go=True,
    do_maximal= True,
    do_set_cover= True,
    time_to_evolve= time_to_evolve,
    sparsity = 0,
)

frb_motion_constraints = MotionConstraints(
    max_vel=1.0,       # max joint angle for sampling
    min_vel=-1.0,      # min joint angle
    max_accel=0.5,     # unused for now
    min_accel=-0.5,
    max_torque=5.0,    # used for action sampling
    min_torque=-5.0,
    max_yaw_rate=0.0,  # unused
    min_yaw_rate=0.0,
    max_pitch_rate=0.0,
    min_pitch_rate=0.0,
    max_yaw=0.0,
    min_yaw=0.0,
    max_pitch=0.0,
    min_pitch=0.0,
    max_thrust=0.0,
    min_thrust=0.0,
    max_roll=0.0,
    min_roll=0.0,
    max_angle_vel=0.0,
    min_angle_vel=0.0
)

eeb_motion_constraints = MotionConstraints(
    min_vel = -1.0,
    max_vel = 1.0,
    min_torque = -3.0,
    max_torque = 3.0
)

# Workspace bounds for the block
frb_bounds = Bounds(
    min_x=-0.3, max_x=.8,
    min_y=-0.3, max_y=.8,
    min_z=0.0, max_z=0.3
)

callables_FRB = Callables(
    prop_fn=None,               # filled at runtime
    valid_fn=lambda *args: True, # unused
    sample_fn=helper.sample_FRB,
    dist_fn=lambda a, b: 0.0,    # dummy
    sampact_fn=helper.sample_actions_FRB,
)

sim_params_FRB = MJXparams(
    motion_constraints=frb_motion_constraints,  # torques bounded later
    physics_constants=PhysicsConstants(),
    batch_size=batch_size,
    bounds=frb_bounds,        # unused for now
    dims=27,
    action_dims=7,
    dt=0.02,                # MJX timestep
    seed=seed,
)

sst_params_FRB = SSTparams(
    batch_size=batch_size,
    δBN=0.2,
    δs=0.15,
    decay=0.8,
    start=Position(x=0.0, y=0.0, z=0.05),
    goal=Position(x=0.3, y=0.0, z=0.05),
    goal_radius=0.05,
    geo_cost_to_go_weight=0.0,
    do_cost_to_go=False,
    do_maximal=True,
    do_set_cover=True,
    time_to_evolve=5,
    sparsity=0,
)

sim_params_EEB = MJXparams(
    motion_constraints=eeb_motion_constraints,  # torques bounded later
    physics_constants=PhysicsConstants(),
    batch_size=batch_size,
    bounds=frb_bounds,        # unused for now
    dims=10,
    action_dims=2,
    dt=0.02,                # MJX timestep
    seed=seed,
)

sst_params_EEB = SSTparams(
    batch_size=batch_size,
    δBN=0.2,
    δs=0.15,
    decay=0.8,
    start=Position(x=0.1, y=0.0, z=0.0), # z is theta
    goal=Position(x=0.3, y=0.0, z=0.0),
    goal_radius=0.05,
    geo_cost_to_go_weight=0.0,
    do_cost_to_go=False,
    do_maximal=True,
    do_set_cover=True,
    time_to_evolve=5,
    sparsity=0,
)