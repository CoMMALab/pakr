import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from flax import struct
from params import Position
from vine.load_env import load_box_config

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from flax import struct
from params import Position
from vine.load_env import load_box_config

def visualize_trajectory(traj_path, obstacles, sst_params, sim_params):
    # 1. Load data
    # Shape: (Steps, Tips(3) + CSpace(max_bodies+1) + Bodies(1))
    data = np.load(traj_path)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 2. Setup static environment
    def draw_env():
        for obs in obstacles:
            x1, y1, x2, y2 = obs
            ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, color='#404040', zorder=1))
        ax.scatter(sst_params.start.x, sst_params.start.y, color='green', s=100, marker='*', zorder=5)
        ax.add_patch(patches.Circle((sst_params.goal.x, sst_params.goal.y), sst_params.goal_radius, 
                                    color='red', fill=False, linestyle='--', zorder=5))
        ax.set_xlim(sst_params.min_x, sst_params.max_x)
        ax.set_ylim(sst_params.min_y, sst_params.max_y)
        ax.set_aspect('equal')

    line, = ax.plot([], [], color='royalblue', linewidth=2, zorder=2)
    
    def update(frame):
        ax.clear()
        draw_env()
        
        state = data[frame]
        angles = state[3:33] # CSpace
        tip_len = state[33]
        n_bodies = int(state[34])
        
        # Kinematics to reconstruct segments
        curr_x, curr_y = sst_params.start.x, sst_params.start.y
        curr_h = sst_params.start.z
        
        path = [(curr_x, curr_y)]
        for i in range(sim_params.max_bodies):
            length = sim_params.body_length if i < n_bodies else (tip_len if i == n_bodies else 0.0)
            if i > n_bodies: break
            
            curr_h += angles[i]
            curr_x += length * np.cos(curr_h)
            curr_y += length * np.sin(curr_h)
            path.append((curr_x, curr_y))
        
        xs, ys = zip(*path)
        ax.plot(xs, ys, color='royalblue', linewidth=2, zorder=3)
        ax.scatter(xs[-1], ys[-1], color='blue', s=20, zorder=4)
        ax.set_title(f"Step: {frame}")

    # 3. Create Animation
    ani = FuncAnimation(fig, update, frames=len(data), interval=50)
    
    # Save as MP4 (Requires ffmpeg installed on system)

    writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    ani.save("videos/vine_growth.mp4", writer=writer)
    print("Saved to videos/vine_growth.mp4")



@struct.dataclass
class VineParams:
    batch_size: int
    max_bodies: int
    dims: int
    action_dims: int
    body_length: float
    radius: float
    dt: float
    grow_rate: float
    grow_force: float
    stiffness: float
    damping: float
    substeps: int
    alpha: float

@struct.dataclass
class SSTparams:
    batch_size: int
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    start: Position
    goal: Position
    goal_radius: float
    time_to_evolve: int = 70

if __name__ == "__main__":
    cfg = load_box_config('vine/envs/env_live.txt')

    batch_size = 128
    A = 2
    max_bodies = 30
    sim_params = VineParams(
        batch_size=batch_size,
        max_bodies=max_bodies,
        dims=max_bodies + 5, # tip + cspace + tip + n_bodies
        action_dims=max_bodies, # bending control for each body
        body_length=68.0, # 25.0 mm
        radius=50, # 16.0,
        dt=1.0,
        grow_rate=20.0,
        grow_force=15.0,
        stiffness=50.0,
        damping=50.0,
        # Curiously, decreasing substeps helps prevent penetration bugs. But it doesn't fix the root problem
        substeps=15, # FIXME THIS NUMBER CAN BE MUCH SMALLER IF WE DO LANGRANGE PROPERRLY
        alpha=1e-2,
    )
    
    obstacles=cfg['obstacles']

    print(obstacles.shape)
    # SST params
    sst_params = SSTparams(
        batch_size=1024,
        min_x=0.0,
        max_x=float(cfg['bound_x']),
        min_y=0.0,
        max_y=float(cfg['bound_y']),
        # USE TUPLES HERE. Lists [x, y, z] are unhashable.
        start=Position(float(cfg['start'][0]), float(cfg['start'][1]), float(cfg['start'][2])),
        goal=Position(float(cfg['goal'][0]), float(cfg['goal'][1]), float(cfg['goal'][2])),
        goal_radius=float(cfg['goal_radius']),
    )
# Usage
    visualize_trajectory('vine/vine_traj.npy', obstacles, sst_params, sim_params)