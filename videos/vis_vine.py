import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from flax import struct
from params import Position
from vine.load_env import load_box_config
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap


def intersect_segment_rect(x1, y1, x2, y2, rect):
    """
    Checks if segment (x1, y1)->(x2, y2) intersects rectangle.
    Returns intersection point or None.
    """
    rx1, ry1, rx2, ry2 = rect

    best_t = 1.1

    def check_vertical(x):
        nonlocal best_t
        if abs(x2 - x1) < 1e-6:
            return
        t = (x - x1) / (x2 - x1)
        if 0 <= t <= 1:
            y = y1 + t * (y2 - y1)
            if ry1 <= y <= ry2:
                best_t = min(best_t, t)

    def check_horizontal(y):
        nonlocal best_t
        if abs(y2 - y1) < 1e-6:
            return
        t = (y - y1) / (y2 - y1)
        if 0 <= t <= 1:
            x = x1 + t * (x2 - x1)
            if rx1 <= x <= rx2:
                best_t = min(best_t, t)

    check_vertical(rx1)
    check_vertical(rx2)
    check_horizontal(ry1)
    check_horizontal(ry2)

    if best_t <= 1.0:
        return x1 + best_t * (x2 - x1), y1 + best_t * (y2 - y1)

    return None


def visualize_trajectory(traj_path, obstacles, sst_params, sim_params):
    data = np.load(traj_path)
    data = data[:558:3]
    last_frame = data[-1]
    first_frame = data[0]
    pause_first = np.tile(first_frame, (10, 1))
    pause_frames = np.tile(last_frame, (20, 1))
    data = np.vstack((pause_first, data, pause_frames))

    # 1. Set Figure background (the area 'outside' the grid) to black
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')

    rects = []
    cmap = LinearSegmentedColormap.from_list("obs_gray", ["#404040", "#757575"])

    def draw_env():
        rects.clear()
        
        # 2. Set the internal grid area to white
        ax.set_facecolor('white')
        
        # Optional: Add a subtle border to the grid so it's defined against the black
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('#333333')
            spine.set_linewidth(2)

        for obs in obstacles:
            x1, y1, x2, y2 = obs
            width, height = x2 - x1, y2 - y1
            rects.append((x1, y1, x2, y2))

            ax.imshow(
                np.array([[0, 0], [1, 1]]),
                cmap=cmap,
                interpolation="bicubic",
                extent=(x1, x2, y1, y2),
                aspect="auto",
                alpha=0.9, # Slightly higher alpha to pop against white
                zorder=1
            )

            rect_outline = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=1.2, edgecolor="#2b2b2b",
                facecolor="none", zorder=1.1
            )
            ax.add_patch(rect_outline)

        # Start / Goal
        ax.scatter(sst_params.start.x, sst_params.start.y, 
                   color="green", s=100, marker="*", zorder=5)

        ax.add_patch(
            patches.Circle((sst_params.goal.x, sst_params.goal.y), sst_params.goal_radius,
                           color="red", fill=False, linestyle="--", zorder=5)
        )

        ax.set_xlim(sst_params.min_x, sst_params.max_x)
        ax.set_ylim(sst_params.min_y, sst_params.max_y)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    def update(frame):
        print(f"Rendering frame {frame}/{len(data)}")
        ax.clear()
        draw_env()

        state = data[frame]
        angles = state[3:33]
        tip_len = state[33]
        n_bodies = int(state[34])

        curr_x, curr_y = sst_params.start.x, sst_params.start.y
        curr_h = sst_params.start.z
        path, segments = [(curr_x, curr_y)], []

        for i in range(sim_params.max_bodies):
            length = sim_params.body_length if i < n_bodies else tip_len if i == n_bodies else None
            if length is None: break

            curr_h += angles[i]
            next_x = curr_x + length * np.cos(curr_h)
            next_y = curr_y + length * np.sin(curr_h)

            hit = False
            for rect in rects:
                ipt = intersect_segment_rect(curr_x, curr_y, next_x, next_y, rect)
                if ipt is not None:
                    path.append(ipt)
                    segments.append((curr_x, curr_y, ipt[0], ipt[1]))
                    hit = True
                    break

            if hit: break
            path.append((next_x, next_y))
            segments.append((curr_x, curr_y, next_x, next_y))
            curr_x, curr_y = next_x, next_y

        xs, ys = zip(*path)
        # Royal blue stands out well on white
        ax.plot(xs, ys, color="royalblue", linewidth=2.5, zorder=3)
        ax.scatter(xs[-1], ys[-1], color="blue", s=25, zorder=4)

        for x1, y1, x2, y2 in segments:
            mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            circ = patches.Circle((mx, my), sim_params.radius, 
                                  color="royalblue", alpha=0.1, linewidth=0, zorder=2)
            ax.add_patch(circ)

        # 3. Set text to white (appears on the black figure background)
        ax.set_title(f"Frame: {frame}", color="white", fontsize=14, pad=15)

    ani = FuncAnimation(fig, update, frames=len(data), interval=25)
    
    # Set global background color for the save process
    plt.rcParams['savefig.facecolor'] = 'black'

    print("Saving animation as MP4...")
    
    # bitrate=1000 to 2000 is usually plenty for 800x800 resolution
    # fps=20 keeps the motion smooth
    writer = FFMpegWriter(fps=50, bitrate=1500)
    
    # We use dpi=100 for 800x800, or dpi=50 for 400x400
    ani.save("videos/vine_growth.mp4", writer=writer, dpi=100)

    print("Successfully saved to videos/vine_growth.mp4")


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

    cfg = load_box_config("vine/envs/env_live.txt")

    batch_size = 128
    max_bodies = 30

    sim_params = VineParams(
        batch_size=batch_size,
        max_bodies=max_bodies,
        dims=max_bodies + 5,
        action_dims=max_bodies,
        body_length=68.0,
        radius=50,
        dt=1.0,
        grow_rate=20.0,
        grow_force=15.0,
        stiffness=50.0,
        damping=50.0,
        substeps=15,
        alpha=1e-2,
    )

    obstacles = cfg["obstacles"]

    sst_params = SSTparams(
        batch_size=1024,
        min_x=0.0,
        max_x=float(cfg["bound_x"]),
        min_y=0.0,
        max_y=float(cfg["bound_y"]),
        start=Position(
            float(cfg["start"][0]),
            float(cfg["start"][1]),
            float(cfg["start"][2]),
        ),
        goal=Position(
            float(cfg["goal"][0]),
            float(cfg["goal"][1]),
            float(cfg["goal"][2]),
        ),
        goal_radius=float(cfg["goal_radius"]),
    )

    visualize_trajectory(
        "vine/vine_traj.npy",
        obstacles,
        sst_params,
        sim_params,
    )