import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from flax import struct
from params import Position
from vine.load_env import load_box_config
from matplotlib.animation import FuncAnimation, PillowWriter
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

    # Only use first 630 frames and every 3rd frame
    data = data[:627:3]

    print("Frames used:", len(data))

    fig, ax = plt.subplots(figsize=(8, 8))

    rects = []
    cmap = LinearSegmentedColormap.from_list("obs_gray", ["#404040", "#757575"])

    def draw_env():
        rects.clear()

        for obs in obstacles:
            x1, y1, x2, y2 = obs
            width = x2 - x1
            height = y2 - y1

            rects.append((x1, y1, x2, y2))

            # gradient fill
            ax.imshow(
                np.array([[0, 0], [1, 1]]),
                cmap=cmap,
                interpolation="bicubic",
                extent=(x1, x2, y1, y2),
                aspect="auto",
                alpha=0.8,
                zorder=1
            )

            # crisp outline
            rect_outline = patches.Rectangle(
                (x1, y1),
                width,
                height,
                linewidth=1.5,
                edgecolor="#2b2b2b",
                facecolor="none",
                zorder=1.1
            )
            ax.add_patch(rect_outline)

        # start / goal
        ax.scatter(
            sst_params.start.x,
            sst_params.start.y,
            color="green",
            s=100,
            marker="*",
            zorder=5,
        )

        ax.add_patch(
            patches.Circle(
                (sst_params.goal.x, sst_params.goal.y),
                sst_params.goal_radius,
                color="red",
                fill=False,
                linestyle="--",
                zorder=5,
            )
        )

        ax.set_xlim(sst_params.min_x, sst_params.max_x)
        ax.set_ylim(sst_params.min_y, sst_params.max_y)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    def update(frame):

        ax.clear()
        draw_env()

        print(f"Visualizing frame {frame}/{len(data)}")

        state = data[frame]

        angles = state[3:33]
        tip_len = state[33]
        n_bodies = int(state[34])

        curr_x = sst_params.start.x
        curr_y = sst_params.start.y
        curr_h = sst_params.start.z

        path = [(curr_x, curr_y)]

        segments = []

        for i in range(sim_params.max_bodies):

            if i < n_bodies:
                length = sim_params.body_length
            elif i == n_bodies:
                length = tip_len
            else:
                break

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

            if hit:
                break

            path.append((next_x, next_y))
            segments.append((curr_x, curr_y, next_x, next_y))

            curr_x, curr_y = next_x, next_y

        xs, ys = zip(*path)

        ax.plot(xs, ys, color="royalblue", linewidth=2, zorder=3)

        ax.scatter(xs[-1], ys[-1], color="blue", s=20, zorder=4)

        # draw midpoint radius circles
        for x1, y1, x2, y2 in segments:

            mx = 0.5 * (x1 + x2)
            my = 0.5 * (y1 + y2)

            circ = patches.Circle(
                (mx, my),
                sim_params.radius,
                color="royalblue",
                alpha=0.15,
                linewidth=0,
                zorder=2,
            )

            ax.add_patch(circ)

        ax.set_title(f"Frame: {frame}")

    ani = FuncAnimation(
        fig,
        update,
        frames=len(data),
        interval=50,
    )

    print("Saving animation as GIF...")

    writer = PillowWriter(fps=20)

    ani.save("videos/vine_growth.gif", writer=writer)

    print("Successfully saved to videos/vine_growth.gif")


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