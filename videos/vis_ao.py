import numpy as np
import plotly.graph_objects as go

def add_box_edges(fig, x, y, z):
    """
    Adds explicit black edges to the figure for visual clarity.
    """

    edges = [
        (0,1),(1,2),(2,3),(3,0),      # bottom
        (4,5),(5,6),(6,7),(7,4),      # top
        (0,4),(1,5),(2,6),(3,7)       # vertical
    ]

    for e in edges:
        fig.add_trace(go.Scatter3d(
            x=[x[e[0]], x[e[1]]],
            y=[y[e[0]], y[e[1]]],
            z=[z[e[0]], z[e[1]]],
            mode='lines',
            line=dict(color='grey', width=3),
            opacity=0.5,
            showlegend=False
        ))


def get_sphere_mesh(x0, y0, z0, radius=0.05, resolution=20):
    """Generates vertices and triangles for a sphere at (x0, y0, z0)."""
    phi = np.linspace(0, np.pi, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi, theta = np.meshgrid(phi, theta)
    
    x = x0 + radius * np.sin(phi) * np.cos(theta)
    y = y0 + radius * np.sin(phi) * np.sin(theta)
    z = z0 + radius * np.cos(phi)
    
    # Flatten arrays for Mesh3d
    x, y, z = x.flatten(), y.flatten(), z.flatten()
    
    # Generate triangle indices
    triangles = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            v0 = i * resolution + j
            v1 = v0 + 1
            v2 = (i + 1) * resolution + j
            v3 = v2 + 1
            triangles.append([v0, v1, v2])
            triangles.append([v1, v3, v2])
            
    triangles = np.array(triangles)
    return x, y, z, triangles[:, 0], triangles[:, 1], triangles[:, 2]


# --- Helper functions (from your provided snippet) ---
def create_box_mesh(x1, y1, z1, x2, y2, z2):
    x = [x1, x1, x2, x2, x1, x1, x2, x2]
    y = [y1, y2, y2, y1, y1, y2, y2, y1]
    z = [z1, z1, z1, z1, z2, z2, z2, z2]
    i = [0, 0, 4, 4, 0, 1, 2, 3, 6, 6, 5, 4]
    j = [1, 2, 5, 6, 4, 5, 3, 7, 2, 7, 1, 0]
    k = [2, 3, 6, 7, 5, 6, 7, 6, 7, 4, 0, 5]
    return x, y, z, i, j, k


def visualize_animated_trajectories(env_path, trajectories, output_name="videos/ao_solutions.html"):
    fig = go.Figure()

    # 1. Load Environment Obstacles
    data = np.loadtxt(env_path, delimiter=',', skiprows=1)
    if data.ndim == 1: data = data.reshape(1, -1)
    
    # Draw obstacles (Static)
    for box in data:
        x, y, z, i, j, k = create_box_mesh(*box)
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k,
            color='lightgrey', opacity=0.3, flatshading=True,
            lighting=dict(ambient=0.05, diffuse=1.0, roughness=1.0, specular=0.0, fresnel=0.0),
            lightposition=dict(x=5, y=5, z=10), showlegend=False
        ))
        add_box_edges(fig, x, y, z)

    # 2. Add Start and Goal
    fig.add_trace(go.Scatter3d(
        x=[0.1], y=[0.08], z=[0.05],
        mode='markers', marker=dict(size=6, color='blue'), name='Start'
    ))
    
    gx, gy, gz, gi, gj, gk = get_sphere_mesh(0.8, 0.95, 0.9)
    fig.add_trace(go.Mesh3d(x=gx, y=gy, z=gz, i=gi, j=gj, k=gk, color='limegreen', opacity=0.6, name='Goal'))

    
    # 3. Create Frames
    # 3. Create Frames
    frames = []
    
    for b in range(5):  # Current active bucket
        frame_traces = []
        for traj_idx in range(50):
            target_bucket = traj_idx // 10
            
            # Logic: 1.0 if active, 0.1 if background
            is_active = (target_bucket == b)
            opacity = 1.0 if is_active else 0.1
            width = 8.0 if is_active else 2.0
            
            # Use a slightly softer green for background lines to reduce visual noise
            color = f'rgba(50, 205, 50, {opacity})'
            
            frame_traces.append(go.Scatter3d(
                x=trajectories[traj_idx][:, 0],
                y=trajectories[traj_idx][:, 1],
                z=trajectories[traj_idx][:, 2],
                mode='lines',
                line=dict(color=color, width=width),
                showlegend=False
            ))
        frames.append(go.Frame(data=frame_traces, name=f"Bucket_{b}"))

    fig.frames = frames

    # Initial setup in the base figure must match the frame structure (50 empty traces)
    for traj in trajectories:
        fig.add_trace(go.Scatter3d(
            x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
            mode='lines', line=dict(color='rgba(0,0,0,0)', width=0), showlegend=False
        ))
    
    # Initial display: Add the first batch (Bucket 0) to the base figure so it's not empty
    # ... (Add the first 10 trajectories to fig as traces) ...

    # 4. Final Layout with Play Button
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(
                label="Play", 
                method="animate", 
                args=[None, {
                    "frame": {"duration": 2000, "redraw": True}, 
                    "fromcurrent": True,
                    "transition": {"duration": 300}
                }]
            )]
        )]
    )
    
    fig.write_html(output_name)
    print(f"Animation saved: {output_name}")


# Assuming visualize_animated_trajectories is defined as per our previous discussion
def load_and_run_visualizer(npz_path, env_path):
    # 1. Load the compressed data
    data_container = np.load(npz_path)

    # 2. Extract trajectories in sorted order
    sorted_keys = sorted(data_container.files, key=lambda x: int(x.split('_')[1]))
    trajectories = [data_container[key] for key in sorted_keys]
    
    # 3. Call the visualizer
    visualize_animated_trajectories(env_path, trajectories)

load_and_run_visualizer('videos/trajectories_data.npz', 'envs/tree.csv',)