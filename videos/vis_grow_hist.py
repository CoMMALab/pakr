import numpy as np
import plotly.graph_objects as go


def add_box_edges(fig, x, y, z):
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
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
    phi = np.linspace(0, np.pi, resolution)
    theta = np.linspace(0, 2*np.pi, resolution)
    phi, theta = np.meshgrid(phi, theta)

    x = x0 + radius*np.sin(phi)*np.cos(theta)
    y = y0 + radius*np.sin(phi)*np.sin(theta)
    z = z0 + radius*np.cos(phi)

    x, y, z = x.flatten(), y.flatten(), z.flatten()

    triangles = []
    for i in range(resolution-1):
        for j in range(resolution-1):
            v0 = i*resolution + j
            v1 = v0 + 1
            v2 = (i+1)*resolution + j
            v3 = v2 + 1
            triangles.append([v0, v1, v2])
            triangles.append([v1, v3, v2])

    triangles = np.array(triangles)

    return x, y, z, triangles[:,0], triangles[:,1], triangles[:,2]


def create_box_mesh(x1, y1, z1, x2, y2, z2):
    x = [x1, x1, x2, x2, x1, x1, x2, x2]
    y = [y1, y2, y2, y1, y1, y2, y2, y1]
    z = [z1, z1, z1, z1, z2, z2, z2, z2]

    i = [0,0,4,4,0,1,2,3,6,6,5,4]
    j = [1,2,5,6,4,5,3,7,2,7,1,0]
    k = [2,3,6,7,5,6,7,6,7,4,0,5]

    return x,y,z,i,j,k


def visualize_single_bucket_animation(env_path, current_bucket, history_buckets, output_name):
    all_trees = np.loadtxt(env_path, delimiter=',', skiprows=1)
    if all_trees.ndim == 1:
        all_trees = all_trees.reshape(1, -1)

    fig = go.Figure()

    # 3. Add History Buckets (Static, non-animated)
    # Tiers: [.6, .4, .2, .1]. Most recent history gets highest opacity.
    opacity_tiers = [0.6, 0.4, 0.2, 0.1]
    
    for idx, bucket in enumerate(reversed(history_buckets)):
        alpha = opacity_tiers[idx] if idx < len(opacity_tiers) else 0.05
        hx, hy, hz = [], [], []
        for traj in bucket:
            # Use None to tell Plotly to "lift the pen" between lines
            hx.extend(traj[:, 0].tolist() + [None])
            hy.extend(traj[:, 1].tolist() + [None])
            hz.extend(traj[:, 2].tolist() + [None])
                
        fig.add_trace(go.Scatter3d(
            x=hx, y=hy, z=hz,
            mode='lines',
            line=dict(color=f'rgba(50,205,50,{alpha})', width=8),
            showlegend=False
        ))


    # 1. Render all obstacles once (REMAINING COMMENTED OUT AS REQUESTED)
    for box in all_trees:
        x, y, z, i, j, k = create_box_mesh(*box)
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k,
            color='lightgrey',
            opacity=0.3,
            flatshading=True,
            lighting=dict(
                ambient=0.05, diffuse=1.0, roughness=1.0,
                specular=0.0, fresnel=0.0
            ),
            lightposition=dict(x=5, y=5, z=10),
            showlegend=False
        ))
        add_box_edges(fig, x, y, z)

    # 2. Add Start/Goal traces once
    fig.add_trace(go.Scatter3d(
        x=[0.1], y=[0.08], z=[0.05],
        mode='markers', marker=dict(size=6, color='blue'), name='Start'
    ))

    gx, gy, gz, gi, gj, gk = get_sphere_mesh(0.8, 0.95, 0.9)
    fig.add_trace(go.Mesh3d(
        x=gx, y=gy, z=gz, i=gi, j=gj, k=gk,
        color='limegreen', opacity=0.6, name='Goal'
    ))


    # 4. Add initial EMPTY traces for the CURRENT bucket animation
    base_trace_count = len(fig.data) 
    
    for _ in current_bucket:
        fig.add_trace(go.Scatter3d(
            x=[], y=[], z=[], 
            mode='lines',
            line=dict(color='rgba(50,205,50,1.0)', width=10), # Solid Green
            opacity=1.0,
            name="Current Growth"
        ))

    # 5. Create animation frames
    max_points = max(len(t) for t in current_bucket)
    step = 5
    frames = []
    
    traj_indices = list(range(base_trace_count, base_trace_count + len(current_bucket)))

    for i in range(0, max_points + step, step):
        frame_data = []
        for traj in current_bucket:
            curr = traj[:min(i, len(traj))]
            frame_data.append(go.Scatter3d(
                x=curr[:,0], y=curr[:,1], z=curr[:,2]
            ))
        
        frames.append(go.Frame(
            data=frame_data, 
            name=f"step_{i}",
            traces=traj_indices 
        ))

    fig.frames = frames

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            zaxis=dict(range=[0, 1]),
            aspectmode='cube'
        ),
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None, {
                    "frame": {"duration": 50, "redraw": True},
                    "fromcurrent": True,
                    "transition": {"duration": 0},
                    "repeat": False
                }]
            )]
        )]
    )

    fig.write_html(output_name)


def run_all_buckets(npz_path, env_path):
    data_container = np.load(npz_path)
    sorted_keys = sorted(data_container.files, key=lambda x: int(x.split('_')[1]))
    all_trajectories = [data_container[key] for key in sorted_keys]

    # Organize data into buckets of 10
    buckets_list = []
    trajs_per_bucket = 10
    num_buckets = len(all_trajectories) // trajs_per_bucket

    for b in range(num_buckets):
        buckets_list.append(all_trajectories[b*trajs_per_bucket : (b+1)*trajs_per_bucket])

    # Generate an HTML for each bucket
    for b in range(num_buckets):
        current_bucket = buckets_list[b]
        history_buckets = buckets_list[:b] # All buckets before index b

        print(f"Visualizing Bucket {b+1} with {len(history_buckets)} historical buckets...")

        visualize_single_bucket_animation(
            env_path,
            current_bucket,
            history_buckets,
            f"videos/bucket_{b+1}_growth.html"
        )


run_all_buckets('videos/trajectories_data.npz', 'envs/tree.csv')