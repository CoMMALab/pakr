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


def visualize_single_bucket_animation(env_path, bucket_trajectories, output_name):

    all_trees = np.loadtxt(env_path, delimiter=',', skiprows=1)
    if all_trees.ndim == 1:
        all_trees = all_trees.reshape(1, -1)

    fig = go.Figure()

    # -------------------------------------------------
    # Precompute obstacle meshes once
    # -------------------------------------------------

    obstacle_data = []

    for box in all_trees:

        x,y,z,i,j,k = create_box_mesh(*box)

        obstacle_data.append(dict(
            box=box,
            mesh=go.Mesh3d(
                x=x,y=y,z=z,i=i,j=j,k=k,
                color='lightgrey',
                opacity=0.3,
                flatshading=True,
                lighting=dict(
                    ambient=0.05,
                    diffuse=1.0,
                    roughness=1.0,
                    specular=0.0,
                    fresnel=0.0
                ),
                lightposition=dict(x=5,y=5,z=10),
                showlegend=False
            ),
            edges=(x,y,z)
        ))

    # -------------------------------------------------
    # Start + Goal meshes
    # -------------------------------------------------

    start_trace = go.Scatter3d(
        x=[0.1], y=[0.08], z=[0.05],
        mode='markers',
        marker=dict(size=6, color='blue'),
        name='Start'
    )

    gx,gy,gz,gi,gj,gk = get_sphere_mesh(0.8,0.95,0.9)

    goal_trace = go.Mesh3d(
        x=gx,y=gy,z=gz,i=gi,j=gj,k=gk,
        color='limegreen',
        opacity=0.6,
        name='Goal'
    )

    # -------------------------------------------------
    # Animation frames
    # -------------------------------------------------

    max_points = max(len(t) for t in bucket_trajectories)
    step = 5
    frames = []
    prev_cube_size = 0
    for i in range(0, 50, step):
        print("frame step:", i, "/", max_points)

        frame_traces = []

        # -------------------------------------------------
        # Maintain historical maximum expansion
        # -------------------------------------------------

        curr_points = []

        for traj in bucket_trajectories:

            curr = traj[:min(i+step, len(traj))]

            frame_traces.append(go.Scatter3d(
                x=curr[:,0], y=curr[:,1], z=curr[:,2],
                mode='lines',
                line=dict(color='rgba(50,205,50,1.0)', width=8)
            ))

            curr_points.append(curr[:, :3])

        curr_points = np.vstack(curr_points)

        # historical cube expansion
        if i == 0:
            cube_size = np.max(curr_points)
        else:
            cube_size = max(prev_cube_size, np.max(curr_points))

        cube_size += 0.15
        prev_cube_size = cube_size
        # draw bounding cube
        bx, by, bz, bi, bj, bk = create_box_mesh(
            0, 0, 0,
            cube_size, cube_size, cube_size
        )

        frame_traces.append(go.Mesh3d(
            x=bx, y=by, z=bz,
            i=bi, j=bj, k=bk,
            color='blue',
            opacity=0.05,
            flatshading=True,
            showlegend=False
        ))

        add_box_edges(fig, bx, by, bz)
        for item in obstacle_data:

            box = item["box"]
            tree_min = box[:3]

            visible = np.all(tree_min <= cube_size)

            if visible:

                frame_traces.append(item["mesh"])

                x,y,z = item["edges"]
                add_box_edges(fig, x, y, z)

                frame_traces.append(start_trace)
                frame_traces.append(goal_trace)

                frames.append(go.Frame(data=frame_traces, name=f"step_{i}"))

            fig.frames = frames

    # -------------------------------------------------
    # Initial traces
    # -------------------------------------------------

    for _ in bucket_trajectories:
        fig.add_trace(go.Scatter3d(x=[],y=[],z=[],mode='lines'))

    fig.add_trace(start_trace)
    fig.add_trace(goal_trace)

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None,{
                    "frame":{"duration":70,"redraw":True},
                    "fromcurrent":True,
                    "transition":{"duration":0}
                }]
            )]
        )],
        scene=dict(aspectmode='data')
    )

    fig.write_html(output_name)


def run_all_buckets(npz_path, env_path):

    data_container = np.load(npz_path)

    sorted_keys = sorted(data_container.files, key=lambda x: int(x.split('_')[1]))

    all_trajectories = [data_container[key] for key in sorted_keys]

    for b in range(1):

        bucket_trajs = all_trajectories[b*10:(b+1)*10]
        print(f"Visualizing Bucket {b+1} with {len(bucket_trajs)} trajectories...")

        visualize_single_bucket_animation(
            env_path,
            bucket_trajs,
            f"videos/bucket_{b+1}_growth.html"
        )


run_all_buckets('videos/trajectories_data.npz', 'envs/tree.csv')