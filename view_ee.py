import plotly.graph_objects as go
import numpy as np

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

def visualize_trajectory_no_obs(trajectory, output_name="./trajectory.html"):
    """
    trajectory: (N, 10) array
    State order: [ee_x, ee_y, ee_dx, ee_dy, block_x, block_y, block_theta, block_dx, block_dy, block_dtheta]
    """
    fig = go.Figure()
    
    # 1. Add Floor Plane (Bounding box: -0.3 to 0.7)
    fig.add_trace(go.Mesh3d(
        x=[-0.3, 0.7, 0.7, -0.3], 
        y=[-0.3, -0.3, 0.7, 0.7], 
        z=[0, 0, 0, 0],
        color='rgb(230, 230, 230)',
        opacity=0.5,
        showlegend=False
    ))

    # 2. EE (Sphere) Trajectory - Green Line
    fig.add_trace(go.Scatter3d(
        x=trajectory[:, 0], y=trajectory[:, 1], z=np.full_like(trajectory[:, 0], 0.05),
        mode='lines', line=dict(color='limegreen', width=5), name='EE Trajectory'
    ))

    # 3. Block (Cube) Snapshots - Every 5 frames
    block_size = 0.05 
    for i in range(0, len(trajectory), 5):
        bx, by, btheta = trajectory[i, 4], trajectory[i, 5], trajectory[i, 6]
        
        # Rotation Matrix
        c, s = np.cos(btheta), np.sin(btheta)
        R = np.array([[c, -s], [s, c]])
        
        # Base vertices
        v = np.array([[-block_size, -block_size], [block_size, -block_size], 
                      [block_size, block_size], [-block_size, block_size]])
        
        # Rotate and translate
        v_rot = (R @ v.T).T + np.array([bx, by])
        
        # Plot cube snapshot
        fig.add_trace(go.Mesh3d(
            x=np.concatenate([v_rot[:, 0], v_rot[:, 0]]),
            y=np.concatenate([v_rot[:, 1], v_rot[:, 1]]),
            z=[0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1],
            color='red', opacity=0.15, showlegend=(i==0), name='Block Path'
        ))

    # 4. Goal Area (Marker at 0.3, 0)
    gx, gy, gz, gi, gj, gk = get_sphere_mesh(0.3, 0.0, 0.001, radius=0.05)
    fig.add_trace(go.Mesh3d(x=gx, y=gy, z=gz, i=gi, j=gj, k=gk, color='green', opacity=0.5, name='Goal'))

    # 5. Scene Configuration
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-0.3, 0.7]),
            yaxis=dict(range=[-0.3, 0.7]),
            zaxis=dict(range=[0, 0.5]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    fig.write_html(output_name)
    print(f"Saved visualization to {output_name}")

trajectory = np.load("mjx_full_trajectory.npy")
visualize_trajectory_no_obs(trajectory)