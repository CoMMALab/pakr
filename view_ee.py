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

import plotly.graph_objects as go
import numpy as np

def visualize_trajectory_v2(trajectory, output_name="./trajectory_v2.html"):
    """
    trajectory: (N, 10) array
    State: [ee_x, ee_y, block_x, block_y, block_theta, ee_dx, ee_dy, block_dx, block_dy, block_dtheta]
    """
    fig = go.Figure()
    
    # 1. Floor Plane (-0.3 to 0.7)
    fig.add_trace(go.Mesh3d(
        x=[-0.3, 0.7, 0.7, -0.3], y=[-0.3, -0.3, 0.7, 0.7], z=[0, 0, 0, 0],
        color='rgb(230, 230, 230)', opacity=0.3, showlegend=False
    ))

    # 2. EE Trajectory (Indices 0, 1)
    fig.add_trace(go.Scatter3d(
        x=trajectory[:, 0], y=trajectory[:, 1], z=np.full_like(trajectory[:, 0], 0.05),
        mode='lines', line=dict(color='limegreen', width=5), name='EE Path'
    ))

    # 3. Block Snapshots (Indices 2, 3, 4) - Drawn as full 3D Cubes
    block_size = 0.05
    for i in range(0, len(trajectory), 5):
        bx, by, btheta = trajectory[i, 2], trajectory[i, 3], trajectory[i, 4]
        c, s = np.cos(btheta), np.sin(btheta)
        
        # Define 8 corners of the cube
        x_pts = [-block_size, block_size, block_size, -block_size]
        y_pts = [-block_size, -block_size, block_size, block_size]
        
        # Rotate footprint
        x_rot = [x * c - y * s + bx for x, y in zip(x_pts, y_pts)]
        y_rot = [x * s + y * c + by for x, y in zip(x_pts, y_pts)]
        
        # Add as a Mesh3d cube (bottom and top faces)
        fig.add_trace(go.Mesh3d(
            x=x_rot + x_rot, y=y_rot + y_rot, z=[0]*4 + [0.1]*4,
            alphahull=0, color='red', opacity=0.2, showlegend=(i==0), name='Block Snapshot'
        ))

    # 4. Goal Area (Flat 2D Circle)
    theta = np.linspace(0, 2*np.pi, 50)
    fig.add_trace(go.Scatter3d(
        x=0.3 + 0.05 * np.cos(theta), y=0.0 + 0.05 * np.sin(theta), z=np.zeros(50),
        mode='lines', line=dict(color='green', width=4), name='Goal'
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-0.3, 0.7]),
            yaxis=dict(range=[-0.3, 0.7]),
            zaxis=dict(range=[0, 0.5]),
            aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.5)
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    fig.write_html(output_name)

trajectory = np.load("mjx_full_trajectory.npy")
visualize_trajectory_v2(trajectory)