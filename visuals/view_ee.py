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

def visualize_trajectory_1(trajectory, output_name="visuals/trajectory_final1.html"):
    """
    trajectory: (N, 10) array
    State: [ee_x, ee_y, block_x, block_y, block_theta, ee_dx, ee_dy, block_dx, block_dy, block_dtheta]
    """
    fig = go.Figure()
    
    # 1. Floor Plane
    fig.add_trace(go.Mesh3d(
        x=[-0.3, 0.7, 0.7, -0.3], y=[-0.3, -0.3, 0.7, 0.7], z=[0, 0, 0, 0],
        color='rgb(230, 230, 230)', opacity=0.3, showlegend=False
    ))

    # 2. EE Trajectory (Only up to t=65)
    traj_limit = 66 # Indices 0 to 65
    fig.add_trace(go.Scatter3d(
        x=trajectory[:traj_limit, 0], y=trajectory[:traj_limit, 1], z=np.full_like(trajectory[:traj_limit, 0], 0.05),
        mode='lines', line=dict(color='#66c2a5', width=4), name='EE Path'
    ))

    # 3. Periodic Snapshots (Every 10 timesteps)
    num_steps = len(trajectory)
    for i in range(0, 30, 3):
        opacity = 0.1 + 0.8 * ((i) / 30)
        opacity2 = 0.1 + 0.8 * ((i) / 30)
        
        ex, ey = trajectory[i, 0], trajectory[i, 1]
        sx, sy, sz, si, sj, sk = get_sphere_mesh(ex, ey, 0.05, radius=0.03)
        fig.add_trace(go.Mesh3d(
            x=sx, y=sy, z=sz, i=si, j=sj, k=sk,
            color='#66c2a5', opacity=opacity, showlegend=(i==0), name='EE Sphere'
        ))
    
        bx, by, btheta = trajectory[i, 2], trajectory[i, 3], trajectory[i, 4]
        c, s = np.cos(btheta), np.sin(btheta)
        
        x_pts = [-0.05, 0.05, 0.05, -0.05]
        y_pts = [-0.05, -0.05, 0.05, 0.05]
        xr = [x * c - y * s + bx for x, y in zip(x_pts, y_pts)]
        yr = [x * s + y * c + by for x, y in zip(x_pts, y_pts)]
        
        fig.add_trace(go.Mesh3d(
            x=xr + xr, y=yr + yr, z=[0]*4 + [0.1]*4,
            alphahull=0, color='#fc8d62', opacity=opacity2, showlegend=(i==50), name='Block'
        ))

    # 4. Goal Marker
    theta = np.linspace(0, 2*np.pi, 50)
    fig.add_trace(go.Scatter3d(
        x=0.3 + 0.05 * np.cos(theta), y=0.0 + 0.05 * np.sin(theta), z=[0.001]*50,
        mode='lines', line=dict(color='green', width=4), name='Goal'
    ))

    # 5. Scene Configuration
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-0.3, 0.7]),
            yaxis=dict(range=[-0.3, 0.7]),
            zaxis=dict(range=[0, 0.3]),
            aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.3)
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    
    fig.write_html(output_name)
    print(f"Visualization saved to {output_name}")


def visualize_trajectory_2(trajectory, output_name="visuals/trajectory_final2.html"):
    """
    trajectory: (N, 10) array
    State: [ee_x, ee_y, block_x, block_y, block_theta, ee_dx, ee_dy, block_dx, block_dy, block_dtheta]
    """
    fig = go.Figure()
    
    # 1. Floor Plane
    fig.add_trace(go.Mesh3d(
        x=[-0.3, 0.7, 0.7, -0.3], y=[-0.3, -0.3, 0.7, 0.7], z=[0, 0, 0, 0],
        color='rgb(230, 230, 230)', opacity=0.3, showlegend=False
    ))

    # 2. EE Trajectory (Only up to t=65)
    traj_limit = 66 # Indices 0 to 65
    fig.add_trace(go.Scatter3d(
        x=trajectory[:traj_limit, 0], y=trajectory[:traj_limit, 1], z=np.full_like(trajectory[:traj_limit, 0], 0.05),
        mode='lines', line=dict(color='#66c2a5', width=4), name='EE Path'
    ))

    # 3. Periodic Snapshots (Every 10 timesteps)
    num_steps = len(trajectory)
    for i in range(30, 50, 3):
        opacity = 0.1 + 0.8 * ((i-30) / 20)
        opacity2 = 0.1 + 0.8 * ((i-30) / 20)
        
        ex, ey = trajectory[i, 0], trajectory[i, 1]
        sx, sy, sz, si, sj, sk = get_sphere_mesh(ex, ey, 0.05, radius=0.03)
        fig.add_trace(go.Mesh3d(
            x=sx, y=sy, z=sz, i=si, j=sj, k=sk,
            color='#66c2a5', opacity=opacity, showlegend=(i==0), name='EE Sphere'
        ))
    
        bx, by, btheta = trajectory[i, 2], trajectory[i, 3], trajectory[i, 4]
        c, s = np.cos(btheta), np.sin(btheta)
        
        x_pts = [-0.05, 0.05, 0.05, -0.05]
        y_pts = [-0.05, -0.05, 0.05, 0.05]
        xr = [x * c - y * s + bx for x, y in zip(x_pts, y_pts)]
        yr = [x * s + y * c + by for x, y in zip(x_pts, y_pts)]
        
        fig.add_trace(go.Mesh3d(
            x=xr + xr, y=yr + yr, z=[0]*4 + [0.1]*4,
            alphahull=0, color='#fc8d62', opacity=opacity2, showlegend=(i==50), name='Block'
        ))

    # 4. Goal Marker
    theta = np.linspace(0, 2*np.pi, 50)
    fig.add_trace(go.Scatter3d(
        x=0.3 + 0.05 * np.cos(theta), y=0.0 + 0.05 * np.sin(theta), z=[0.001]*50,
        mode='lines', line=dict(color='green', width=4), name='Goal'
    ))

    # 5. Scene Configuration
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-0.3, 0.7]),
            yaxis=dict(range=[-0.3, 0.7]),
            zaxis=dict(range=[0, 0.3]),
            aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.3)
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    
    fig.write_html(output_name)
    print(f"Visualization saved to {output_name}")

def visualize_trajectory_3(trajectory, output_name="visuals/trajectory_final3.html"):
    """
    trajectory: (N, 10) array
    State: [ee_x, ee_y, block_x, block_y, block_theta, ee_dx, ee_dy, block_dx, block_dy, block_dtheta]
    """
    fig = go.Figure()
    
    # 1. Floor Plane
    fig.add_trace(go.Mesh3d(
        x=[-0.3, 0.7, 0.7, -0.3], y=[-0.3, -0.3, 0.7, 0.7], z=[0, 0, 0, 0],
        color='rgb(230, 230, 230)', opacity=0.3, showlegend=False
    ))

    # 2. EE Trajectory (Only up to t=65)
    traj_limit = 66 # Indices 0 to 65
    fig.add_trace(go.Scatter3d(
        x=trajectory[:traj_limit, 0], y=trajectory[:traj_limit, 1], z=np.full_like(trajectory[:traj_limit, 0], 0.05),
        mode='lines', line=dict(color='#66c2a5', width=4), name='EE Path'
    ))

    # 3. Periodic Snapshots (Every 10 timesteps)
    num_steps = len(trajectory)
    for i in range(50, 70, 3):
        opacity = 0.1 + 0.8 * ((i-50) / 20)
        opacity2 = 0.1 + 0.8 * ((i-50) / 20)
        
        ex, ey = trajectory[i, 0], trajectory[i, 1]
        sx, sy, sz, si, sj, sk = get_sphere_mesh(ex, ey, 0.05, radius=0.03)
        fig.add_trace(go.Mesh3d(
            x=sx, y=sy, z=sz, i=si, j=sj, k=sk,
            color='#66c2a5', opacity=opacity, showlegend=(i==0), name='EE Sphere'
        ))
    
        bx, by, btheta = trajectory[i, 2], trajectory[i, 3], trajectory[i, 4]
        c, s = np.cos(btheta), np.sin(btheta)
        
        x_pts = [-0.05, 0.05, 0.05, -0.05]
        y_pts = [-0.05, -0.05, 0.05, 0.05]
        xr = [x * c - y * s + bx for x, y in zip(x_pts, y_pts)]
        yr = [x * s + y * c + by for x, y in zip(x_pts, y_pts)]
        
        fig.add_trace(go.Mesh3d(
            x=xr + xr, y=yr + yr, z=[0]*4 + [0.1]*4,
            alphahull=0, color='#fc8d62', opacity=opacity2, showlegend=(i==50), name='Block'
        ))

    # 4. Goal Marker
    theta = np.linspace(0, 2*np.pi, 50)
    fig.add_trace(go.Scatter3d(
        x=0.3 + 0.05 * np.cos(theta), y=0.0 + 0.05 * np.sin(theta), z=[0.001]*50,
        mode='lines', line=dict(color='green', width=4), name='Goal'
    ))

    # 5. Scene Configuration
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-0.3, 0.7]),
            yaxis=dict(range=[-0.3, 0.7]),
            zaxis=dict(range=[0, 0.3]),
            aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.3)
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    
    fig.write_html(output_name)
    print(f"Visualization saved to {output_name}")

def visualize_trajectory_4(trajectory, output_name="visuals/trajectory_final4.html"):
    """
    trajectory: (N, 10) array
    State: [ee_x, ee_y, block_x, block_y, block_theta, ee_dx, ee_dy, block_dx, block_dy, block_dtheta]
    """
    fig = go.Figure()
    
    # 1. Floor Plane
    fig.add_trace(go.Mesh3d(
        x=[-0.3, 0.7, 0.7, -0.3], y=[-0.3, -0.3, 0.7, 0.7], z=[0, 0, 0, 0],
        color='rgb(230, 230, 230)', opacity=0.3, showlegend=False
    ))

    # 2. EE Trajectory (Only up to t=65)
    traj_limit = 66 # Indices 0 to 65
    fig.add_trace(go.Scatter3d(
        x=trajectory[:traj_limit, 0], y=trajectory[:traj_limit, 1], z=np.full_like(trajectory[:traj_limit, 0], 0.05),
        mode='lines', line=dict(color='#66c2a5', width=4), name='EE Path'
    ))

    # 3. Periodic Snapshots (Every 10 timesteps)
    num_steps = len(trajectory)
    for i in range(70, 90, 3):
        opacity = 0.1 + 0.8 * ((i-70) / 20)
        opacity2 = 0.1 + 0.8 * ((i-70) / 20)
        
        # A. EE Sphere (Visible only if i <= 65)

        ex, ey = trajectory[i, 0], trajectory[i, 1]
        sx, sy, sz, si, sj, sk = get_sphere_mesh(ex, ey, 0.05, radius=0.03)
        fig.add_trace(go.Mesh3d(
            x=sx, y=sy, z=sz, i=si, j=sj, k=sk,
            color='#66c2a5', opacity=opacity, showlegend=(i==0), name='EE Sphere'
        ))
    
        bx, by, btheta = trajectory[i, 2], trajectory[i, 3], trajectory[i, 4]
        c, s = np.cos(btheta), np.sin(btheta)
        
        x_pts = [-0.05, 0.05, 0.05, -0.05]
        y_pts = [-0.05, -0.05, 0.05, 0.05]
        xr = [x * c - y * s + bx for x, y in zip(x_pts, y_pts)]
        yr = [x * s + y * c + by for x, y in zip(x_pts, y_pts)]
        
        fig.add_trace(go.Mesh3d(
            x=xr + xr, y=yr + yr, z=[0]*4 + [0.1]*4,
            alphahull=0, color='#fc8d62', opacity=opacity2, showlegend=(i==50), name='Block'
        ))

    # 4. Goal Marker
    theta = np.linspace(0, 2*np.pi, 50)
    fig.add_trace(go.Scatter3d(
        x=0.3 + 0.05 * np.cos(theta), y=0.0 + 0.05 * np.sin(theta), z=[0.001]*50,
        mode='lines', line=dict(color='green', width=4), name='Goal'
    ))

    # 5. Scene Configuration
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-0.3, 0.7]),
            yaxis=dict(range=[-0.3, 0.7]),
            zaxis=dict(range=[0, 0.3]),
            aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.3)
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    
    fig.write_html(output_name)
    print(f"Visualization saved to {output_name}")

trajectory = np.load("visuals/mjx_full_trajectory.npy")
visualize_trajectory_1(trajectory)
visualize_trajectory_2(trajectory)
visualize_trajectory_3(trajectory)
visualize_trajectory_4(trajectory)