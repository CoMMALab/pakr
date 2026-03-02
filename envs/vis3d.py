import numpy as np
import plotly.graph_objects as go
import os

def visualize_obstacles_plotly_ssh(csv_path):
    # Load data
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    fig = go.Figure()

    for i, box in enumerate(data):
        x1, y1, z1, x2, y2, z2 = box
        
        # Logic to differentiate Floors/Walls from Furniture
        volume = (x2-x1) * (y2-y1) * (z2-z1)
        is_thin = (z2 - z1) < 0.05 or (x2 - x1) < 0.05 or (y2 - y1) < 0.05
        
        # Visual settings: Floors/Walls are light grey/transparent, Furniture is Red
        color = 'lightgrey' if (is_thin or volume > 0.5) else 'red'
        opacity = 0.2 if (is_thin or volume > 0.5) else 0.9

        # Vertices and Triangles for a 3D Mesh Box
        x = [x1, x1, x2, x2, x1, x1, x2, x2]
        y = [y1, y2, y2, y1, y1, y2, y2, y1]
        z = [z1, z1, z1, z1, z2, z2, z2, z2]
        
        # Standard triangulation for a cube
        i_idx = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
        j_idx = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
        k_idx = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]

        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z, i=i_idx, j=j_idx, k=k_idx,
            opacity=opacity,
            color=color,
            flatshading=True,
            lighting=dict(ambient=0.6, diffuse=1, roughness=0.1, specular=1, fresnel=2),
            lightposition=dict(x=10, y=10, z=10)
        ))

    title = os.path.basename(csv_path).split('.')[0]
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data',
            # Set a better default viewing angle for the save
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='white'
    )

    # Save to the specific directory your previous script used
    output_path = f'envs/vis/{title}.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Scale=2 makes the PNG high-res
    fig.write_html(output_path)
    print(f"Successfully saved improved render to: {output_path}")

if __name__ == "__main__":
    visualize_obstacles_plotly_ssh('envs/house.csv')