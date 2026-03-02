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
        
        # Color and Opacity
        color = 'grey' 
        opacity = 0.7

        # Vertices and Triangles
        x = [x1, x1, x2, x2, x1, x1, x2, x2]
        y = [y1, y2, y2, y1, y1, y2, y2, y1]
        z = [z1, z1, z1, z1, z2, z2, z2, z2]
        
        i_idx = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
        j_idx = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
        k_idx = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]

        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z, i=i_idx, j=j_idx, k=k_idx,
            opacity=opacity,
            color=color,
            flatshading=True,
            # --- DARK OUTLINES ---
            contour=dict(show=True, color='darkgrey', width=4),
            # --- IMPROVED LIGHTING ---
            lighting=dict(
                ambient=0.4, 
                diffuse=0.8, 
                roughness=0.2, 
                specular=1, 
                fresnel=1
            ),
            lightposition=dict(x=10, y=10, z=10)
        ))

    title = os.path.basename(csv_path).split('.')[0]
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=True, backgroundcolor="rgb(230, 230,230)"),
            yaxis=dict(showbackground=True, backgroundcolor="rgb(230, 230,230)"),
            zaxis=dict(showbackground=True, backgroundcolor="rgb(230, 230,230)"),
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='white'
    )

    output_path = f'envs/vis/{title}.html'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig.write_html(output_path)
    print(f"Successfully saved outlined render to: {output_path}")

if __name__ == "__main__":
    visualize_obstacles_plotly_ssh('envs/house.csv')