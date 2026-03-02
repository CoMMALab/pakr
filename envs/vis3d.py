import numpy as np
import plotly.graph_objects as go
import os


def create_box_mesh(x1, y1, z1, x2, y2, z2):
    """
    Returns vertices and triangle indices for a box defined
    by min corner (x1,y1,z1) and max corner (x2,y2,z2)
    """

    # 8 vertices of box
    x = [x1, x1, x2, x2, x1, x1, x2, x2]
    y = [y1, y2, y2, y1, y1, y2, y2, y1]
    z = [z1, z1, z1, z1, z2, z2, z2, z2]

    # 12 triangles (2 per face)
    i = [0, 0, 4, 4, 0, 1, 2, 3, 6, 6, 5, 4]
    j = [1, 2, 5, 6, 4, 5, 3, 7, 2, 7, 1, 0]
    k = [2, 3, 6, 7, 5, 6, 7, 6, 7, 4, 0, 5]

    return x, y, z, i, j, k


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
            line=dict(color='black', width=6),
            showlegend=False
        ))


def visualize_obstacles_plotly(csv_path):
    # -----------------------------
    # Load obstacle CSV
    # -----------------------------
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    fig = go.Figure()

    # -----------------------------
    # Add each obstacle box
    # -----------------------------
    for box in data:
        x1, y1, z1, x2, y2, z2 = box

        x, y, z, i, j, k = create_box_mesh(x1, y1, z1, x2, y2, z2)

        # Add solid box
        fig.add_trace(go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            color='lightgrey',
            opacity=0.9,
            flatshading=True,

            # Improved lighting for strong face contrast
            lighting=dict(
                ambient=0.05,     # very low ambient -> strong depth
                diffuse=1.0,
                roughness=1.0,    # matte surface
                specular=0.0,
                fresnel=0.0
            ),
            lightposition=dict(x=5, y=5, z=10)
        ))

        # Add real edges
        add_box_edges(fig, x, y, z)

    # -----------------------------
    # Scene Configuration
    # -----------------------------
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='X',
                showbackground=True,
                backgroundcolor="rgb(240,240,240)"
            ),
            yaxis=dict(
                title='Y',
                showbackground=True,
                backgroundcolor="rgb(240,240,240)"
            ),
            zaxis=dict(
                title='Z',
                showbackground=True,
                backgroundcolor="rgb(240,240,240)"
            ),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.3)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='white'
    )

    # -----------------------------
    # Save HTML
    # -----------------------------
    title = os.path.basename(csv_path).split('.')[0]
    output_path = f'envs/vis/{title}.html'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig.write_html(output_path)
    print(f"Saved render to: {output_path}")


if __name__ == "__main__":
    visualize_obstacles_plotly('envs/house.csv')