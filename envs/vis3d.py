import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_cube(ax, dims, color='red', alpha=0.5):
    """dims: [x1, y1, z1, x2, y2, z2]"""
    x1, y1, z1, x2, y2, z2 = dims
    
    # Define the 8 vertices of the box
    v = np.array([[x1, y1, z1], [x2, y1, z1], [x2, y2, z1], [x1, y2, z1],
                  [x1, y1, z2], [x2, y1, z2], [x2, y2, z2], [x1, y2, z2]])
    
    # Define the 6 faces (each face is a list of 4 indices)
    faces = [
        [v[0], v[1], v[2], v[3]], # Bottom
        [v[4], v[5], v[6], v[7]], # Top
        [v[0], v[1], v[5], v[4]], # Side 1
        [v[2], v[3], v[7], v[6]], # Side 2
        [v[0], v[3], v[7], v[4]], # Side 3
        [v[1], v[2], v[6], v[5]]  # Side 4
    ]
    
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='black', alpha=alpha))

def visualize_obstacles(csv_path):
    # Load data, skipping the first header row
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    
    # Handle single-row CSVs
    if data.ndim == 1:
        data = data.reshape(1, -1)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for box in data:
        plot_cube(ax, box)

    # Auto-scale axes based on the obstacle bounds
    all_coords = data.reshape(-1, 3)
    max_range = np.array([all_coords[:,0].max(), all_coords[:,1].max(), all_coords[:,2].max()]).max()
    
    ax.set_xlim(0, max_range)
    ax.set_ylim(0, max_range)
    ax.set_zlim(0, max_range)
    title = csv_path.split('/')[-1].split('.')[0]  # Extract filename without extension for title
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Env: ' + title)
    ax.view_init(elev=30, azim=60)
    
    plt.savefig('envs/vis/' + title + '.png')

if __name__ == "__main__":
    # Ensure 'envs/tree.csv' or your specific filename is correct
    visualize_obstacles('envs/house.csv')