import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def generate_and_visualize_env(num_obstacles=40, size=0.05, filename='envs/tree2d.csv'):
    # Define a grid for alignment
    grid = np.arange(0.1, 0.9, size) 
    
    obs_list = []
    seen_coords = set()
    
    while len(obs_list) < num_obstacles:
        # Pick grid-aligned corner
        x1 = np.random.choice(grid)
        y1 = np.random.choice(grid)
        
        # Round to prevent floating point noise in the 'seen' set
        x1, y1 = round(x1, 4), round(y1, 4)
        
        if (x1, y1) not in seen_coords:
            x2, y2 = round(x1 + size, 4), round(y1 + size, 4)
            obs_list.append([x1, y1, x2, y2])
            seen_coords.add((x1, y1))

    # Convert to numpy and Save to CSV
    obstacles = np.array(obs_list)
    df = pd.DataFrame(obstacles)
    
    # header=False removes the column labels
    # float_format='%.3f' removes rounding noise like 0.1000000000001
    df.to_csv(filename, index=False, float_format='%.2f')
    print(f"Environment saved to {filename}")

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(6, 6))
    
    for obs in obstacles:
        rect = patches.Rectangle((obs[0], obs[1]), size, size, 
                                 linewidth=1, edgecolor='r', facecolor='gray', alpha=0.5)
        ax.add_patch(rect)
    
    ax.scatter(0.05, 0.05, color='green', s=100, label='Start', zorder=5)
    ax.scatter(0.95, 0.95, color='blue', s=100, label='Goal', zorder=5)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title("2D Double Integrator Environment")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    plt.savefig('envs/vis/tree2d.png')

if __name__ == "__main__":
    if not os.path.exists('envs'): 
        os.makedirs('envs')
    generate_and_visualize_env()