import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_and_visualize_env(num_obstacles=30, size=0.05, filename='envs/tree2d.csv'):
    # Define a grid for alignment (e.g., intervals of 0.05)
    grid = np.arange(0.1, 0.9, size) 
    
    obs_list = []
    seen_coords = set()
    
    while len(obs_list) < num_obstacles:
        # Randomly pick a grid-aligned bottom-left corner
        x1 = np.random.choice(grid)
        y1 = np.random.choice(grid)
        
        # Avoid overlapping exactly on the same grid cell
        if (x1, y1) not in seen_coords:
            x2, y2 = x1 + size, y1 + size
            obs_list.append([x1, y1, x2, y2])
            seen_coords.add((x1, y1))

    # Convert to numpy and Save to CSV
    obstacles = np.array(obs_list)
    df = pd.DataFrame(obstacles)
    df.to_csv(filename, index=False)
    print(f"Environment saved to {filename}")

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Draw obstacles
    for obs in obstacles:
        rect = patches.Rectangle((obs[0], obs[1]), size, size, 
                                 linewidth=1, edgecolor='r', facecolor='gray', alpha=0.5)
        ax.add_patch(rect)
    
    # Mark Start and Goal (from your previous prompt)
    ax.scatter(0.05, 0.05, color='green', s=100, label='Start', zorder=5)
    ax.scatter(0.95, 0.95, color='blue', s=100, label='Goal', zorder=5)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title("2D Double Integrator Environment")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    plt.savefig('envs/env_visual.png')

if __name__ == "__main__":
    import os
    if not os.path.exists('envs'): os.makedirs('envs')
    generate_and_visualize_env()