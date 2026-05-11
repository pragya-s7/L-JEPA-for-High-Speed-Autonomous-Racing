import os
import numpy as np
import yaml
from PIL import Image
import matplotlib.pyplot as plt
import glob

MAP_DIR = '/home/pragya/f1tenth_gym/gym/f110_gym/envs/maps'
DATA_DIR = 'data'

def visualize_trajectories(map_name):
    # Load original map metadata
    ypath = os.path.join(MAP_DIR, f"{map_name}.yaml")
    if not os.path.exists(ypath):
        print(f"Map {map_name} not found.")
        return
        
    with open(ypath, 'r') as f:
        meta = yaml.safe_load(f)
    
    res = meta['resolution']
    origin = meta['origin']
    img_path = os.path.join(MAP_DIR, meta['image'])
    img = np.array(Image.open(img_path).convert('L'))
    h, w = img.shape
    
    # Load trajectories
    traj_dir = os.path.join(DATA_DIR, map_name)
    ep_files = sorted(glob.glob(os.path.join(traj_dir, 'ep_*.npz')))
    if not ep_files:
        print(f"No data for {map_name}")
        return
        
    plt.figure(figsize=(12, 12))
    # Using 'upper' and the correct extent ensures alignment with our (H-row) logic
    plt.imshow(img, cmap='gray', origin='upper', 
               extent=[origin[0], origin[0] + w * res, origin[1], origin[1] + h * res],
               alpha=0.6)
               
    for i, f in enumerate(ep_files[:5]): # Plot first 5 episodes
        data = np.load(f)
        poses = data['poses'] # [x, y, theta]
        plt.plot(poses[:, 0], poses[:, 1], lw=1, label=f"Ep {i}")
        plt.scatter(poses[0, 0], poses[0, 1], s=20, marker='o') # Start
        
    plt.title(f"Trajectories: {map_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_png = f"trajectories_{map_name}.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved {out_png}")

if __name__ == "__main__":
    for m in ['berlin', 'Austin_map', 'vegas', 'levine']:
        visualize_trajectories(m)
