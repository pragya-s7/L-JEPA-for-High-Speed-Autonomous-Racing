#!/usr/bin/env python3
"""
Final Corrected Data Collection for L-JEPA.
Uses official F1TENTH racelines for perfect spawning.
"""
import sys
import os
import argparse
import numpy as np
import yaml
import glob
import random

def find_gym_path():
    candidates = [
        os.environ.get('F1TENTH_GYM_PATH'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'f1tenth_gym', 'gym'),
        os.path.join(os.path.expanduser('~'), 'f1tenth_gym', 'gym')
    ]
    for p in candidates:
        if p and os.path.isdir(os.path.join(p, 'f110_gym')): return os.path.normpath(p)
    raise RuntimeError("Cannot find f1tenth_gym.")

def find_maps_dir(gym_path):
    return os.path.join(gym_path, 'f110_gym', 'envs', 'maps')

NUM_BEAMS = 1080
FOV_RAD = 4.7
ANGLE_MIN = -FOV_RAD / 2.0
ANGLE_INCREMENT = FOV_RAD / (NUM_BEAMS - 1)
RANGE_MAX = 10.0

from controllers import FollowTheGap, PurePursuitPlanner

class WallFollower:
    def __init__(self, side='left'):
        self.side = side
        self.target_dist = 0.7
        self.prev_error = 0.0
    def plan(self, scan, *args, **kwargs):
        if self.side == 'left': s = scan[700:900]
        else: s = scan[180:380]
        s = s[s > 0.1]
        if s.size == 0: return 0.0, 1.2
        dist = np.min(s)
        error = (dist - self.target_dist) * (1 if self.side == 'right' else -1)
        steer = 1.5 * error + 0.4 * (error - self.prev_error) / 0.01
        self.prev_error = error
        return np.clip(steer, -0.4, 0.4), 1.5
    def reset(self): self.prev_error = 0.0

def load_official_raceline(map_name, maps_dir):
    """Loads the high-quality raceline or generated centerline."""
    path = os.path.join(maps_dir, f"{map_name}_raceline.csv")
    if not os.path.exists(path):
        path = os.path.join('data/centerlines', f"{map_name}_centerline.csv")
        if not os.path.exists(path): return None
    
    # Try different delimiters
    for delim in [';', ',']:
        try:
            data = np.loadtxt(path, delimiter=delim, skiprows=1)
            if data.shape[1] >= 4:
                return data[:, 1:3], data[:, 3] # X, Y, Heading
            else:
                return data[:, 1:3], None
        except: continue
    return None

def is_on_track(obs, env=None):
    if env is not None:
        # Use ground truth position from env if possible
        px, py = obs['poses_x'][0], obs['poses_y'][0]
        # env.map_free_threshold and env.map_data could be used but it's simpler to check scan
        # Actually, let's stick to scan but be more rigorous
        pass

    scan = obs['scans'][0]
    # If car is in the void, most rays hit nothing (max range)
    if np.sum(scan >= (RANGE_MAX - 0.2)) > (NUM_BEAMS * 0.5): return False
    # Check for immediate collision
    if obs['collisions'][0]: return False
    # If average scan is too large, we are likely in a huge open area (not a track)
    if np.mean(scan) > (RANGE_MAX * 0.8): return False
    return True

def collect_episode(env, controller, points, headings, rng, max_steps, subsample):
    for _ in range(50): # More attempts
        idx = rng.integers(0, len(points))
        x, y = points[idx]
        theta = headings[idx] if headings is not None else rng.uniform(0, 2*np.pi)
        
        obs, _, done, _ = env.reset(np.array([[x, y, theta]]))
        if is_on_track(obs): break
    else: return None

    if hasattr(controller, 'reset'): controller.reset()
    scans, vels, actions, poses = [], [], [], []
    
    for step in range(max_steps):
        scan = np.asarray(obs['scans'][0], dtype=np.float64)
        px, py, pt = float(obs['poses_x'][0]), float(obs['poses_y'][0]), float(obs['poses_theta'][0])
        
        try:
            if isinstance(controller, PurePursuitPlanner):
                speed, steer = controller.plan(px, py, pt)
            else:
                steer, speed = controller.plan(scan, ANGLE_MIN, ANGLE_INCREMENT, float(obs['linear_vels_x'][0]), 0.01)
        except: break

        scans.append(scan[::subsample].astype(np.float32) / RANGE_MAX)
        vels.append(np.array([obs['linear_vels_x'][0], obs['ang_vels_z'][0]], dtype=np.float32))
        actions.append(np.array([steer, speed], dtype=np.float32))
        poses.append(np.array([px, py, pt], dtype=np.float32))

        obs, _, done, _ = env.step(np.array([[steer, speed]]))
        if done or step > 2500: break

    if len(scans) < 250: return None
    return np.array(scans), np.array(vels), np.array(actions), np.array(poses), bool(obs['collisions'][0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=20)
    args = parser.parse_args()
    
    gym_path = find_gym_path()
    maps_dir = find_maps_dir(gym_path)
    rng = np.random.default_rng(42)
    
    import gym as openai_gym
    from f110_gym.envs.base_classes import Integrator

    # Discover maps
    all_yamls = sorted([f[:-5] for f in os.listdir(maps_dir) if f.endswith('.yaml')])
    # Prefer fixed maps
    fixed_maps = [f for f in all_yamls if f.endswith('_fixed')]
    base_maps = [f for f in all_yamls if not f.endswith('_fixed') and f + '_fixed' not in all_yamls]
    map_names = sorted(fixed_maps + base_maps)
    
    for name in map_names:
        # For loading raceline, use the base name
        base_name = name.replace('_fixed', '')
        res_rl = load_official_raceline(base_name, maps_dir)
        if res_rl is None: continue
        pts, heads = res_rl
        
        map_path = os.path.join(maps_dir, name)
        ext = '.png' if os.path.exists(map_path + '.png') else '.pgm'
        
        print(f"\nProcessing Map: {name}")
        try:
            env = openai_gym.make('f110_gym:f110-v0', map=map_path, map_ext=ext, num_agents=1, timestep=0.01, integrator=Integrator.RK4)
        except: continue

        # Use the official raceline for Pure Pursuit
        cl_path = os.path.join(maps_dir, f"{base_name}_raceline.csv")
        if not os.path.exists(cl_path):
            cl_path = os.path.join('data/centerlines', f"{base_name}_centerline.csv")
        
        # Determine delimiter
        delim = ';'
        with open(cl_path, 'r') as f:
            if ',' in f.readline(): delim = ','

        pp = PurePursuitPlanner(cl_path, delimiter=delim, skiprows=1, vgain=2.0, x_col=1, y_col=2, v_col=5)

        # High-quality controllers only for pretraining
        controllers = [FollowTheGap(), pp]
        out_dir = f"data/{base_name}" # Save to base name directory
        os.makedirs(out_dir, exist_ok=True)
        
        good = 0
        for i in range(args.episodes * 4):
            ctrl = random.choice(controllers)
            res = collect_episode(env, ctrl, pts, heads, rng, 3000, 5)
            if res:
                s, v, a, p, coll = res
                np.savez_compressed(f"{out_dir}/ep_{good:04d}.npz", scans=s, vels=v, actions=a, poses=p)
                good += 1
                print(f"  ep {good:2d}: {len(s):4d} steps {'💥' if coll else '✓'} ({type(ctrl).__name__})")
            if good >= args.episodes: break
        env.close()

if __name__ == "__main__":
    main()
