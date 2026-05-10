#!/usr/bin/env python3
"""
Overhauled Data Collection for L-JEPA pretraining.

Features:
    - Multi-map discovery (20+ maps)
    - Centerline-anchored random spawning (high diversity, high yield)
    - Controller Ensemble (FTG, Pure Pursuit, Wall Follower, etc.)
    - Increased prediction horizon support

Usage:
    python3 collect_data.py --episodes 10 --horizon 30
"""
import sys
import os
import argparse
import time
import numpy as np
import yaml
import glob
import random

# ---------------------------------------------------------------------------
# Gym path auto-detection
# ---------------------------------------------------------------------------
def find_gym_path(cli_override=None):
    candidates = []
    if cli_override:
        candidates.append(cli_override)
    if 'F1TENTH_GYM_PATH' in os.environ:
        candidates.append(os.environ['F1TENTH_GYM_PATH'])
    here = os.path.dirname(os.path.abspath(__file__))
    candidates += [
        os.path.join(here, '..', 'f1tenth_gym', 'gym'),
        os.path.join(os.path.expanduser('~'), 'f1tenth_gym', 'gym'),
        '/opt/f1tenth_gym/gym',
    ]
    for p in candidates:
        p = os.path.normpath(p)
        if os.path.isdir(os.path.join(p, 'f110_gym')):
            return p
    raise RuntimeError("Cannot find f1tenth_gym. Pass --gym-path.")

def find_maps_dir(gym_path):
    return os.path.join(gym_path, 'f110_gym', 'envs', 'maps')

# ---------------------------------------------------------------------------
# LiDAR constants
# ---------------------------------------------------------------------------
NUM_BEAMS = 1080
FOV_RAD = 4.7
ANGLE_MIN = -FOV_RAD / 2.0
ANGLE_INCREMENT = FOV_RAD / (NUM_BEAMS - 1)
RANGE_MAX = 10.0

# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------
from controllers import FollowTheGap, PurePursuitPlanner

class WallFollower:
    """Simple Wall Follower for variety."""
    def __init__(self, side='left'):
        self.side = side
        self.target_dist = 0.8
        self.prev_error = 0.0
        self.kp = 2.0
        self.kd = 0.5
        
    def plan(self, scan, angle_min, angle_increment, current_speed=None, dt=0.01):
        # Sample side beams
        angles = angle_min + np.arange(len(scan)) * angle_increment
        if self.side == 'left':
            mask = (angles > 1.2) & (angles < 1.7)
        else:
            mask = (angles < -1.2) & (angles > -1.7)
        
        side_ranges = scan[mask]
        side_ranges = side_ranges[side_ranges > 0]
        if side_ranges.size == 0:
            return 0.0, 1.2
            
        dist = np.mean(side_ranges)
        error = self.target_dist - dist
        if self.side == 'right': error = -error
        
        steer = self.kp * error + self.kd * (error - self.prev_error) / dt
        self.prev_error = error
        
        speed = 2.0 if abs(steer) < 0.1 else 1.2
        return np.clip(steer, -0.4, 0.4), speed

    def reset(self):
        self.prev_error = 0.0

class ConstantController:
    """Just drives straightish with noise."""
    def plan(self, scan, *args, **kwargs):
        steer = np.random.normal(0, 0.05)
        return steer, 1.5
    def reset(self): pass

# ---------------------------------------------------------------------------
# Map & Centerline logic
# ---------------------------------------------------------------------------
def discover_maps(maps_dir):
    map_names = []
    for f in os.listdir(maps_dir):
        if f.endswith('.yaml'):
            map_names.append(f[:-5])
    return sorted(map_names)

def load_centerline(map_name, centerline_dir):
    paths = [
        os.path.join(centerline_dir, f"{map_name}_centerline.csv"),
        os.path.join(centerline_dir, f"{map_name}_map_centerline.csv")
    ]
    for p in paths:
        if os.path.exists(p):
            # Format: s;x;y
            data = np.loadtxt(p, delimiter=';', skiprows=1)
            return data[:, 1:3] # Return (N, 2) array of [x, y]
    return None

def get_random_start_pose(centerline, rng):
    """Pick a random point on centerline and add noise."""
    idx = rng.integers(0, len(centerline))
    x, y = centerline[idx]
    
    # Heading: point to next waypoint
    next_idx = (idx + 1) % len(centerline)
    nx, ny = centerline[next_idx]
    theta = np.arctan2(ny - y, nx - x)
    
    # Add noise
    x += rng.normal(0, 0.3)
    y += rng.normal(0, 0.3)
    theta += rng.normal(0, 0.2)
    
    return [float(x), float(y), float(theta)]

# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------
def collect_episode(env, controller, start_pose, max_steps, subsample_step, warmup_steps=30):
    obs, _, done, _ = env.reset(np.array([[start_pose[0], start_pose[1], start_pose[2]]]))
    if hasattr(controller, 'reset'): controller.reset()

    scans_raw, vels_raw, actions_raw, poses_raw = [], [], [], []
    step = 0
    while not done and step < max_steps:
        scan_full = np.asarray(obs['scans'][0], dtype=np.float64)
        vel_x = float(obs['linear_vels_x'][0])
        ang_vel = float(obs['ang_vels_z'][0])
        pose_x = float(obs['poses_x'][0])
        pose_y = float(obs['poses_y'][0])
        pose_th = float(obs['poses_theta'][0])

        try:
            # Handle different plan signatures
            if isinstance(controller, PurePursuitPlanner):
                speed, steer = controller.plan(pose_x, pose_y, pose_th)
            else:
                steer, speed = controller.plan(scan_full, ANGLE_MIN, ANGLE_INCREMENT, vel_x, 0.01)
        except Exception as e:
            print(f"Controller error: {e}")
            break

        # Record
        scan_sub = scan_full[::subsample_step].astype(np.float32)
        scan_sub = np.clip(scan_sub, 0.0, RANGE_MAX) / RANGE_MAX
        
        scans_raw.append(scan_sub)
        vels_raw.append(np.array([vel_x, ang_vel], dtype=np.float32))
        actions_raw.append(np.array([steer, speed], dtype=np.float32))
        poses_raw.append(np.array([pose_x, pose_y, pose_th], dtype=np.float32))

        obs, _, done, _ = env.step(np.array([[steer, speed]]))
        step += 1

    if step <= warmup_steps: return None
    
    return (np.array(scans_raw[warmup_steps:], dtype=np.float32),
            np.array(vels_raw[warmup_steps:], dtype=np.float32),
            np.array(actions_raw[warmup_steps:], dtype=np.float32),
            np.array(poses_raw[warmup_steps:], dtype=np.float32),
            bool(obs['collisions'][0]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--gym-path', default=None)
    parser.add_argument('--output-dir', default='data')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    gym_path = find_gym_path(args.gym_path)
    maps_dir = find_maps_dir(gym_path)
    all_maps = discover_maps(maps_dir)
    centerline_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'centerlines')
    
    print(f"Found {len(all_maps)} maps. Centerlines in {centerline_dir}")
    
    import gym as openai_gym
    from f110_gym.envs.base_classes import Integrator

    subsample = cfg['model']['lidar_subsample']
    rng = np.random.default_rng(42)

    for map_name in all_maps:
        centerline = load_centerline(map_name, centerline_dir)
        if centerline is None:
            print(f"  [SKIP] {map_name}: No centerline found.")
            continue
        
        # Determine map extension
        map_path = os.path.join(maps_dir, map_name)
        ext = '.png' if os.path.exists(map_path + '.png') else '.pgm'
        
        print(f"\nMap: {map_name}")
        env = openai_gym.make('f110_gym:f110-v0', map=map_path, map_ext=ext, 
                              num_agents=1, timestep=0.01, integrator=Integrator.RK4)
        
        # Setup Controllers
        controllers = [
            FollowTheGap(),
            PurePursuitPlanner(os.path.join(centerline_dir, f"{map_name}_centerline.csv") if os.path.exists(os.path.join(centerline_dir, f"{map_name}_centerline.csv")) else os.path.join(centerline_dir, f"{map_name}_map_centerline.csv"), x_col=1, y_col=2, v_col=0, vgain=2.0, skiprows=1),
            WallFollower(side='left'),
            WallFollower(side='right'),
            ConstantController()
        ]

        out_dir = os.path.join(args.output_dir, map_name)
        os.makedirs(out_dir, exist_ok=True)
        
        for ep in range(args.episodes):
            start_pose = get_random_start_pose(centerline, rng)
            ctrl = random.choice(controllers)
            
            res = collect_episode(env, ctrl, start_pose, 2000, subsample)
            if res:
                scans, vels, actions, poses, collided = res
                idx = len(glob.glob(os.path.join(out_dir, 'ep_*.npz')))
                if not args.dry_run:
                    np.savez_compressed(os.path.join(out_dir, f'ep_{idx:04d}.npz'),
                                        scans=scans, vels=vels, actions=actions, poses=poses)
                status = '💥' if collided else '✓'
                print(f"  ep {ep:3d}: {len(scans):4d} steps {status} ({type(ctrl).__name__})")
        env.close()

if __name__ == '__main__':
    main()
