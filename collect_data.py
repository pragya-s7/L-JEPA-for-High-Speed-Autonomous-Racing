#!/usr/bin/env python3
"""
Data collection for L-JEPA pretraining.

Runs Follow-the-Gap in f1tenth_gym and saves trajectory data.

Each episode is saved as its own .npz file under data/{map_name}/ep_NNNN.npz.
Keeping episodes separate is critical: the L-JEPA dataset must never create a
training sample whose context window spans an episode boundary (post-reset
observations are from a different track position and are unrelated).

Saved arrays per episode:
    scans:   (T, lidar_subsampled_dim)  float32  — normalised to [0, 1]
    vels:    (T, 2)                     float32  — [linear_vels_x, ang_vels_z]
    actions: (T, 2)                     float32  — [steer, speed] sent to gym
    poses:   (T, 3)                     float32  — [x, y, theta] for diagnostics

Gym path resolution (in order):
    1. --gym-path CLI argument
    2. F1TENTH_GYM_PATH environment variable
    3. Sibling directory  <this_script>/../f1tenth_gym/gym
    4. ~/f1tenth_gym/gym

Usage:
    python3 collect_data.py
    python3 collect_data.py --maps berlin vegas --episodes 30
    python3 collect_data.py --gym-path /path/to/f1tenth_gym/gym
    python3 collect_data.py --dry-run          # validate without saving
"""
import sys
import os
import argparse
import time
import numpy as np
import yaml
import glob

# ---------------------------------------------------------------------------
# Optional: headless display for servers without a monitor
# ---------------------------------------------------------------------------
def _setup_headless():
    """
    Attempt to set up a virtual X display for headless environments.
    Silently skips if not needed or not available.
    """
    if os.environ.get('DISPLAY'):
        return  # real display present
    try:
        from xvfbwrapper import Xvfb
        vdisplay = Xvfb(width=1280, height=720, colordepth=24)
        vdisplay.start()
        import atexit
        atexit.register(vdisplay.stop)
        print("[INFO] Started virtual display via xvfbwrapper.")
    except ImportError:
        # No xvfbwrapper — try setting a dummy DISPLAY.
        # If this is a real headless machine, collection will still work as
        # long as render() is never called (we never call it here).
        os.environ.setdefault('DISPLAY', ':0')


# ---------------------------------------------------------------------------
# Gym path auto-detection
# ---------------------------------------------------------------------------
def find_gym_path(cli_override=None):
    """Return the path to f1tenth_gym/gym (the one containing f110_gym/)."""
    candidates = []
    if cli_override:
        candidates.append(cli_override)
    if 'F1TENTH_GYM_PATH' in os.environ:
        candidates.append(os.environ['F1TENTH_GYM_PATH'])
    # Sibling of this script's parent (ljepa/ → f1tenth_gym/)
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
    raise RuntimeError(
        "Cannot find f1tenth_gym. Pass --gym-path or set F1TENTH_GYM_PATH.\n"
        "  Example: export F1TENTH_GYM_PATH=~/f1tenth_gym/gym"
    )


def find_maps_dir(gym_path):
    return os.path.join(gym_path, 'f110_gym', 'envs', 'maps')


# ---------------------------------------------------------------------------
# Map catalogue
# Maps that have a .png file and a confirmed working start pose.
# start_pose: [x, y, theta_rad]
# ---------------------------------------------------------------------------
MAP_CATALOGUE = {
    'berlin':         {'ext': '.png', 'start_pose': [0.0, 0.0, 1.37]},
    'skirk':          {'ext': '.png', 'start_pose': [0.0, 0.0, 0.0]},
    'vegas':          {'ext': '.png', 'start_pose': [0.0, 0.0, 0.0]},
    'stata_basement': {'ext': '.png', 'start_pose': [0.0, 0.0, 0.0]},
}

# LiDAR constants (f1tenth_gym defaults, do not change unless you modified the gym)
NUM_BEAMS = 1080
FOV_RAD = 4.7
ANGLE_MIN = -FOV_RAD / 2.0
ANGLE_INCREMENT = FOV_RAD / (NUM_BEAMS - 1)
RANGE_MAX = 10.0   # clip and normalise to this — 6 m clips to indoor corridor scale
                   # 10 m avoids clipping at the start of straight sections


# ---------------------------------------------------------------------------
# Per-episode collection
# ---------------------------------------------------------------------------
def _jitter_pose(start_pose, xy_std=0.3, theta_std=0.08, rng=None):
    """
    Add small Gaussian noise to start pose so repeated episodes explore
    different parts of the track even from the same nominal start.

    xy_std:    metres, 1-sigma lateral/longitudinal jitter (~1 car width)
    theta_std: radians (~4.6 deg default) — keeps the car roughly aligned
    """
    if rng is None:
        rng = np.random.default_rng()
    x  = start_pose[0] + rng.normal(0, xy_std)
    y  = start_pose[1] + rng.normal(0, xy_std)
    th = start_pose[2] + rng.normal(0, theta_std)
    return [x, y, th]


def collect_episode(env, controller, start_pose, max_steps, subsample_step,
                    warmup_steps=30, normalize=True, jitter=True, rng=None):
    """
    Run one episode and return trajectory arrays.

    warmup_steps: discard the first N steps while the car accelerates from rest.
    jitter:       add small Gaussian noise to start pose for trajectory diversity.
    Returns None if the episode ended during warmup (rare early collision).
    """
    pose = _jitter_pose(start_pose, rng=rng) if jitter else start_pose
    obs, _, done, _ = env.reset(np.array([[pose[0], pose[1], pose[2]]]))
    controller.reset()

    scans_raw = []
    vels_raw = []
    actions_raw = []
    poses_raw = []

    step = 0
    while not done and step < max_steps:
        scan_full = np.asarray(obs['scans'][0], dtype=np.float64)
        vel_x = float(obs['linear_vels_x'][0])
        ang_vel = float(obs['ang_vels_z'][0])
        pose_x = float(obs['poses_x'][0])
        pose_y = float(obs['poses_y'][0])
        pose_th = float(obs['poses_theta'][0])

        steer, speed = controller.plan(
            scan_full,
            angle_min=ANGLE_MIN,
            angle_increment=ANGLE_INCREMENT,
            current_speed=vel_x,
            dt=0.01,
        )

        # Record current observation and action
        scan_sub = scan_full[::subsample_step].astype(np.float32)
        if normalize:
            scan_sub = np.clip(scan_sub, 0.0, RANGE_MAX) / RANGE_MAX

        scans_raw.append(scan_sub)
        vels_raw.append(np.array([vel_x, ang_vel], dtype=np.float32))
        actions_raw.append(np.array([steer, speed], dtype=np.float32))
        poses_raw.append(np.array([pose_x, pose_y, pose_th], dtype=np.float32))

        obs, _, done, _ = env.step(np.array([[steer, speed]]))
        step += 1

    if step <= warmup_steps:
        return None  # too short even with warmup

    # Drop warmup steps at the start (car is near-stationary)
    scans = np.array(scans_raw[warmup_steps:], dtype=np.float32)
    vels = np.array(vels_raw[warmup_steps:], dtype=np.float32)
    actions = np.array(actions_raw[warmup_steps:], dtype=np.float32)
    poses = np.array(poses_raw[warmup_steps:], dtype=np.float32)
    collided = bool(obs['collisions'][0])

    return scans, vels, actions, poses, collided


# ---------------------------------------------------------------------------
# Map collection
# ---------------------------------------------------------------------------
def collect_map(map_name, map_info, gym_path, num_episodes, output_dir,
                max_steps, min_steps, subsample_step, dry_run=False):
    """Collect episodes for one map and save to disk."""
    import gym as openai_gym
    from f110_gym.envs.base_classes import Integrator

    maps_dir = find_maps_dir(gym_path)
    map_path = os.path.join(maps_dir, map_name)
    map_ext = map_info['ext']
    start_pose = map_info['start_pose']

    # Verify map file exists
    if not os.path.exists(map_path + map_ext):
        print(f"  [SKIP] Map file not found: {map_path}{map_ext}")
        return 0, 0

    print(f"\n{'='*60}")
    print(f"Map: {map_name}  |  Episodes: {num_episodes}  |  Start: {start_pose}")
    print(f"{'='*60}")

    env = openai_gym.make(
        'f110_gym:f110-v0',
        map=map_path,
        map_ext=map_ext,
        num_agents=1,
        timestep=0.01,
        integrator=Integrator.RK4,
    )
    from controllers import FollowTheGap
    # from controllers import PurePursuit
    from controllers import RRT
    # controller = FollowTheGap()
    controller = RRT()  # or FollowTheGap()

    map_out_dir = os.path.join(output_dir, map_name)
    if not dry_run:
        os.makedirs(map_out_dir, exist_ok=True)

    # Find the next episode index (safe to resume partial collections)
    existing = sorted(glob.glob(os.path.join(map_out_dir, 'ep_*.npz')))
    ep_start_idx = len(existing)

    total_steps = 0
    good_eps = 0
    collision_eps = 0
    step_counts = []
    rng = np.random.default_rng(seed=42)  # reproducible across machines

    try:
        for ep in range(num_episodes):
            t0 = time.time()
            result = collect_episode(
                env, controller, start_pose,
                max_steps=max_steps,
                subsample_step=subsample_step,
                jitter=True,
                rng=rng,
            )
            elapsed = time.time() - t0

            if result is None:
                print(f"  ep {ep+1:3d}: too short (warmup not reached) — skipping")
                continue

            scans, vels, actions, poses, collided = result
            T = len(scans)

            if T < min_steps:
                print(f"  ep {ep+1:3d}: {T:4d} steps — below min_steps={min_steps}, skipping")
                continue

            good_eps += 1
            total_steps += T
            step_counts.append(T)
            if collided:
                collision_eps += 1

            status = '💥 collided' if collided else '✓ clean'
            print(f"  ep {ep+1:3d}: {T:4d} steps  {status}  ({elapsed:.1f}s)")

            if not dry_run:
                ep_idx = ep_start_idx + good_eps - 1
                out_path = os.path.join(map_out_dir, f'ep_{ep_idx:04d}.npz')
                np.savez_compressed(out_path, scans=scans, vels=vels,
                                    actions=actions, poses=poses)
    finally:
        env.close()

    if good_eps == 0:
        print(f"  WARNING: 0 usable episodes collected for {map_name}!")
        print(f"  Check that start_pose {start_pose} is inside the track.")
        return 0, 0

    print(f"\n  Summary: {good_eps}/{num_episodes} episodes saved "
          f"({collision_eps} collisions), {total_steps} total steps")
    if step_counts:
        print(f"  Step counts — min: {min(step_counts)}, "
              f"max: {max(step_counts)}, mean: {np.mean(step_counts):.0f}")
    if not dry_run:
        print(f"  Saved to: {map_out_dir}/")
    return good_eps, total_steps


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------
def print_dataset_stats(output_dir):
    """Print a summary of what's been collected."""
    print(f"\n{'='*60}")
    print(f"Dataset at: {output_dir}")
    print(f"{'='*60}")
    grand_total = 0
    for map_dir in sorted(glob.glob(os.path.join(output_dir, '*'))):
        if not os.path.isdir(map_dir):
            continue
        eps = sorted(glob.glob(os.path.join(map_dir, 'ep_*.npz')))
        if not eps:
            continue
        map_name = os.path.basename(map_dir)
        total_steps = 0
        for ep_path in eps:
            d = np.load(ep_path)
            total_steps += len(d['scans'])
        grand_total += total_steps
        print(f"  {map_name:20s}: {len(eps):4d} episodes, {total_steps:7,} steps")
    print(f"  {'TOTAL':20s}: {grand_total:7,} steps")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Collect L-JEPA pretraining data')
    parser.add_argument('--config',
                        default=os.path.join(os.path.dirname(__file__), 'config.yaml'))
    parser.add_argument('--gym-path', default=None,
                        help='Path to f1tenth_gym/gym directory. '
                             'Overrides auto-detection and F1TENTH_GYM_PATH env var.')
    parser.add_argument('--maps', nargs='+', default=None,
                        help='Maps to collect (default: all). '
                             'Options: berlin skirk vegas stata_basement')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Episodes per map (overrides config)')
    parser.add_argument('--output-dir', default=None,
                        help='Where to save data (overrides config)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run without saving — useful for verifying setup')
    parser.add_argument('--stats', action='store_true',
                        help='Print dataset statistics and exit')
    args = parser.parse_args()

    cfg = load_config(args.config)
    dc = cfg['data_collection']

    output_dir = args.output_dir or dc['output_dir']
    episodes_per_map = args.episodes or dc['episodes_per_map']
    max_steps = dc['max_steps_per_episode']
    min_steps = dc['min_episode_steps']
    subsample_step = cfg['model']['lidar_subsample']

    if args.stats:
        print_dataset_stats(output_dir)
        return

    # Gym path
    gym_path = find_gym_path(args.gym_path)
    print(f"Using gym at: {gym_path}")

    # Add gym to path
    if gym_path not in sys.path:
        sys.path.insert(0, gym_path)

    # Install gym env if needed
    try:
        import gym as openai_gym
        openai_gym.make  # ensure it's importable
    except Exception as e:
        raise RuntimeError(
            f"Could not import gym: {e}\n"
            f"Install with: cd {os.path.dirname(gym_path)} && pip install -e ."
        )

    # Try headless setup (safe no-op if not needed)
    _setup_headless()

    # Determine maps
    if args.maps:
        selected = {m: MAP_CATALOGUE[m] for m in args.maps if m in MAP_CATALOGUE}
        unknown = [m for m in args.maps if m not in MAP_CATALOGUE]
        if unknown:
            print(f"[WARN] Unknown maps ignored: {unknown}")
    else:
        selected = MAP_CATALOGUE

    if not selected:
        print("No valid maps selected. Available:", list(MAP_CATALOGUE.keys()))
        return

    lidar_out_dim = NUM_BEAMS // subsample_step
    print(f"LiDAR: {NUM_BEAMS} beams → subsample every {subsample_step} → {lidar_out_dim} dims")
    print(f"Episodes per map: {episodes_per_map}  |  Max steps: {max_steps}")
    if args.dry_run:
        print("[DRY RUN] No files will be written.")

    grand_eps, grand_steps = 0, 0
    for map_name, map_info in selected.items():
        ep_count, step_count = collect_map(
            map_name=map_name,
            map_info=map_info,
            gym_path=gym_path,
            num_episodes=episodes_per_map,
            output_dir=output_dir,
            max_steps=max_steps,
            min_steps=min_steps,
            subsample_step=subsample_step,
            dry_run=args.dry_run,
        )
        grand_eps += ep_count
        grand_steps += step_count

    print(f"\n{'='*60}")
    print(f"DONE: {grand_eps} episodes, {grand_steps:,} steps total")
    if not args.dry_run and grand_steps > 0:
        print_dataset_stats(output_dir)
        print(f"\nNext step: python3 training/pretrain.py")


if __name__ == '__main__':
    main()
