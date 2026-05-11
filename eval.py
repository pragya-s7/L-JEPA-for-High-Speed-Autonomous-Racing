#!/usr/bin/env python3
"""
Evaluate a trained L-JEPA + PPO policy in the f1tenth gym.

Usage:
    python eval.py
    python eval.py --encoder checkpoints/pretrain/best.pt \
                   --policy  checkpoints/rl/best.pt \
                   --maps berlin vegas --episodes 20
"""
import sys
import os
import argparse
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import yaml

_DEFAULT_CFG = os.path.join(PROJECT_ROOT, 'config.yaml')
if os.path.isfile(_DEFAULT_CFG):
    with open(_DEFAULT_CFG) as _f:
        _cfg0 = yaml.safe_load(_f)
    _gym_root = _cfg0.get('gym', {}).get('gym_path', '')
else:
    _gym_root = ''
if not _gym_root:
    _gym_root = os.environ.get('F1TENTH_GYM', '')
if _gym_root and not os.path.isabs(_gym_root):
    _gym_root = os.path.join(PROJECT_ROOT, _gym_root)
GYM_PATH = _gym_root or '/home/yim/f1tenth_gym'
sys.path.insert(0, os.path.join(GYM_PATH, 'gym'))

import torch

NUM_BEAMS = 1080
FOV = 4.7
RANGE_MAX = 10.0


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_models(cfg, encoder_path, policy_path, device):
    from models import ContextEncoder
    from rl.ppo import ActorCritic, raw_to_env_action  # noqa: F401 (re-exported for callers)

    mc = cfg['model']
    encoder = ContextEncoder(
        lidar_dim=mc['lidar_dim'],
        vel_dim=mc['vel_dim'],
        context_window=mc['context_window'],
        latent_dim=mc['latent_dim'],
        conv_channels=tuple(mc['encoder']['channels']),
        conv_kernel=mc['encoder']['kernel_size'],
        conv_stride=mc['encoder']['stride'],
    ).to(device)

    ckpt = torch.load(encoder_path, map_location=device)
    key = 'encoder_state_dict' if 'encoder_state_dict' in ckpt else 'model_state_dict'
    if key == 'model_state_dict':
        state = {k.replace('context_encoder.', ''): v
                 for k, v in ckpt[key].items() if k.startswith('context_encoder.')}
    else:
        state = ckpt[key]
    encoder.load_state_dict(state)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    ac = ActorCritic(
        latent_dim=mc['latent_dim'],
        action_dim=mc['action_dim'],
        hidden_dim=128,
    ).to(device)
    ac_ckpt = torch.load(policy_path, map_location=device)
    ac.load_state_dict(ac_ckpt['actor_critic_state_dict'])
    ac.eval()

    print(f"Encoder:      {encoder_path}  (epoch {ckpt.get('epoch', '?')})")
    print(f"Actor-critic: {policy_path}  (update {ac_ckpt.get('update', '?')})")
    return encoder, ac


class ObsBuffer:
    def __init__(self, context_window, obs_dim, lidar_dim, subsample_step=5):
        self.h = context_window
        self.obs_dim = obs_dim
        self.lidar_dim = lidar_dim
        self.subsample_step = subsample_step
        self.buffer = np.zeros((context_window, obs_dim), dtype=np.float32)

    def reset(self):
        self.buffer[:] = 0.0

    def update(self, scan_raw, vel_x, ang_vel):
        scan_sub = scan_raw[::self.subsample_step].astype(np.float32)
        scan_sub = np.clip(scan_sub, 0.0, RANGE_MAX) / RANGE_MAX
        
        # Explicitly cast to float32 for Python 3.12
        v = np.float32(vel_x)
        a = np.float32(ang_vel)
        
        obs = np.concatenate([scan_sub, [v, a]], dtype=np.float32)
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = obs

    def get(self):
        return self.buffer.copy()


def run_episode(env, encoder, ac, obs_buf, start_pose, max_steps, device,
                render=False, deterministic=True, tracker=None):
    """
    Run one eval episode. If `tracker` (CenterlineTracker) is given, also reports
    real lap-progress metrics — without this the agent could circle in place and
    still score "clean 2000 steps".
    """
    # Use env.unwrapped.reset to bypass newer gym's strict API checks
    obs, _, done, _ = env.unwrapped.reset(np.array([[start_pose[0], start_pose[1], start_pose[2]]]))
    obs_buf.reset()
    obs_buf.update(obs['scans'][0], float(obs['linear_vels_x'][0]), float(obs['ang_vels_z'][0]))
    if render:
        env.unwrapped.render(mode='human_fast')

    total_reward = 0.0
    speeds = []
    step = 0
    laps_completed = 0
    forward_arc = 0.0
    max_progress = 0.0  # furthest fractional lap progress (0..1) reached this episode

    x0 = float(obs['poses_x'][0])
    y0 = float(obs['poses_y'][0])
    if tracker is not None:
        tracker.reset(x0, y0)

    from rl.ppo import raw_to_env_action

    while not done and step < max_steps:
        obs_seq = torch.from_numpy(obs_buf.get()).unsqueeze(0).to(device)
        with torch.no_grad():
            z = encoder(obs_seq)
            action, _, _ = ac.get_action(z, deterministic=deterministic)

        env_action = raw_to_env_action(action[0]).cpu().numpy()
        steer = float(env_action[0])
        speed = float(env_action[1])

        obs, _, done, _ = env.unwrapped.step(np.array([[steer, speed]]))
        collision = bool(obs['collisions'][0])
        vel_x = float(obs['linear_vels_x'][0])

        if collision:
            reward = -100.0
        else:
            reward = 3.0 * max(vel_x, 0.0) * 0.01 + 0.1

        total_reward += reward
        speeds.append(vel_x)

        if tracker is not None:
            x = float(obs['poses_x'][0])
            y = float(obs['poses_y'][0])
            th = float(obs['poses_theta'][0])
            delta_arc, _, lap_complete, _, _, _ = tracker.step(x, y, th)
            if delta_arc > 0:
                forward_arc += float(delta_arc)
            if lap_complete:
                laps_completed += 1
            if tracker.total_length > 0:
                # progress_since_reset is accumulated forward arc since last reset/lap
                frac = float(tracker._progress_since_reset) / float(tracker.total_length)
                if frac > max_progress:
                    max_progress = frac

        obs_buf.update(obs['scans'][0], vel_x, float(obs['ang_vels_z'][0]))
        step += 1

        if render:
            env.unwrapped.render(mode='human_fast')

        if collision:
            break

    return {
        'steps': step,
        'reward': total_reward,
        'collided': bool(obs['collisions'][0]),
        'mean_speed': float(np.mean(speeds)) if speeds else 0.0,
        'max_speed': float(np.max(speeds)) if speeds else 0.0,
        'forward_arc_m': forward_arc,
        'max_progress_frac': max_progress,  # in [0, 1], best partial-lap completion
        'laps_completed': laps_completed,
    }


MAP_INFO = {
    'berlin':         {'start_pose': [0.0, 0.0, 1.37]},
    'skirk':          {'start_pose': [0.0, 0.0, 0.0]},
    'vegas':          {'start_pose': [0.0, 0.0, 0.0]},
    'stata_basement': {'start_pose': [0.0, 0.0, 0.0]},
}


def _load_tracker(map_name, cfg):
    """
    Try to load a CenterlineTracker for `map_name`. Returns None if no centerline
    CSV exists for that map. The training centerlines folder uses inconsistent
    naming (`vegas_centerline.csv`, `Austin_map_centerline.csv`), so we try a
    couple of common forms.
    """
    from rl.rl_train import CenterlineTracker

    cl_dir = cfg.get('rl', {}).get('centerline_dir', '')
    if not cl_dir:
        return None
    candidates = [
        f"{map_name}_centerline.csv",
        f"{map_name.capitalize()}_map_centerline.csv",
        f"{map_name}_map_centerline.csv",
    ]
    for fname in candidates:
        full = os.path.join(cl_dir, fname)
        if os.path.isfile(full):
            return CenterlineTracker(full)
    return None


def eval_map(map_name, gym_path, encoder, ac, obs_buf, cfg, num_episodes, max_steps, device, render=False, deterministic=True):
    import gym as openai_gym
    from f110_gym.envs.base_classes import Integrator

    maps_dir = os.path.join(gym_path, 'f110_gym', 'envs', 'maps')
    map_path = os.path.join(maps_dir, map_name)

    if not os.path.exists(map_path + '.png'):
        print(f"  [SKIP] map file not found: {map_path}.png")
        return []

    env = openai_gym.make(
        'f110_gym:f110-v0',
        map=map_path,
        map_ext='.png',
        num_agents=1,
        timestep=0.01,
        integrator=Integrator.RK4,
        disable_env_checker=True
    )
    start_pose = MAP_INFO[map_name]['start_pose']

    tracker = _load_tracker(map_name, cfg)
    if tracker is None:
        print(f"  [WARN] no centerline CSV for '{map_name}' — progress metrics disabled.")

    results = []

    for ep in range(num_episodes):
        t0 = time.time()
        r = run_episode(env, encoder, ac, obs_buf, start_pose, max_steps, device,
                        render=render, deterministic=deterministic, tracker=tracker)
        elapsed = time.time() - t0
        status = 'COLLISION' if r['collided'] else 'clean    '
        if tracker is not None:
            print(f"  ep {ep+1:3d}: {r['steps']:5d} steps  {status}  "
                  f"reward {r['reward']:8.2f}  mean_spd {r['mean_speed']:.2f} m/s  "
                  f"arc {r['forward_arc_m']:6.1f} m  prog {100*r['max_progress_frac']:5.1f}%  "
                  f"laps {r['laps_completed']}  ({elapsed:.1f}s)")
        else:
            print(f"  ep {ep+1:3d}: {r['steps']:5d} steps  {status}  "
                  f"reward {r['reward']:8.2f}  mean_spd {r['mean_speed']:.2f} m/s  ({elapsed:.1f}s)")
        results.append(r)

    env.close()
    return results


def print_summary(map_name, results):
    if not results:
        return
    steps       = [r['steps']      for r in results]
    rewards     = [r['reward']     for r in results]
    speeds      = [r['mean_speed'] for r in results]
    collisions  = sum(r['collided'] for r in results)
    has_prog    = 'max_progress_frac' in results[0] and results[0].get('max_progress_frac') is not None

    print(f"\n  {map_name} summary ({len(results)} episodes):")
    print(f"    Collision rate : {collisions}/{len(results)} ({100*collisions/len(results):.0f}%)")
    print(f"    Steps          : min={min(steps)}  max={max(steps)}  mean={np.mean(steps):.0f}")
    print(f"    Reward         : min={min(rewards):.1f}  max={max(rewards):.1f}  mean={np.mean(rewards):.1f}")
    print(f"    Mean speed     : {np.mean(speeds):.2f} m/s")
    if has_prog:
        arcs   = [r['forward_arc_m']      for r in results]
        progs  = [r['max_progress_frac']  for r in results]
        laps   = [r['laps_completed']     for r in results]
        print(f"    Forward arc    : min={min(arcs):.1f}  max={max(arcs):.1f}  mean={np.mean(arcs):.1f} m")
        print(f"    Lap progress   : min={100*min(progs):.1f}%  max={100*max(progs):.1f}%  mean={100*np.mean(progs):.1f}%")
        print(f"    Laps completed : total={sum(laps)}  best_ep={max(laps)}  mean={np.mean(laps):.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',   default=os.path.join(PROJECT_ROOT, 'config.yaml'))
    parser.add_argument('--encoder',  default=None)
    parser.add_argument('--policy',   default=None)
    parser.add_argument('--maps',     nargs='+', default=['berlin', 'stata_basement', 'vegas'])
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--render', action='store_true', help='Show live simulation window')
    parser.add_argument('--stochastic', action='store_true',
                        help='Sample actions from the policy instead of using the mean (more exploration)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    mc  = cfg['model']
    rl  = cfg['rl']

    encoder_path = args.encoder or rl['encoder_checkpoint']
    policy_path  = args.policy  or os.path.join(rl['checkpoint_dir'], 'best.pt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    encoder, ac = load_models(cfg, encoder_path, policy_path, device)

    obs_buf = ObsBuffer(
        context_window=mc['context_window'],
        obs_dim=mc['obs_dim'],
        lidar_dim=mc['lidar_dim'],
        subsample_step=mc['lidar_subsample'],
    )

    gym_path = cfg.get('gym', {}).get('gym_path', GYM_PATH)
    if not os.path.isabs(gym_path):
        gym_path = os.path.join(PROJECT_ROOT, gym_path)
    gym_path = os.path.join(gym_path, 'gym')

    for map_name in args.maps:
        if map_name not in MAP_INFO:
            print(f"Unknown map: {map_name}. Options: {list(MAP_INFO.keys())}")
            continue
        print(f"\n{'='*60}")
        print(f"Map: {map_name}  |  Episodes: {args.episodes}")
        print(f"{'='*60}")
        results = eval_map(map_name, gym_path, encoder, ac, obs_buf, cfg,
                           args.episodes, cfg['data_collection']['max_steps_per_episode'], device,
                           render=args.render, deterministic=not args.stochastic)
        print_summary(map_name, results)

    print(f"\nDone.")


if __name__ == '__main__':
    main()
