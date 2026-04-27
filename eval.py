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
import yaml

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GYM_PATH = os.path.join(
    os.path.dirname(PROJECT_ROOT),   # roboracer_ws/src/
    'f1tenth_gym', 'gym'
)
if GYM_PATH not in sys.path:
    sys.path.insert(0, GYM_PATH)
    # sys.path.insert(0, os.path.join(GYM_PATH, 'gym'))
sys.path.insert(0, PROJECT_ROOT)

import torch

NUM_BEAMS = 1080
FOV = 4.7
RANGE_MAX = 10.0


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_models(cfg, encoder_path, policy_path, device):
    from models import ContextEncoder
    from rl.ppo import ActorCritic

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
        obs = np.concatenate([scan_sub, [vel_x, ang_vel]], dtype=np.float32)
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = obs

    def get(self):
        return self.buffer.copy()


def run_episode(env, encoder, ac, obs_buf, start_pose, max_steps, device, render=False):
    obs, _, done, _ = env.reset(np.array([[start_pose[0], start_pose[1], start_pose[2]]]))
    obs_buf.reset()
    obs_buf.update(obs['scans'][0], float(obs['linear_vels_x'][0]), float(obs['ang_vels_z'][0]))

    prev_pose = [float(obs['poses_x'][0]), float(obs['poses_y'][0]), float(obs['poses_theta'][0])]

    total_reward = 0.0
    speeds = []
    step = 0

    while not done and step < max_steps:
        obs_seq = torch.from_numpy(obs_buf.get()).unsqueeze(0).to(device)
        with torch.no_grad():
            z = encoder(obs_seq)
            action, _, _ = ac.get_action(z, deterministic=True)

        steer = float(np.clip(action[0, 0].cpu(), -0.4189, 0.4189))
        speed = float(np.clip(action[0, 1].cpu(), 0.5, 5.0))

        obs, _, done, _ = env.step(np.array([[steer, speed]]))
        collision = bool(obs['collisions'][0])

        curr_pose = [float(obs['poses_x'][0]), float(obs['poses_y'][0]), float(obs['poses_theta'][0])]
        dx = curr_pose[0] - prev_pose[0]
        dy = curr_pose[1] - prev_pose[1]
        heading = prev_pose[2]
        progress = dx * np.cos(heading) + dy * np.sin(heading)
        reward = max(progress, 0.0) if not collision else -10.0

        total_reward += reward
        speeds.append(float(obs['linear_vels_x'][0]))
        obs_buf.update(obs['scans'][0], float(obs['linear_vels_x'][0]), float(obs['ang_vels_z'][0]))
        prev_pose = curr_pose
        step += 1

        if render:
            env.render(mode='human')

        if collision:
            break

    return {
        'steps': step,
        'reward': total_reward,
        'collided': bool(obs['collisions'][0]),
        'mean_speed': float(np.mean(speeds)) if speeds else 0.0,
        'max_speed': float(np.max(speeds)) if speeds else 0.0,
    }


MAP_INFO = {
    'berlin':         {'start_pose': [0.0, 0.0, 1.37]},
    'skirk':          {'start_pose': [0.0, 0.0, 0.0]},
    'vegas':          {'start_pose': [0.0, 0.0, 0.0]},
    'stata_basement': {'start_pose': [0.0, 0.0, 0.0]},
    'levine':         {'start_pose': [0.0, 0.0, 0.0]},
}

from f110_gym.envs.base_classes import Integrator

def eval_map(map_name, gym_path, encoder, ac, obs_buf, cfg, num_episodes, max_steps, device, render=False):
    from f110_gym.envs.f110_env import F110Env

    maps_dir = os.path.join(gym_path, 'f110_gym', 'envs', 'maps')
    map_path = os.path.join(maps_dir, map_name)

    if not os.path.exists(map_path + '.png'):
        print(f"  [SKIP] map file not found: {map_path}.png")
        return []

    env = F110Env(
        map=map_path,
        map_ext='.png',
        num_agents=1,
        timestep=0.01,
        integrator=Integrator.RK4,
    )
    start_pose = MAP_INFO[map_name]['start_pose']
    results = []

    for ep in range(num_episodes):
        t0 = time.time()
        r = run_episode(env, encoder, ac, obs_buf, start_pose, max_steps, device, render=render)
        elapsed = time.time() - t0
        status = 'COLLISION' if r['collided'] else 'clean    '
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

    print(f"\n  {map_name} summary ({len(results)} episodes):")
    print(f"    Collision rate : {collisions}/{len(results)} ({100*collisions/len(results):.0f}%)")
    print(f"    Steps          : min={min(steps)}  max={max(steps)}  mean={np.mean(steps):.0f}")
    print(f"    Reward         : min={min(rewards):.1f}  max={max(rewards):.1f}  mean={np.mean(rewards):.1f}")
    print(f"    Mean speed     : {np.mean(speeds):.2f} m/s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',   default=os.path.join(PROJECT_ROOT, 'config.yaml'))
    parser.add_argument('--encoder',  default=None)
    parser.add_argument('--policy',   default=None)
    parser.add_argument('--maps',     nargs='+', default=['berlin'])
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--render', action='store_true')
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

    gym_path = os.path.join(GYM_PATH)
    print(gym_path)
    for map_name in args.maps:
        if map_name not in MAP_INFO:
            print(f"Unknown map: {map_name}. Options: {list(MAP_INFO.keys())}")
            continue
        print(f"\n{'='*60}")
        print(f"Map: {map_name}  |  Episodes: {args.episodes}")
        print(f"{'='*60}")
        results = eval_map(map_name, gym_path, encoder, ac, obs_buf, cfg,
                           args.episodes, cfg['data_collection']['max_steps_per_episode'], device, render=args.render)
        print_summary(map_name, results)

    print(f"\nDone.")


if __name__ == '__main__':
    main()