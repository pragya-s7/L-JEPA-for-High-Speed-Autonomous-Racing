#!/usr/bin/env python3
"""
Stage 2 of L-JEPA: PPO training on frozen encoder latents.

Trains across multiple maps by randomly selecting a new map at each episode
reset. This prevents the policy from overfitting to a single track layout and
produces a controller that generalises to unseen maps (including the real car).

Usage:
    python rl/rl_train.py
    python rl/rl_train.py --config ../config.yaml --encoder ../checkpoints/pretrain/best.pt
"""
import sys
import os
import argparse
import time
from collections import deque

import numpy as np
import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

GYM_PATH = '/home/pragya/f1tenth_gym'
if GYM_PATH not in sys.path:
    sys.path.insert(0, os.path.join(GYM_PATH, 'gym'))

import gym
from f110_gym.envs.base_classes import Integrator
from models import ContextEncoder
from rl.ppo import PPO


# LiDAR constants (must match collect_data.py)
NUM_BEAMS = 1080
FOV = 4.7
RANGE_MAX = 10.0   # must match RANGE_MAX in collect_data.py


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_encoder(checkpoint_path, cfg, device):
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

    checkpoint = torch.load(checkpoint_path, map_location=device)
    key = 'encoder_state_dict' if 'encoder_state_dict' in checkpoint else 'model_state_dict'
    if key == 'model_state_dict':
        state = {
            k.replace('context_encoder.', ''): v
            for k, v in checkpoint[key].items()
            if k.startswith('context_encoder.')
        }
    else:
        state = checkpoint[key]
    encoder.load_state_dict(state)
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    print(f"Loaded encoder from {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")
    return encoder


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


def make_env(map_path, map_ext, timestep):
    return gym.make(
        'f110_gym:f110-v0',
        map=map_path,
        map_ext=map_ext,
        num_agents=1,
        timestep=timestep,
        integrator=Integrator.RK4,
    )


def compute_reward(prev_pose, curr_pose, collision, reward_cfg):
    if collision:
        return reward_cfg['collision_penalty']
    dx = curr_pose[0] - prev_pose[0]
    dy = curr_pose[1] - prev_pose[1]
    heading = prev_pose[2]
    progress = dx * np.cos(heading) + dy * np.sin(heading)
    r = reward_cfg['progress_weight'] * max(progress, 0.0)
    if reward_cfg.get('velocity_bonus', 0.0) > 0:
        r += reward_cfg['velocity_bonus'] * np.sqrt(dx*dx + dy*dy) / 0.01
    return r


def clamp_action(steer, speed):
    return float(np.clip(steer, -0.4189, 0.4189)), float(np.clip(speed, 0.5, 5.0))


def build_map_pool(cfg):
    """
    Build a list of (env, start_pose, name) from rl.training_maps in config.
    Falls back to the single rl.map_name if training_maps is not set.
    Skips maps whose PNG file is missing.
    """
    rl_cfg = cfg['rl']
    maps_dir = os.path.join(GYM_PATH, 'gym', 'f110_gym', 'envs', 'maps')
    timestep = cfg['gym']['timestep']

    training_maps = rl_cfg.get('training_maps', None)
    if not training_maps:
        # Fallback: single map from old config
        training_maps = [{
            'name':       rl_cfg['map_name'],
            'map_path':   rl_cfg['map_path'],
            'map_ext':    rl_cfg['map_ext'],
            'start_pose': rl_cfg['start_pose'],
        }]

    pool = []
    for m in training_maps:
        path = m['map_path']
        ext  = m.get('map_ext', '.png')
        if not os.path.exists(path + ext):
            print(f"  [SKIP] map not found: {path}{ext}")
            continue
        print(f"  Loading map: {m['name']}")
        env = make_env(path, ext, timestep)
        pool.append({'env': env, 'start_pose': m['start_pose'], 'name': m['name']})

    if not pool:
        raise RuntimeError("No valid training maps found. Check paths in config.yaml.")
    return pool


def train(cfg_path, encoder_path_override=None):
    cfg = load_config(cfg_path)
    mc = cfg['model']
    rl_cfg = cfg['rl']
    reward_cfg = rl_cfg['reward']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    encoder_path = encoder_path_override or rl_cfg['encoder_checkpoint']
    encoder = load_encoder(encoder_path, cfg, device)

    ppo = PPO(
        latent_dim=mc['latent_dim'],
        action_dim=mc['action_dim'],
        hidden_dim=128,
        lr=rl_cfg['lr'],
        clip_eps=rl_cfg['clip_eps'],
        value_coef=rl_cfg['value_coef'],
        entropy_coef=rl_cfg['entropy_coef'],
        epochs_per_update=rl_cfg['epochs_per_update'],
        minibatch_size=rl_cfg['minibatch_size'],
        max_grad_norm=rl_cfg['max_grad_norm'],
        gamma=rl_cfg['gamma'],
        gae_lambda=rl_cfg['gae_lambda'],
        rollout_steps=rl_cfg['rollout_steps'],
        device=str(device),
    )
    print(f"PPO actor-critic: {sum(p.numel() for p in ppo.ac.parameters()):,} parameters")

    print("\nBuilding map pool...")
    map_pool = build_map_pool(cfg)
    print(f"Training on {len(map_pool)} map(s): {[m['name'] for m in map_pool]}")

    obs_buf = ObsBuffer(
        context_window=mc['context_window'],
        obs_dim=mc['obs_dim'],
        lidar_dim=mc['lidar_dim'],
        subsample_step=mc['lidar_subsample'],
    )

    checkpoint_dir = rl_cfg['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    rng = np.random.default_rng(seed=0)

    # Start on a random map
    current_map = map_pool[rng.integers(len(map_pool))]
    env = current_map['env']
    start_pose = current_map['start_pose']

    obs, _, done, _ = env.reset(np.array([[start_pose[0], start_pose[1], start_pose[2]]]))
    obs_buf.reset()
    obs_buf.update(obs['scans'][0], float(obs['linear_vels_x'][0]), float(obs['ang_vels_z'][0]))
    prev_pose = [float(obs['poses_x'][0]), float(obs['poses_y'][0]), float(obs['poses_theta'][0])]

    total_steps = 0
    update_count = 0
    episode_count = 0
    ep_rewards = deque(maxlen=20)
    ep_lengths = deque(maxlen=20)
    map_counts = {m['name']: 0 for m in map_pool}
    best_mean_reward = float('-inf')
    ep_reward = 0.0
    ep_len = 0

    print(f"\nStarting RL training for {rl_cfg['total_steps']:,} steps...")

    while total_steps < rl_cfg['total_steps']:
        for _ in range(rl_cfg['rollout_steps']):
            obs_seq = torch.from_numpy(obs_buf.get()).unsqueeze(0).to(device)
            with torch.no_grad():
                z = encoder(obs_seq)

            action_t, log_prob, value = ppo.act(z)
            steer, speed = clamp_action(float(action_t[0, 0].cpu()), float(action_t[0, 1].cpu()))

            obs, _, done, _ = env.step(np.array([[steer, speed]]))
            curr_pose = [float(obs['poses_x'][0]), float(obs['poses_y'][0]), float(obs['poses_theta'][0])]
            collision = bool(obs['collisions'][0])
            reward = compute_reward(prev_pose, curr_pose, collision, reward_cfg)

            ppo.store(z.squeeze(0), action_t.squeeze(0), log_prob.squeeze(0),
                      reward, done, value.squeeze(0))

            ep_reward += reward
            ep_len += 1
            total_steps += 1

            if done or collision:
                ep_rewards.append(ep_reward)
                ep_lengths.append(ep_len)
                episode_count += 1
                map_counts[current_map['name']] += 1
                ep_reward = 0.0
                ep_len = 0

                # Switch to a random map each episode
                current_map = map_pool[rng.integers(len(map_pool))]
                env = current_map['env']
                start_pose = current_map['start_pose']
                obs, _, done, _ = env.reset(np.array([[start_pose[0], start_pose[1], start_pose[2]]]))
                obs_buf.reset()
                prev_pose = [float(obs['poses_x'][0]), float(obs['poses_y'][0]), float(obs['poses_theta'][0])]
            else:
                prev_pose = curr_pose

            obs_buf.update(obs['scans'][0], float(obs['linear_vels_x'][0]), float(obs['ang_vels_z'][0]))

        # GAE last value
        obs_seq = torch.from_numpy(obs_buf.get()).unsqueeze(0).to(device)
        with torch.no_grad():
            z_last = encoder(obs_seq)
            _, _, last_value = ppo.ac.get_action(z_last)

        stats = ppo.update(last_value.squeeze(0))
        update_count += 1

        if update_count % rl_cfg['log_every'] == 0:
            mean_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            mean_l = float(np.mean(ep_lengths)) if ep_lengths else 0.0
            print(f"Update {update_count:5d} | steps {total_steps:8,} | "
                  f"mean_ep_reward {mean_r:7.2f} | mean_ep_len {mean_l:6.0f} | "
                  f"pi_loss {stats['policy_loss']:.4f} | v_loss {stats['value_loss']:.4f} | "
                  f"entropy {stats['entropy']:.4f}")

        if ep_rewards:
            mean_r = float(np.mean(ep_rewards))
            if mean_r > best_mean_reward:
                best_mean_reward = mean_r
                torch.save({
                    'update': update_count,
                    'total_steps': total_steps,
                    'actor_critic_state_dict': ppo.ac.state_dict(),
                    'optimizer_state_dict': ppo.optimizer.state_dict(),
                    'mean_reward': best_mean_reward,
                    'config': cfg,
                }, os.path.join(checkpoint_dir, 'best.pt'))

        if update_count % 50 == 0:
            torch.save({
                'update': update_count,
                'total_steps': total_steps,
                'actor_critic_state_dict': ppo.ac.state_dict(),
                'config': cfg,
            }, os.path.join(checkpoint_dir, f'update_{update_count:05d}.pt'))

    for m in map_pool:
        m['env'].close()

    print(f"\nRL training done. Best mean episode reward: {best_mean_reward:.2f}")
    print(f"Episodes per map: {map_counts}")
    print(f"Checkpoints: {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join(PROJECT_ROOT, 'config.yaml'))
    parser.add_argument('--encoder', default=None)
    args = parser.parse_args()
    train(args.config, encoder_path_override=args.encoder)


if __name__ == '__main__':
    main()
