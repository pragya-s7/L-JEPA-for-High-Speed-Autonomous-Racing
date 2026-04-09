#!/usr/bin/env python3
"""
Stage 2 of L-JEPA: PPO training on frozen encoder latents.

The ContextEncoder from pretraining is loaded and frozen. The PPO agent
(ActorCritic) operates entirely on the compact latent vectors z, not on
raw LiDAR data.

Reward:
  r = progress_weight * track_progress (meters along track per step)
    + collision_penalty  (if collision)
    + velocity_bonus * speed (optional)

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
ANGLE_MIN = -FOV / 2.0
ANGLE_INCREMENT = FOV / (NUM_BEAMS - 1)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_encoder(checkpoint_path, cfg, device):
    """Load a pretrained ContextEncoder from a checkpoint."""
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
        # Extract just the context_encoder weights from the full LJEPA checkpoint
        state = {
            k.replace('context_encoder.', ''): v
            for k, v in checkpoint[key].items()
            if k.startswith('context_encoder.')
        }
    else:
        state = checkpoint[key]
    encoder.load_state_dict(state)

    # Freeze encoder
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    print(f"Loaded encoder from {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")
    return encoder


class ObsBuffer:
    """
    Maintains a sliding window of (subsampled_scan + vel) observations
    for the ContextEncoder, matching what was used during pretraining.
    """

    def __init__(self, context_window, obs_dim, lidar_dim, subsample_step=5,
                 normalize=True, range_max=6.0):
        self.h = context_window
        self.obs_dim = obs_dim
        self.lidar_dim = lidar_dim
        self.subsample_step = subsample_step
        self.normalize = normalize
        self.range_max = range_max
        self.buffer = np.zeros((context_window, obs_dim), dtype=np.float32)

    def reset(self):
        self.buffer[:] = 0.0

    def update(self, scan_raw, vel_x, ang_vel):
        scan_sub = scan_raw[::self.subsample_step].astype(np.float32)
        if self.normalize:
            scan_sub = np.clip(scan_sub, 0.0, self.range_max) / self.range_max
        obs = np.concatenate([scan_sub, [vel_x, ang_vel]], dtype=np.float32)
        # Shift buffer and add new observation
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = obs

    def get(self):
        return self.buffer.copy()


def compute_reward(prev_pose, curr_pose, collision, reward_cfg):
    """
    Compute step reward.

    Progress is the displacement projected onto the forward direction
    (approximated as displacement magnitude * cos(heading_error)).
    """
    if collision:
        return reward_cfg['collision_penalty']

    dx = curr_pose[0] - prev_pose[0]
    dy = curr_pose[1] - prev_pose[1]
    # Project movement onto the car's heading at previous step
    heading = prev_pose[2]
    progress = dx * np.cos(heading) + dy * np.sin(heading)
    r = reward_cfg['progress_weight'] * max(progress, 0.0)  # only reward forward progress
    return r


def clamp_action(steer, speed, cfg):
    """Clamp raw PPO action outputs to valid gym ranges."""
    rl_cfg = cfg['rl']
    steer = float(np.clip(steer, -0.4189, 0.4189))
    speed = float(np.clip(speed, 0.5, 5.0))
    return steer, speed


def train(cfg_path, encoder_path_override=None):
    cfg = load_config(cfg_path)
    mc = cfg['model']
    rl_cfg = cfg['rl']
    reward_cfg = rl_cfg['reward']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Encoder
    encoder_path = encoder_path_override or rl_cfg['encoder_checkpoint']
    encoder = load_encoder(encoder_path, cfg, device)

    # PPO
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

    # Env
    env = gym.make(
        'f110_gym:f110-v0',
        map=rl_cfg['map_path'],
        map_ext=rl_cfg['map_ext'],
        num_agents=1,
        timestep=cfg['gym']['timestep'],
        integrator=Integrator.RK4,
    )
    start_pose = rl_cfg['start_pose']

    # Observation buffer
    obs_buf = ObsBuffer(
        context_window=mc['context_window'],
        obs_dim=mc['obs_dim'],
        lidar_dim=mc['lidar_dim'],
        subsample_step=mc['lidar_subsample'],
    )

    checkpoint_dir = rl_cfg['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    total_steps = 0
    update_count = 0
    episode_count = 0
    ep_rewards = deque(maxlen=20)
    ep_lengths = deque(maxlen=20)
    best_mean_reward = float('-inf')

    # Reset env
    obs, _, done, _ = env.reset(np.array([[start_pose[0], start_pose[1], start_pose[2]]]))
    obs_buf.reset()
    obs_buf.update(obs['scans'][0], float(obs['linear_vels_x'][0]), float(obs['ang_vels_z'][0]))
    prev_pose = [float(obs['poses_x'][0]), float(obs['poses_y'][0]), float(obs['poses_theta'][0])]

    ep_reward = 0.0
    ep_len = 0

    print(f"\nStarting RL training for {rl_cfg['total_steps']:,} steps...")

    while total_steps < rl_cfg['total_steps']:
        # Collect one rollout
        for _ in range(rl_cfg['rollout_steps']):
            # Encode current observation
            obs_seq = torch.from_numpy(obs_buf.get()).unsqueeze(0).to(device)  # (1, h, obs_dim)
            with torch.no_grad():
                z = encoder(obs_seq)  # (1, latent_dim)

            # Sample action
            action_t, log_prob, value = ppo.act(z)
            steer = float(action_t[0, 0].cpu())
            speed = float(action_t[0, 1].cpu())
            steer, speed = clamp_action(steer, speed, cfg)

            # Step env
            gym_action = np.array([[steer, speed]])
            obs, _, done, _ = env.step(gym_action)

            curr_pose = [float(obs['poses_x'][0]), float(obs['poses_y'][0]), float(obs['poses_theta'][0])]
            collision = bool(obs['collisions'][0])
            reward = compute_reward(prev_pose, curr_pose, collision, reward_cfg)

            # Velocity bonus
            if reward_cfg.get('velocity_bonus', 0.0) > 0:
                reward += reward_cfg['velocity_bonus'] * float(obs['linear_vels_x'][0])

            ppo.store(z.squeeze(0), action_t.squeeze(0), log_prob.squeeze(0),
                      reward, done, value.squeeze(0))

            ep_reward += reward
            ep_len += 1
            total_steps += 1

            if done or collision:
                ep_rewards.append(ep_reward)
                ep_lengths.append(ep_len)
                episode_count += 1
                ep_reward = 0.0
                ep_len = 0

                # Reset
                obs, _, done, _ = env.reset(np.array([[start_pose[0], start_pose[1], start_pose[2]]]))
                obs_buf.reset()
                prev_pose = [float(obs['poses_x'][0]), float(obs['poses_y'][0]), float(obs['poses_theta'][0])]
            else:
                prev_pose = curr_pose

            obs_buf.update(obs['scans'][0], float(obs['linear_vels_x'][0]), float(obs['ang_vels_z'][0]))

        # Compute last value for GAE
        obs_seq = torch.from_numpy(obs_buf.get()).unsqueeze(0).to(device)
        with torch.no_grad():
            z_last = encoder(obs_seq)
            _, _, last_value = ppo.ac.get_action(z_last)

        # PPO update
        stats = ppo.update(last_value.squeeze(0))
        update_count += 1

        if update_count % rl_cfg['log_every'] == 0:
            mean_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            mean_l = float(np.mean(ep_lengths)) if ep_lengths else 0.0
            print(f"Update {update_count:5d} | steps {total_steps:8,} | "
                  f"mean_ep_reward {mean_r:7.2f} | mean_ep_len {mean_l:6.0f} | "
                  f"pi_loss {stats['policy_loss']:.4f} | v_loss {stats['value_loss']:.4f} | "
                  f"entropy {stats['entropy']:.4f}")

        # Save best
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

    env.close()
    print(f"\nRL training done. Best mean episode reward: {best_mean_reward:.2f}")
    print(f"Checkpoints: {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join(PROJECT_ROOT, 'config.yaml'))
    parser.add_argument('--encoder', default=None, help='Path to pretrained encoder checkpoint')
    args = parser.parse_args()
    train(args.config, encoder_path_override=args.encoder)


if __name__ == '__main__':
    main()
