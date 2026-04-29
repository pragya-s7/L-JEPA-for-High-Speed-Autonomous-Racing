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
from scipy.spatial import cKDTree

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
        
        # Explicitly cast scalars to float32 to avoid Python 3.12 overflow warnings during concat
        v = np.float32(vel_x)
        a = np.float32(ang_vel)
        
        obs = np.concatenate([scan_sub, [v, a]], dtype=np.float32)
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = obs

    def get(self):
        return self.buffer.copy()


class CenterlineTracker:
    """
    Tracks a car's arc-length progress and lateral deviation along a pre-built
    centerline CSV (columns: s_m; x_m; y_m).

    For closed-loop tracks (loop_gap < 5 m) the arc-length delta wraps around
    correctly at the finish line.  For open-ended centerlines (e.g. stata_basement
    where one episode doesn't cover a full lap) wrap-around is disabled and the
    tracker simply accumulates forward progress within each episode.
    """

    _CLOSED_GAP_M = 5.0

    def __init__(self, csv_path):
        data = np.loadtxt(csv_path, delimiter=';', skiprows=1)
        self.s = data[:, 0].astype(np.float32)
        self.x = data[:, 1].astype(np.float32)
        self.y = data[:, 2].astype(np.float32)
        self.total_length = float(self.s[-1])
        self._tree = cKDTree(np.column_stack([self.x, self.y]))
        self._last_s = None

        loop_gap = float(np.hypot(self.x[-1] - self.x[0], self.y[-1] - self.y[0]))
        self.is_closed = loop_gap < self._CLOSED_GAP_M
        print(f"    centerline: {len(self.s)} wp  length={self.total_length:.1f}m  "
              f"{'closed' if self.is_closed else 'open (no wrap-around)'}")

    def reset(self, x, y):
        _, idx = self._tree.query([x, y])
        self._last_s = float(self.s[idx])

    def step(self, x, y):
        """Returns (delta_arc_m, lateral_m)."""
        lateral, idx = self._tree.query([x, y])
        curr_s = float(self.s[idx])
        if self._last_s is None:
            self._last_s = curr_s
            return 0.0, float(lateral)
        delta_s = curr_s - self._last_s
        if self.is_closed:
            half = self.total_length / 2.0
            if delta_s > half:
                delta_s -= self.total_length
            elif delta_s < -half:
                delta_s += self.total_length
        self._last_s = curr_s
        return delta_s, float(lateral)


def make_env(map_path, map_ext, timestep):
    # Disable the env checker to prevent AssertionError on newer gym versions (0.25+)
    # while using the legacy f110_gym.
    return gym.make(
        'f110_gym:f110-v0',
        map=map_path,
        map_ext=map_ext,
        num_agents=1,
        timestep=timestep,
        integrator=Integrator.RK4,
        disable_env_checker=True
    )


class DomainRandomizer:
    def __init__(self, config):
        self.cfg = config.get('domain_randomization', {})
        self.enabled = self.cfg.get('enabled', False)
        self.action_delay_buffer = deque(maxlen=10)
        self.current_delay = 0

    def randomize(self, env):
        if not self.enabled:
            return
        
        # Access existing params to avoid wiping them out (gym is destructive)
        # Correct path for f110_gym: env.unwrapped.sim.agents[0].params
        try:
            current_params = env.unwrapped.sim.agents[0].params.copy()
        except AttributeError:
            try:
                current_params = env.unwrapped.agents[0].params.copy()
            except AttributeError:
                # Comprehensive fallback with all required SOTA constants
                current_params = {
                    'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'm': 3.74,
                    'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'I': 0.04712,
                    's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2,
                    'v_switch': 7.319, 'a_max': 9.51, 'v_min': -5.0, 'v_max': 20.0,
                    'width': 0.31, 'length': 0.58
                }

        # Nominal values (from f1tenth_rl_humble or standard params)
        mu = np.random.uniform(*self.cfg.get('friction_range', [0.6, 1.2]))
        m = np.random.uniform(*self.cfg.get('mass_range', [3.0, 4.5]))
        stiff_scale = np.random.uniform(*self.cfg.get('stiffness_range', [0.8, 1.2]))
        
        # Update only specific keys
        current_params['mu'] = mu
        current_params['m'] = m
        current_params['C_Sf'] = 4.718 * stiff_scale
        current_params['C_Sr'] = 5.4562 * stiff_scale

        # f110_gym (old) update_params takes a dict
        # env.unwrapped is the F110Env which supports (params, agent_idx) positionally
        env.unwrapped.update_params(current_params, 0)

        # Action delay randomization
        max_delay = self.cfg.get('max_action_delay', 3)
        self.current_delay = np.random.randint(0, max_delay + 1)
        self.action_delay_buffer.clear()

    def apply_action_delay(self, action):
        if not self.enabled or self.current_delay == 0:
            return action
        
        self.action_delay_buffer.append(action)
        if len(self.action_delay_buffer) > self.current_delay:
            return self.action_delay_buffer[0]
        else:
            return np.array([0.0, 0.0], dtype=np.float32)

    def apply_sensor_noise(self, scan):
        if not self.enabled:
            return scan
        
        # Gaussian noise
        noise_std = self.cfg.get('lidar_noise_std', 0.02)
        scan = scan + np.random.normal(0, noise_std, size=scan.shape)
        
        # Dropout
        dropout_prob = self.cfg.get('lidar_dropout_prob', 0.01)
        mask = np.random.random(size=scan.shape) < dropout_prob
        scan[mask] = 0.0
        
        return scan


class BacktrackBuffer:
    def __init__(self, timestep, seconds=2.0, max_backtracks=3):
        self.capacity = int(seconds / timestep)
        self.buffer = deque(maxlen=self.capacity)
        self.max_backtracks = max_backtracks
        self.current_backtracks = 0

    def add(self, pose):
        # pose is [x, y, theta]
        self.buffer.append(pose)

    def get_backtrack_pose(self):
        if not self.buffer:
            return None
        # Return the oldest pose in buffer (which is 'seconds' ago)
        return self.buffer[0]

    def reset_count(self):
        self.current_backtracks = 0

    def increment_count(self):
        self.current_backtracks += 1
        return self.current_backtracks


def compute_reward(delta_arc, lateral, collision, vel_x, steer, prev_steer, reward_cfg, tracker=None):
    """State-of-the-art reward function for F1TENTH RL."""
    if collision:
        return reward_cfg['collision_penalty']
    
    reward_type = reward_cfg.get('type', 'dense')
    
    if reward_type == 'sparse':
        # Sparse reward: milestones based on total track length
        if tracker is None or tracker.total_length == 0:
            return 0.0
        
        # Checkpoint reward: every 10% of the track
        prev_progress = (tracker._last_s - delta_arc) / tracker.total_length
        curr_progress = tracker._last_s / tracker.total_length
        
        # If we crossed a 0.1 boundary
        r = 0.0
        if int(curr_progress * 10) > int(prev_progress * 10):
            r += reward_cfg.get('checkpoint_reward', 1.0)
            
        # Lap bonus (if we wrapped around)
        if delta_arc < -tracker.total_length / 2.0:
            r += reward_cfg.get('lap_bonus', 50.0)
            
        return r
    else:
        # State-of-the-art Hybrid Dense Reward (inspired by f1tenth_rl_humble / Evans et al.)
        # 1. Forward progress with heavy penalty for reversing
        if delta_arc >= 0:
            r = reward_cfg.get('progress_weight', 10.0) * delta_arc
        else:
            # Heavy penalty for driving backwards to prevent "progress undoing"
            r = reward_cfg.get('progress_weight', 10.0) * delta_arc * 5.0
            
        # 2. Cross-track penalty
        r -= reward_cfg.get('deviation_weight', 0.05) * lateral
        
        # 3. Speed bonus
        r += reward_cfg.get('speed_weight', 0.1) * max(0.0, vel_x)
        
        # 4. Steering smoothness penalty
        if prev_steer is not None:
            steer_delta = abs(steer - prev_steer)
            r -= reward_cfg.get('steer_penalty', 0.2) * steer_delta
            
        # 5. Survival Reward: bonus for just staying alive to offset deviation penalties
        r += reward_cfg.get('survival_reward', 0.05)
            
        return float(r)


def compute_reward_fallback(vel_x, collision, steer, prev_steer, reward_cfg):
    """vel_x fallback for maps without a centerline."""
    if collision:
        return reward_cfg['collision_penalty']
    
    r = reward_cfg.get('progress_weight', 5.0) * max(float(vel_x), 0.0) * 0.01
    if prev_steer is not None:
        r -= reward_cfg.get('steer_penalty', 0.0) * abs(steer - prev_steer)
    return float(r)


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
        # Fallback: build from top-level maps list in config
        maps_lookup = {m['name']: m for m in cfg.get('maps', [])}
        fallback_names = ['berlin', 'stata_basement', 'vegas', 'skirk']
        training_maps = []
        for name in fallback_names:
            m = maps_lookup.get(name)
            if m:
                training_maps.append({
                    'name':       name,
                    'map_path':   m['map_path'],
                    'map_ext':    m.get('map_ext', '.png'),
                    'start_pose': m['start_pose'],
                })
        if not training_maps:
            raise RuntimeError(
                "No training_maps found in config. Add a training_maps section to config.yaml."
            )

    centerline_dir = rl_cfg.get('centerline_dir', '')

    pool = []
    for m in training_maps:
        path = m['map_path']
        ext  = m.get('map_ext', '.png')
        if not os.path.exists(path + ext):
            print(f"  [SKIP] map not found: {path}{ext}")
            continue
        print(f"  Loading map: {m['name']}")
        env = make_env(path, ext, timestep)
        entry = {'env': env, 'start_pose': m['start_pose'], 'name': m['name']}

        cl_path = os.path.join(centerline_dir, f"{m['name']}_centerline.csv") if centerline_dir else ''
        if cl_path and os.path.exists(cl_path):
            entry['tracker'] = CenterlineTracker(cl_path)
        else:
            print(f"    no centerline found — using vel_x fallback reward")

        pool.append(entry)

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

    randomizer = DomainRandomizer(rl_cfg)
    backtracker = BacktrackBuffer(
        timestep=cfg['gym']['timestep'],
        seconds=rl_cfg.get('backtrack', {}).get('seconds', 2.0),
        max_backtracks=rl_cfg.get('backtrack', {}).get('max_backtracks', 3)
    )

    rng = np.random.default_rng(seed=0)

    # Start on a random map
    current_map = map_pool[rng.integers(len(map_pool))]
    env = current_map['env']
    start_pose = current_map['start_pose']

    randomizer.randomize(env)
    # Satisfy newer gym's OrderEnforcing wrapper by manually setting its internal flag
    if hasattr(env, '_has_reset'):
        env._has_reset = True
    elif hasattr(env, 'env') and hasattr(env.env, '_has_reset'):
        env.env._has_reset = True

    # Use env.unwrapped.reset to actually set the pose in legacy f110_gym
    obs, _, done, _ = env.unwrapped.reset(np.array([[start_pose[0], start_pose[1], start_pose[2]]]))
    obs_buf.reset()
    scan_noise = randomizer.apply_sensor_noise(obs['scans'][0])
    obs_buf.update(scan_noise, float(obs['linear_vels_x'][0]), float(obs['ang_vels_z'][0]))
    tracker = current_map.get('tracker')
    if tracker is not None:
        tracker.reset(float(obs['poses_x'][0]), float(obs['poses_y'][0]))
    backtracker.reset_count()

    total_steps = 0
    update_count = 0
    episode_count = 0
    ep_rewards = deque(maxlen=20)
    ep_lengths = deque(maxlen=20)
    map_counts = {m['name']: 0 for m in map_pool}
    best_mean_reward = float('-inf')
    ep_reward = 0.0
    ep_len = 0
    prev_steer = 0.0

    print(f"\nStarting RL training for {rl_cfg['total_steps']:,} steps...")

    while total_steps < rl_cfg['total_steps']:
        for _ in range(rl_cfg['rollout_steps']):
            obs_seq = torch.from_numpy(obs_buf.get()).unsqueeze(0).to(device)
            with torch.no_grad():
                z = encoder(obs_seq)

            action_t, log_prob, value = ppo.act(z)
            steer, speed = clamp_action(float(action_t[0, 0].cpu()), float(action_t[0, 1].cpu()))
            
            # Apply action delay (DR)
            delayed_action = randomizer.apply_action_delay(np.array([steer, speed], dtype=np.float32))

            obs, _, done, _ = env.step(np.array([delayed_action]))
            collision = bool(obs['collisions'][0])
            vel_x = float(obs['linear_vels_x'][0])
            pose = [float(obs['poses_x'][0]), float(obs['poses_y'][0]), float(obs['poses_theta'][0])]
            backtracker.add(pose)

            if tracker is not None:
                delta_arc, lateral = tracker.step(pose[0], pose[1])
                reward = compute_reward(delta_arc, lateral, collision, vel_x, steer, prev_steer, reward_cfg, tracker=tracker)
            else:
                reward = compute_reward_fallback(vel_x, collision, steer, prev_steer, reward_cfg)

            prev_steer = steer

            ppo.store(z.squeeze(0), action_t.squeeze(0), log_prob.squeeze(0),
                      reward, done, value.squeeze(0))

            ep_reward += reward
            ep_len += 1
            total_steps += 1
            
            # Heartbeat logging every 500 steps
            if total_steps % 500 == 0:
                prog = (tracker._last_s / tracker.total_length) * 100 if tracker else 0
                print(f"  [Step {total_steps:8,}] Speed: {vel_x:4.2f} m/s | Progress: {prog:5.1f}% | Ep Reward: {ep_reward:7.2f}")

            # Force reset if episode exceeds max_steps or car is stuck
            timeout = ep_len >= cfg['data_collection'].get('max_steps_per_episode', 3000)

            if done or collision or timeout:
                # If collision, try backtracking first
                bt_enabled = rl_cfg.get('backtrack', {}).get('enabled', False)
                bt_pose = backtracker.get_backtrack_pose()
                if collision and bt_enabled and bt_pose and not timeout and backtracker.increment_count() <= backtracker.max_backtracks:
                    # Backtrack reset
                    # Satisfy newer gym's OrderEnforcing wrapper by manually setting its internal flag
                    if hasattr(env, '_has_reset'):
                        env._has_reset = True
                    elif hasattr(env, 'env') and hasattr(env.env, '_has_reset'):
                        env.env._has_reset = True
                    
                    # Use env.unwrapped.reset to actually set the pose in legacy f110_gym
                    obs, _, done, _ = env.unwrapped.reset(np.array([bt_pose]))
                    prev_steer = 0.0
                    
                    # CRITICAL: Reset and resync obs_buf so the model doesn't see "crash frames"
                    obs_buf.reset()
                    scan_noise = randomizer.apply_sensor_noise(obs['scans'][0])
                    # Fill buffer with the backtrack start frame to provide a stable initial context
                    for _ in range(mc['context_window']):
                        obs_buf.update(scan_noise, float(obs['linear_vels_x'][0]), float(obs['ang_vels_z'][0]))
                    
                    if tracker is not None:
                        # Resync tracker to backtrack pose
                        tracker.reset(float(obs['poses_x'][0]), float(obs['poses_y'][0]))
                else:
                    # Full episode reset
                    ep_rewards.append(ep_reward)
                    ep_lengths.append(ep_len)
                    episode_count += 1
                    map_counts[current_map['name']] += 1
                    ep_reward = 0.0
                    ep_len = 0
                    prev_steer = 0.0

                    # Full episode reset
                    current_map = map_pool[rng.integers(len(map_pool))]
                    env = current_map['env']
                    start_pose = current_map['start_pose']
                    randomizer.randomize(env)

                # Satisfy newer gym's OrderEnforcing wrapper by manually setting its internal flag
                if hasattr(env, '_has_reset'):
                    env._has_reset = True
                elif hasattr(env, 'env') and hasattr(env.env, '_has_reset'):
                    env.env._has_reset = True
                
                # Use env.unwrapped.reset to actually set the pose in legacy f110_gym
                obs, _, done, _ = env.unwrapped.reset(np.array([[start_pose[0], start_pose[1], start_pose[2]]]))

                    obs_buf.reset()
                    tracker = current_map.get('tracker')
                    if tracker is not None:
                        tracker.reset(float(obs['poses_x'][0]), float(obs['poses_y'][0]))
                    backtracker.reset_count()

            scan_noise = randomizer.apply_sensor_noise(obs['scans'][0])
            obs_buf.update(scan_noise, float(obs['linear_vels_x'][0]), float(obs['ang_vels_z'][0]))

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
