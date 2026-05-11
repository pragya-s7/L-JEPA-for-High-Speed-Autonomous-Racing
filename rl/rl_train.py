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

_CFG_BOOT = os.path.join(PROJECT_ROOT, 'config.yaml')
_GYM_ROOT = ''
if os.path.isfile(_CFG_BOOT):
    with open(_CFG_BOOT) as _bf:
        _boot = yaml.safe_load(_bf)
    _GYM_ROOT = (_boot.get('gym') or {}).get('gym_path', '') or os.environ.get('F1TENTH_GYM', '')
if _GYM_ROOT and not os.path.isabs(_GYM_ROOT):
    _GYM_ROOT = os.path.join(PROJECT_ROOT, _GYM_ROOT)
GYM_PATH = _GYM_ROOT or '/home/yim/f1tenth_gym'
_sys_gym_pkg = os.path.join(GYM_PATH, 'gym')
if _sys_gym_pkg not in sys.path:
    sys.path.insert(0, _sys_gym_pkg)

import gym
from f110_gym.envs.base_classes import Integrator
from models import ContextEncoder
from rl.ppo import PPO, raw_to_env_action, STEER_LIMIT, SPEED_LOW, SPEED_HIGH


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
    Tracks arc-length progress and lateral deviation along a centerline CSV
    (columns: s_m; x_m; y_m).

    Uses sequential waypoint association (local window around last index) instead
    of pure nearest-neighbour on (x,y). That reduces ambiguous jumps on concave /
    self-overlapping layouts where global NN picks the wrong branch.

    For closed loops (endpoints close) arc-length delta wraps at the cut.
    """

    _CLOSED_GAP_M = 5.0
    _SEQ_WINDOW = 80
    _FALLBACK_DIST_M = 3.0

    def __init__(self, csv_path):
        data = np.loadtxt(csv_path, delimiter=';', skiprows=1)
        self.s = data[:, 0].astype(np.float32)
        self.x = data[:, 1].astype(np.float32)
        self.y = data[:, 2].astype(np.float32)
        # CSV layout: s_m;x_m;y_m;psi_rad;kappa_m;v_mps. We load psi_rad so
        # sample_random_pose() can spawn the car aligned with the track
        # tangent. Falls back to computed tangent if absent.
        if data.shape[1] >= 4:
            self.psi = data[:, 3].astype(np.float32)
        else:
            self.psi = None
        self.n = len(self.s)
        self.total_length = float(self.s[-1])
        self._tree = cKDTree(np.column_stack([self.x, self.y]))
        self._last_s = None
        self._last_idx = None

        loop_gap = float(np.hypot(self.x[-1] - self.x[0], self.y[-1] - self.y[0]))
        self.is_closed = loop_gap < self._CLOSED_GAP_M
        print(f"    centerline: {self.n} wp  length={self.total_length:.1f}m  "
              f"{'closed' if self.is_closed else 'open (no wrap-around)'}  "
              f"(sequential tracking)")

    def sample_random_pose(self, rng=None, margin_frac=0.02):
        """
        Return a random (x, y, theta) on the centerline. Used to break out of
        local-minima training where the policy always crashes at the same
        on-track location and never gets to learn the rest of the track.

        margin_frac: avoid the very first/last few % of waypoints on open
            tracks so a backward-facing spawn doesn't immediately crash.
        """
        if rng is None:
            rng = np.random.default_rng()
        if self.n <= 2:
            return float(self.x[0]), float(self.y[0]), 0.0

        if self.is_closed:
            idx = int(rng.integers(0, self.n))
        else:
            lo = max(1, int(margin_frac * self.n))
            hi = max(lo + 1, self.n - lo)
            idx = int(rng.integers(lo, hi))

        x = float(self.x[idx])
        y = float(self.y[idx])

        if self.psi is not None:
            theta = float(self.psi[idx])
        else:
            # Fall back to a discrete tangent from neighbouring waypoints
            if self.is_closed:
                ip = (idx - 1) % self.n
                iq = (idx + 1) % self.n
            else:
                ip = max(0, idx - 1)
                iq = min(self.n - 1, idx + 1)
            tx = float(self.x[iq] - self.x[ip])
            ty = float(self.y[iq] - self.y[ip])
            theta = float(np.arctan2(ty, tx))
        return x, y, theta

    def reset(self, x, y):
        _, idx = self._tree.query([x, y])
        self._last_idx = int(idx)
        self._last_s = float(self.s[self._last_idx])
        # Tracks total forward progress since last reset; lap_complete is only
        # allowed to fire after the agent has actually driven most of the way
        # around the loop. Without this guard a sequential-association glitch
        # (idx jumping from near the end back to near the start) would falsely
        # award the lap_bonus.
        self._progress_since_reset = 0.0

    def _nearest_idx_sequential(self, x, y):
        """Prefer neighbors of last idx so progress stays on one polyline branch."""
        if self._last_idx is None:
            _, idx = self._tree.query([x, y])
            return int(idx)

        best_d = float('inf')
        best_i = self._last_idx
        W = self._SEQ_WINDOW
        for di in range(-W, W + 1):
            i = self._last_idx + di
            if self.is_closed:
                i = i % self.n
            elif i < 0 or i >= self.n:
                continue
            dx = float(x) - float(self.x[i])
            dy = float(y) - float(self.y[i])
            d = float(np.hypot(dx, dy))
            if d < best_d:
                best_d = d
                best_i = i

        if best_d > self._FALLBACK_DIST_M:
            _, j = self._tree.query([x, y])
            best_i = int(j)
        return best_i

    def _heading_error_rad(self, idx, theta):
        """Angle between vehicle heading and centerline tangent at waypoint idx."""
        if self.is_closed:
            ip = (idx - 1) % self.n
            iq = (idx + 1) % self.n
        else:
            ip = max(0, idx - 1)
            iq = min(self.n - 1, idx + 1)
        tx = float(self.x[iq] - self.x[ip])
        ty = float(self.y[iq] - self.y[ip])
        if tx == 0.0 and ty == 0.0:
            return 0.0
        tangent = np.arctan2(ty, tx)
        err = float(theta) - tangent
        err = (err + np.pi) % (2 * np.pi) - np.pi
        return err

    def step(self, x, y, theta):
        """
        Returns:
            delta_arc_m, lateral_m, lap_complete,
            heading_err_rad, prev_arc_s, curr_arc_s
        """
        prev_arc_s = self._last_s
        idx = self._nearest_idx_sequential(x, y)
        self._last_idx = idx

        lateral = float(np.hypot(float(x) - float(self.x[idx]), float(y) - float(self.y[idx])))
        curr_s = float(self.s[idx])
        heading_err = self._heading_error_rad(idx, theta)

        delta_s = curr_s - prev_arc_s
        wrap_detected = False

        if self.is_closed:
            half = self.total_length / 2.0
            if delta_s > half:
                delta_s -= self.total_length
            elif delta_s < -half:
                delta_s += self.total_length
                wrap_detected = True

        delta_s = float(np.clip(delta_s, -1.0, 1.0))
        self._last_s = curr_s
        if delta_s > 0:
            self._progress_since_reset += delta_s

        # Require ~80% of a full lap of accumulated forward progress to call it
        # a lap. This rejects spurious wraps caused by waypoint association
        # jumps near self-overlapping geometry.
        lap_complete = bool(
            wrap_detected
            and self.is_closed
            and self._progress_since_reset >= 0.8 * self.total_length
        )
        if lap_complete:
            self._progress_since_reset = 0.0  # rearm for next lap

        return delta_s, lateral, lap_complete, heading_err, prev_arc_s, curr_s


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


def compute_reward(
    delta_arc,
    lateral,
    collision,
    vel_x,
    steer,
    prev_steer,
    lap_complete,
    reward_cfg,
    tracker=None,
    heading_err_rad=0.0,
    prev_arc_s=None,
    curr_arc_s=None,
):
    """
    Hybrid dense + sparse lap bonus. Tuned to avoid rewarding:
      - idling / endless survival (no positive survival bonus by default),
      - high speed without forward track progress (speed gated),
      - crashing early still yields large negative return via collision_penalty + little accumulation.

    Progress uses centerline arc coordinate with sequential association (see CenterlineTracker).
    """
    if collision:
        return float(reward_cfg['collision_penalty'])

    reward_type = reward_cfg.get('type', 'dense')
    nbins = int(reward_cfg.get('progress_bins', 10))

    if reward_type == 'sparse':
        if tracker is None or tracker.total_length <= 0:
            return 0.0
        if prev_arc_s is None or curr_arc_s is None:
            return 0.0
        prev_progress = float(prev_arc_s) / tracker.total_length
        curr_progress = float(curr_arc_s) / tracker.total_length
        r = 0.0
        if int(curr_progress * nbins + 1e-9) > int(prev_progress * nbins + 1e-9):
            r += float(reward_cfg.get('checkpoint_reward', 1.0))
        if lap_complete:
            r += float(reward_cfg.get('lap_bonus', 1000.0))
        return float(r)

    pw = float(reward_cfg.get('progress_weight', 10.0))
    if delta_arc >= 0:
        r = pw * delta_arc
    else:
        r = pw * float(delta_arc) * float(reward_cfg.get('wrong_way_multiplier', 5.0))

    r -= float(reward_cfg.get('deviation_weight', 0.1)) * float(lateral)

    min_df = float(reward_cfg.get('speed_requires_forward_delta_m', 0.03))
    if delta_arc >= min_df:
        r += float(reward_cfg.get('speed_weight', 0.0)) * max(0.0, vel_x)

    hw = float(reward_cfg.get('heading_weight', 0.0))
    if hw > 0.0:
        r -= hw * min(abs(float(heading_err_rad)), float(np.pi))

    if prev_steer is not None:
        r -= float(reward_cfg.get('steer_penalty', 0.2)) * abs(steer - prev_steer)

    r += float(reward_cfg.get('survival_reward', 0.0))
    r += float(reward_cfg.get('time_penalty_per_step', 0.0))

    dcb = float(reward_cfg.get('dense_checkpoint_bonus', 0.0))
    if dcb > 0.0 and tracker is not None and prev_arc_s is not None and curr_arc_s is not None:
        tl = tracker.total_length
        if int(float(curr_arc_s) / tl * nbins + 1e-9) > int(float(prev_arc_s) / tl * nbins + 1e-9):
            r += dcb

    if lap_complete:
        r += float(reward_cfg.get('lap_bonus', 1000.0))

    return float(r)


def compute_reward_fallback(vel_x, collision, steer, prev_steer, reward_cfg):
    """Weak shaping when no centerline CSV exists — prefer generating centerlines instead."""
    if collision:
        return float(reward_cfg['collision_penalty'])

    r = float(reward_cfg.get('fallback_vel_scale', 0.02)) * max(float(vel_x), 0.0)
    if prev_steer is not None:
        r -= float(reward_cfg.get('steer_penalty', 0.0)) * abs(steer - prev_steer)
    r += float(reward_cfg.get('survival_reward', 0.0))
    r += float(reward_cfg.get('time_penalty_per_step', 0.0))
    return float(r)


def raw_action_to_env(raw_action_np):
    """Map a 2-vector raw action sample to (env_steer, env_speed)."""
    env_action = raw_to_env_action(raw_action_np)
    return float(env_action[0]), float(env_action[1])


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
        weight = float(m.get('weight', 1.0))
        if weight <= 0:
            print(f"  [SKIP] map disabled (weight={weight}): {m['name']}")
            continue
        if not os.path.exists(path + ext):
            print(f"  [SKIP] map not found: {path}{ext}")
            continue
        print(f"  Loading map: {m['name']}  (weight={weight:g})")
        env = make_env(path, ext, timestep)
        entry = {
            'env': env,
            'start_pose': m['start_pose'],
            'name': m['name'],
            'weight': weight,
        }

        cl_path = os.path.join(centerline_dir, f"{m['name']}_centerline.csv") if centerline_dir else ''
        if cl_path and os.path.exists(cl_path):
            entry['tracker'] = CenterlineTracker(cl_path)
        else:
            print(f"    no centerline found — using vel_x fallback reward")

        entry['randomize_start'] = bool(m.get('randomize_start', False))
        if entry['randomize_start'] and 'tracker' not in entry:
            print(f"    [warn] randomize_start=True but no centerline for {m['name']} — "
                  f"will fall back to fixed start_pose")
        pool.append(entry)

    if not pool:
        raise RuntimeError("No valid training maps found. Check paths in config.yaml.")

    weights = np.array([m['weight'] for m in pool], dtype=np.float64)
    if not np.all(weights > 0):
        raise RuntimeError(f"Map weights must be > 0. Got: {weights}")
    probs = weights / weights.sum()
    pct = ", ".join(f"{m['name']}={100*p:.0f}%" for m, p in zip(pool, probs))
    print(f"Map sampling weights: {pct}")
    return pool, probs


def train(cfg_path, encoder_path_override=None, checkpoint_dir_override=None,
          total_steps_override=None, resume_path=None, maps_override=None):
    cfg = load_config(cfg_path)
    mc = cfg['model']
    rl_cfg = cfg['rl']
    reward_cfg = rl_cfg['reward']

    if checkpoint_dir_override:
        rl_cfg['checkpoint_dir'] = checkpoint_dir_override
        os.makedirs(rl_cfg['checkpoint_dir'], exist_ok=True)
    if total_steps_override is not None:
        rl_cfg['total_steps'] = int(total_steps_override)

    # Allow --maps to whitelist a subset of training_maps from config (used for
    # curriculum learning: e.g. --maps vegas trains on vegas only).
    if maps_override:
        whitelist = {m.lower() for m in maps_override}
        original = rl_cfg.get('training_maps', []) or []
        filtered = [m for m in original if m['name'].lower() in whitelist]
        if not filtered:
            raise RuntimeError(
                f"--maps={maps_override} matched none of "
                f"{[m['name'] for m in original]}"
            )
        rl_cfg['training_maps'] = filtered
        print(f"Training maps overridden: {[m['name'] for m in filtered]}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    encoder_path = encoder_path_override or rl_cfg['encoder_checkpoint']
    print(f"Encoder checkpoint: {encoder_path}")
    print(f"RL save directory:    {rl_cfg['checkpoint_dir']}")
    print(f"Total env steps:      {rl_cfg['total_steps']:,}")
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

    if resume_path:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        ppo.ac.load_state_dict(ckpt['actor_critic_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            try:
                ppo.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except (ValueError, KeyError) as e:
                print(f"  [warn] could not restore optimizer state: {e}")
        # `optimizer.load_state_dict` overwrites lr with the lr that was used
        # when the checkpoint was saved. For fine-tuning we want the lr from
        # the *current* config to take effect (typically smaller). Force it.
        new_lr = float(rl_cfg['lr'])
        for g in ppo.optimizer.param_groups:
            g['lr'] = new_lr
        prev_update = int(ckpt.get('update', 0))
        prev_steps  = int(ckpt.get('total_steps', 0))
        prev_reward = float(ckpt.get('mean_reward', float('-inf')))
        print(f"Resumed from {resume_path} "
              f"(prev update {prev_update}, prev steps {prev_steps:,}, "
              f"prev mean_reward {prev_reward:.2f})")
        print(f"  Forcing lr = {new_lr} on all param groups for fine-tuning.")

    print("\nBuilding map pool...")
    map_pool, map_probs = build_map_pool(cfg)
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

    def pick_start_pose(map_entry):
        """
        Return [x, y, theta] for a fresh reset on `map_entry`.
        If randomize_start is True and a centerline tracker is loaded, sample
        a random waypoint on the centerline so the policy gets to practice the
        whole track instead of crashing at the same spot every episode.
        """
        tr = map_entry.get('tracker')
        if map_entry.get('randomize_start', False) and tr is not None:
            x, y, th = tr.sample_random_pose(rng)
            return [x, y, th]
        return list(map_entry['start_pose'])

    # Start on a random map (weighted by training_maps[*].weight)
    current_map = map_pool[rng.choice(len(map_pool), p=map_probs)]
    env = current_map['env']
    start_pose = pick_start_pose(current_map)

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
            raw_np = action_t[0].detach().cpu().numpy()
            steer, speed = raw_action_to_env(raw_np)
            
            # Apply action delay (DR)
            delayed_action = randomizer.apply_action_delay(np.array([steer, speed], dtype=np.float32))

            obs, _, done, _ = env.step(np.array([delayed_action]))
            collision = bool(obs['collisions'][0])
            vel_x = float(obs['linear_vels_x'][0])
            pose = [float(obs['poses_x'][0]), float(obs['poses_y'][0]), float(obs['poses_theta'][0])]
            
            # CRITICAL: Detect physics integrator explosion (NaN/Inf)
            if not np.all(np.isfinite(pose)):
                print(f"  [WARNING] Physics exploded at step {total_steps:,}! Forcing full reset.")
                collision = True
                done = True
                # Set a dummy finite pose to prevent tracker crash during this final step
                pose = [0.0, 0.0, 0.0]
            
            backtracker.add(pose)

            if tracker is not None:
                delta_arc, lateral, lap_complete, heading_err, prev_arc_s, curr_arc_s = tracker.step(
                    pose[0], pose[1], pose[2]
                )
                reward = compute_reward(
                    delta_arc,
                    lateral,
                    collision,
                    vel_x,
                    steer,
                    prev_steer,
                    lap_complete,
                    reward_cfg,
                    tracker=tracker,
                    heading_err_rad=heading_err,
                    prev_arc_s=prev_arc_s,
                    curr_arc_s=curr_arc_s,
                )
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
                    if hasattr(env, '_has_reset'): env._has_reset = True
                    elif hasattr(env, 'env') and hasattr(env.env, '_has_reset'): env.env._has_reset = True
                    
                    obs, _, done, _ = env.unwrapped.reset(np.array([bt_pose]))
                    prev_steer = 0.0
                    
                    # Resync temporal buffer
                    obs_buf.reset()
                    scan_noise = randomizer.apply_sensor_noise(obs['scans'][0])
                    for _ in range(mc['context_window']):
                        obs_buf.update(scan_noise, float(obs['linear_vels_x'][0]), float(obs['ang_vels_z'][0]))
                    
                    if tracker is not None:
                        tracker.reset(float(obs['poses_x'][0]), float(obs['poses_y'][0]))
                else:
                    # Full episode reset
                    ep_rewards.append(ep_reward)
                    ep_lengths.append(ep_len)
                    episode_count += 1
                    map_counts[current_map['name']] += 1
                    
                    current_map = map_pool[rng.choice(len(map_pool), p=map_probs)]
                    env = current_map['env']
                    start_pose = pick_start_pose(current_map)
                    randomizer.randomize(env)

                    if hasattr(env, '_has_reset'): env._has_reset = True
                    elif hasattr(env, 'env') and hasattr(env.env, '_has_reset'): env.env._has_reset = True
                    
                    obs, _, done, _ = env.unwrapped.reset(np.array([[start_pose[0], start_pose[1], start_pose[2]]]))
                    obs_buf.reset()
                    tracker = current_map.get('tracker')
                    if tracker is not None:
                        tracker.reset(float(obs['poses_x'][0]), float(obs['poses_y'][0]))
                    
                    ep_reward = 0.0
                    ep_len = 0
                    prev_steer = 0.0
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
    parser.add_argument(
        '--checkpoint-dir',
        default=None,
        help='Override rl.checkpoint_dir so parallel encoder experiments do not overwrite each other.',
    )
    parser.add_argument(
        '--total-steps',
        type=int,
        default=None,
        help='Override rl.total_steps (default: value from config.yaml).',
    )
    parser.add_argument(
        '--resume',
        default=None,
        help='Path to a previous best.pt / update_xxxxx.pt to warm-start the actor-critic.',
    )
    parser.add_argument(
        '--maps',
        nargs='+',
        default=None,
        help='Whitelist subset of rl.training_maps by name (curriculum learning).',
    )
    args = parser.parse_args()
    train(
        args.config,
        encoder_path_override=args.encoder,
        checkpoint_dir_override=args.checkpoint_dir,
        total_steps_override=args.total_steps,
        resume_path=args.resume,
        maps_override=args.maps,
    )


if __name__ == '__main__':
    main()
