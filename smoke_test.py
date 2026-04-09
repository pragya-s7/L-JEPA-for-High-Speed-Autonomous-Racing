#!/usr/bin/env python3
"""
Smoke test: verifies the model architecture and a single data collection step.

Does NOT require a full dataset or GPU. Run this first to confirm that
imports, model shapes, and the gym integration all work.

    python smoke_test.py
"""
import sys
import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GYM_PATH = '/home/pragya/f1tenth_gym'
sys.path.insert(0, os.path.join(GYM_PATH, 'gym'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import yaml


def test_models():
    print("--- Model smoke test ---")
    from models import LJEPA

    with open(os.path.join(PROJECT_ROOT, 'config.yaml')) as f:
        cfg = yaml.safe_load(f)
    mc = cfg['model']

    model = LJEPA(
        lidar_dim=mc['lidar_dim'],
        vel_dim=mc['vel_dim'],
        context_window=mc['context_window'],
        latent_dim=mc['latent_dim'],
        action_dim=mc['action_dim'],
        prediction_horizon=mc['prediction_horizon'] if 'prediction_horizon' in mc else 5,
        conv_channels=tuple(mc['encoder']['channels']),
        conv_kernel=mc['encoder']['kernel_size'],
        conv_stride=mc['encoder']['stride'],
        predictor_hidden=tuple(mc['predictor']['hidden_dims']),
    )

    B = 4
    h = mc['context_window']
    obs_dim = mc['obs_dim']
    k = 5
    action_dim = mc['action_dim']

    ctx = torch.randn(B, h, obs_dim)
    fut = torch.randn(B, h, obs_dim)
    acts = torch.randn(B, k, action_dim)

    loss, z_ctx, z_pred = model(ctx, fut, acts)
    print(f"  context latent shape: {z_ctx.shape}  (expected: [{B}, {mc['latent_dim']}])")
    print(f"  predicted latent shape: {z_pred.shape}")
    print(f"  pretraining loss: {loss.item():.4f}")

    # EMA update
    model.update_ema()
    print("  EMA update: OK")

    # Encoder-only forward (RL path)
    z = model.encode(ctx)
    print(f"  encode() shape: {z.shape}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total:,}  |  Trainable: {trainable:,}")
    print("  PASS\n")


def test_ppo():
    print("--- PPO smoke test ---")
    from rl.ppo import PPO, ActorCritic
    import torch

    latent_dim = 64
    action_dim = 2
    B = 8

    ppo = PPO(latent_dim=latent_dim, action_dim=action_dim,
              rollout_steps=B, epochs_per_update=1, minibatch_size=B)

    z = torch.randn(1, latent_dim)
    action, log_prob, value = ppo.act(z)
    print(f"  action shape: {action.shape}, log_prob: {log_prob.item():.4f}, value: {value.item():.4f}")

    # Fill buffer
    for _ in range(B):
        z_ = torch.randn(1, latent_dim)
        a, lp, v = ppo.act(z_)
        ppo.store(z_.squeeze(0), a.squeeze(0), lp.squeeze(0), 0.1, False, v.squeeze(0))

    last_v = torch.tensor([[0.0]])
    stats = ppo.update(last_v)
    print(f"  update stats: {stats}")
    print("  PASS\n")


def test_controller():
    print("--- FollowTheGap smoke test ---")
    from controllers import FollowTheGap

    ctrl = FollowTheGap()
    NUM_BEAMS = 1080
    FOV = 4.7
    ANGLE_MIN = -FOV / 2.0
    ANGLE_INCREMENT = FOV / (NUM_BEAMS - 1)

    # Fake scan: open corridor ahead, walls on sides
    scan = np.ones(NUM_BEAMS, dtype=np.float32) * 2.0
    scan[NUM_BEAMS // 2 - 50: NUM_BEAMS // 2 + 50] = 5.0  # gap ahead

    steer, speed = ctrl.plan(scan, ANGLE_MIN, ANGLE_INCREMENT, current_speed=2.0)
    print(f"  steer: {steer:.4f} rad  speed: {speed:.4f} m/s")
    assert abs(steer) <= 0.42, "Steering exceeds limit"
    assert 0 < speed <= 5.0, "Speed out of range"
    print("  PASS\n")


def test_gym_import():
    print("--- Gym import smoke test ---")
    try:
        import gym
        from f110_gym.envs.base_classes import Integrator
        print(f"  gym version: {gym.__version__}")
        print("  PASS\n")
    except ImportError as e:
        print(f"  WARN: gym import failed: {e}")
        print("  (install with: cd /home/pragya/f1tenth_gym && pip install -e .)\n")


def test_single_env_step():
    print("--- Single env step smoke test ---")
    try:
        import gym
        from f110_gym.envs.base_classes import Integrator
        from controllers import FollowTheGap

        env = gym.make(
            'f110_gym:f110-v0',
            map='/home/pragya/f1tenth_gym/gym/f110_gym/envs/maps/berlin',
            map_ext='.png',
            num_agents=1,
            timestep=0.01,
            integrator=Integrator.RK4,
        )
        obs, _, done, _ = env.reset(np.array([[0.0, 0.0, 1.37]]))
        scan = obs['scans'][0]
        print(f"  scan shape: {scan.shape}")

        ctrl = FollowTheGap()
        FOV = 4.7
        steer, speed = ctrl.plan(scan, -FOV/2, FOV/1079, current_speed=0.0)
        print(f"  FTG action: steer={steer:.4f}, speed={speed:.4f}")

        obs, _, done, _ = env.step(np.array([[steer, speed]]))
        print(f"  after step: x={obs['poses_x'][0]:.3f}, y={obs['poses_y'][0]:.3f}, "
              f"vx={obs['linear_vels_x'][0]:.3f}")
        env.close()
        print("  PASS\n")
    except Exception as e:
        print(f"  WARN: env step failed: {e}\n")


if __name__ == '__main__':
    test_models()
    test_ppo()
    test_controller()
    test_gym_import()
    test_single_env_step()
    print("All smoke tests done.")
