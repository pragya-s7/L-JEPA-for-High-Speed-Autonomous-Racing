#!/usr/bin/env python3
"""
L-JEPA driver node for the physical F1TENTH car.

Subscribes to /scan and /ego_racecar/odom (or /pf/pose/odom), maintains a
rolling context window of observations, runs the frozen encoder + trained
PPO actor, and publishes AckermannDriveStamped commands to /drive.

The node waits until the context window is fully populated before publishing
any drive commands so the encoder never acts on an all-zero buffer.

ROS parameters (all have defaults):
    ljepa_root          path to the ljepa project root
    encoder_checkpoint  path to pretrain/best.pt
    policy_checkpoint   path to rl/best.pt
    odom_topic          odometry topic (default: /ego_racecar/odom)
    speed_sign          +1.0 or -1.0 depending on car's drive convention
    steer_sign          +1.0 or -1.0 depending on car's drive convention
"""
import sys
import os

import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped


# Training constants — must match collect_data.py
TRAIN_NUM_BEAMS  = 1080
TRAIN_FOV        = 4.7
TRAIN_RANGE_MAX  = 10.0
SUBSAMPLE_STEP   = 5
LIDAR_DIM        = TRAIN_NUM_BEAMS // SUBSAMPLE_STEP  # 216


class LJEPADriver(Node):

    def __init__(self):
        super().__init__('ljepa_driver')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('ljepa_root',         '/home/pragya/ljepa')
        self.declare_parameter('encoder_checkpoint', '')
        self.declare_parameter('policy_checkpoint',  '')
        self.declare_parameter('odom_topic',         '/ego_racecar/odom')
        self.declare_parameter('speed_sign',         -1.0)  # forward is negative on this car
        self.declare_parameter('steer_sign',          1.0)
        self.declare_parameter('max_abs_steer',       0.10)
        self.declare_parameter('steer_smoothing',      0.35)
        self.declare_parameter('steer_bias',           0.0)
        self.declare_parameter('safety_distance',      0.60)
        self.declare_parameter('safety_speed',         0.80)
        self.declare_parameter('inference_deterministic', True)

        ljepa_root  = self.get_parameter('ljepa_root').value
        enc_path    = self.get_parameter('encoder_checkpoint').value
        pol_path    = self.get_parameter('policy_checkpoint').value
        odom_topic  = self.get_parameter('odom_topic').value
        self.speed_sign = float(self.get_parameter('speed_sign').value)
        self.steer_sign = float(self.get_parameter('steer_sign').value)
        self.max_abs_steer = float(self.get_parameter('max_abs_steer').value)
        self.steer_smoothing = float(self.get_parameter('steer_smoothing').value)
        self.steer_bias = float(self.get_parameter('steer_bias').value)
        self.safety_distance = float(self.get_parameter('safety_distance').value)
        self.safety_speed = float(self.get_parameter('safety_speed').value)
        self.inference_deterministic = bool(self.get_parameter('inference_deterministic').value)

        # Default checkpoint paths relative to ljepa_root
        if not enc_path:
            enc_path = os.path.join(ljepa_root, 'checkpoints', 'pretrain', 'best.pt')
        if not pol_path:
            pol_path = os.path.join(ljepa_root, 'checkpoints', 'rl', 'best.pt')

        # ── Load models ──────────────────────────────────────────────────────
        if ljepa_root not in sys.path:
            sys.path.insert(0, ljepa_root)

        import torch
        import yaml
        from models import ContextEncoder
        from rl.ppo import ActorCritic
        from controllers import FollowTheGap

        self._torch = torch
        self.safety_controller = FollowTheGap()
        cfg_path = os.path.join(ljepa_root, 'config.yaml')
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        mc = cfg['model']

        self.context_window = mc['context_window']
        self.obs_dim        = mc['obs_dim']
        self.device         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Encoder
        encoder = ContextEncoder(
            lidar_dim=mc['lidar_dim'],
            vel_dim=mc['vel_dim'],
            context_window=mc['context_window'],
            latent_dim=mc['latent_dim'],
            conv_channels=tuple(mc['encoder']['channels']),
            conv_kernel=mc['encoder']['kernel_size'],
            conv_stride=mc['encoder']['stride'],
        ).to(self.device)

        ckpt = torch.load(enc_path, map_location=self.device)
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
        self.encoder = encoder

        # Actor-critic
        ac = ActorCritic(
            latent_dim=mc['latent_dim'],
            action_dim=mc['action_dim'],
            hidden_dim=128,
        ).to(self.device)
        ac_ckpt = torch.load(pol_path, map_location=self.device)
        ac.load_state_dict(ac_ckpt['actor_critic_state_dict'])
        ac.eval()
        self.ac = ac

        self.get_logger().info(f"Encoder loaded from:      {enc_path}")
        self.get_logger().info(f"Actor-critic loaded from: {pol_path}")
        self.get_logger().info(f"Device: {self.device}  |  speed_sign: {self.speed_sign}")

        # ── Observation buffer ───────────────────────────────────────────────
        self.obs_buffer  = np.zeros((self.context_window, self.obs_dim), dtype=np.float32)
        self.scans_seen  = 0   # counts up to context_window before we start driving
        self.prev_steer = 0.0

        # ── Odometry state ───────────────────────────────────────────────────
        self.vel_x   = 0.0
        self.ang_vel = 0.0

        # ── ROS interfaces ───────────────────────────────────────────────────
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 10)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)

        self.get_logger().info('L-JEPA driver ready — waiting for scan data...')

    # ── Callbacks ────────────────────────────────────────────────────────────

    def odom_callback(self, msg: Odometry):
        self.vel_x   = msg.twist.twist.linear.x
        self.ang_vel = msg.twist.twist.angular.z

    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)

        # Resample to the exact 1080-beam / 4.7-rad grid used during training.
        # This handles cars whose LiDAR has a different beam count or FOV.
        train_angles = np.linspace(-TRAIN_FOV / 2.0, TRAIN_FOV / 2.0, TRAIN_NUM_BEAMS)
        actual_angles = (msg.angle_min +
                         np.arange(len(ranges)) * msg.angle_increment)

        # Only interpolate over the overlapping FOV
        in_fov = ((actual_angles >= train_angles[0]) &
                  (actual_angles <= train_angles[-1]))
        if in_fov.sum() < 2:
            self.get_logger().warn('Scan FOV does not overlap training FOV — skipping.')
            return

        resampled = np.interp(
            train_angles,
            actual_angles[in_fov],
            ranges[in_fov],
            left=0.0,
            right=0.0,
        )
        resampled = np.nan_to_num(resampled, nan=0.0, posinf=0.0, neginf=0.0)

        # Subsample and normalise
        scan_sub = resampled[::SUBSAMPLE_STEP].astype(np.float32)  # (216,)
        scan_sub = np.clip(scan_sub, 0.0, TRAIN_RANGE_MAX) / TRAIN_RANGE_MAX

        # Update rolling buffer
        obs = np.concatenate([scan_sub, [self.vel_x, self.ang_vel]], dtype=np.float32)
        self.obs_buffer[:-1] = self.obs_buffer[1:]
        self.obs_buffer[-1]  = obs
        self.scans_seen = min(self.scans_seen + 1, self.context_window)

        # Don't act until the buffer is fully populated
        if self.scans_seen < self.context_window:
            return

        # ── Inference ────────────────────────────────────────────────────────
        torch = self._torch
        obs_seq = torch.from_numpy(self.obs_buffer).unsqueeze(0).to(self.device)
        with torch.no_grad():
            z      = self.encoder(obs_seq)
            action, _, _ = self.ac.get_action(z, deterministic=self.inference_deterministic)

        raw_steer = float(np.clip(action[0, 0].cpu().numpy(), -0.4189, 0.4189))
        steer = (1.0 - self.steer_smoothing) * self.prev_steer + self.steer_smoothing * raw_steer
        steer = float(np.clip(steer, -self.max_abs_steer, self.max_abs_steer))
        self.prev_steer = steer
        speed = float(np.clip(action[0, 1].cpu().numpy(),  0.5,    5.0))

        front_window = resampled[TRAIN_NUM_BEAMS // 2 - 40: TRAIN_NUM_BEAMS // 2 + 40]
        min_front_dist = float(np.min(front_window)) if front_window.size else float(np.min(resampled))
        if min_front_dist < self.safety_distance:
            safe_steer, safe_speed = self.safety_controller.plan(
                resampled.astype(np.float32),
                angle_min=-TRAIN_FOV / 2.0,
                angle_increment=TRAIN_FOV / (TRAIN_NUM_BEAMS - 1),
                current_speed=float(self.vel_x),
                dt=0.01,
            )
            steer = float(np.clip(safe_steer, -self.max_abs_steer, self.max_abs_steer))
            speed = min(float(np.clip(safe_speed, 0.5, 5.0)), self.safety_speed)
            self.prev_steer = steer

        steer = float(np.clip(steer + self.steer_bias, -self.max_abs_steer, self.max_abs_steer))

        # ── Publish ──────────────────────────────────────────────────────────
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.steering_angle = self.steer_sign * steer
        drive_msg.drive.speed          = self.speed_sign * speed
        self.drive_pub.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LJEPADriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
