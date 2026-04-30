#!/usr/bin/env python3
"""
Diagnose a single episode: plot trajectory, steering, speed, and a LiDAR snapshot
at the start to see what the car was actually "seeing" when it started spinning.

Usage:
    python3 diagnose_episode.py                                        # auto: newest ep in berlin__wall_follow
    python3 diagnose_episode.py /path/to/ep_0000.npz                   # specific file
"""
import sys
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless-safe
import matplotlib.pyplot as plt


def find_ep(arg):
    if arg and os.path.isfile(arg):
        return arg
    candidates = sorted(glob.glob(
        os.path.expanduser('~/ljepa/data/berlin__wall_follow/ep_*.npz')
    ))
    if not candidates:
        print("No episode files found. Run collect_data.py first.")
        sys.exit(1)
    return candidates[-1]


def main():
    ep_path = find_ep(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"Loading: {ep_path}")
    d = np.load(ep_path)
    poses   = d['poses']     # (T, 3) x, y, theta
    vels    = d['vels']      # (T, 2) vx, wz
    actions = d['actions']   # (T, 2) steer, speed
    scans   = d['scans']     # (T, 216) normalized to [0,1]

    T = len(poses)
    dt = 0.01
    t = np.arange(T) * dt

    # LiDAR constants (must match collect_data.py)
    NUM_BEAMS = 1080
    SUBSAMPLE = 5
    FOV = 4.7
    ANG_MIN = -FOV / 2.0
    ANG_INC = FOV / (NUM_BEAMS - 1)
    sub_angles = ANG_MIN + np.arange(0, NUM_BEAMS, SUBSAMPLE) * ANG_INC  # (216,)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- (0,0) Trajectory in world frame ----
    ax = axes[0, 0]
    ax.plot(poses[:, 0], poses[:, 1], '-', lw=1.2, color='tab:blue', label='path')
    ax.plot(poses[0, 0], poses[0, 1], 'go', ms=10, label='start')
    ax.plot(poses[-1, 0], poses[-1, 1], 'rs', ms=10, label='end')
    # Arrows showing heading every N steps
    N = max(1, T // 30)
    ax.quiver(poses[::N, 0], poses[::N, 1],
              np.cos(poses[::N, 2]), np.sin(poses[::N, 2]),
              angles='xy', scale_units='xy', scale=5,
              color='gray', alpha=0.5, width=0.003)
    ax.set_aspect('equal')
    ax.set_title(f'Trajectory ({T} steps, {T*dt:.1f}s)')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.grid(alpha=0.3); ax.legend(loc='best', fontsize=9)

    # ---- (0,1) Commanded steering over time ----
    ax = axes[0, 1]
    ax.plot(t, np.rad2deg(actions[:, 0]), color='tab:red', lw=1.0)
    ax.axhline(24, color='k', ls='--', lw=0.7, alpha=0.5)
    ax.axhline(-24, color='k', ls='--', lw=0.7, alpha=0.5)
    ax.axhline(20, color='orange', ls=':', lw=0.7, alpha=0.5, label='±20° (slow trigger)')
    ax.axhline(-20, color='orange', ls=':', lw=0.7, alpha=0.5)
    ax.axhline(10, color='green', ls=':', lw=0.7, alpha=0.5, label='±10° (fast trigger)')
    ax.axhline(-10, color='green', ls=':', lw=0.7, alpha=0.5)
    ax.set_title('Commanded steering angle')
    ax.set_xlabel('time [s]'); ax.set_ylabel('steer [deg]')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # ---- (1,0) Commanded speed + actual vx ----
    ax = axes[1, 0]
    ax.plot(t, actions[:, 1], color='tab:red', lw=1.2, label='commanded speed')
    ax.plot(t, vels[:, 0], color='tab:blue', lw=1.0, alpha=0.7, label='actual vx')
    ax.set_title('Speed (commanded vs actual)')
    ax.set_xlabel('time [s]'); ax.set_ylabel('speed [m/s]')
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # ---- (1,1) First-frame LiDAR in polar (bird's eye from car) ----
    ax = axes[1, 1]
    # scans are normalized to [0,1] in collect_data.py (divided by RANGE_MAX=10)
    r0 = scans[0] * 10.0
    # car frame: x forward, y left. LiDAR angle 0 = forward.
    xs = r0 * np.cos(sub_angles)
    ys = r0 * np.sin(sub_angles)
    ax.plot(xs, ys, '.', ms=2, color='tab:purple', alpha=0.6)
    ax.plot(0, 0, 'k^', ms=12, label='car')
    # Highlight the two beams wall_follow actually uses: 90° (left) and 45° (forward-left)
    for beam_deg, color, name in [(90, 'red', 'b (90°, left)'),
                                  (45, 'orange', 'a (45°, fwd-left)')]:
        a = np.deg2rad(beam_deg)
        idx = int(round((a - ANG_MIN) / (ANG_INC * SUBSAMPLE)))
        if 0 <= idx < len(r0):
            ax.plot([0, r0[idx]*np.cos(a)], [0, r0[idx]*np.sin(a)],
                    color=color, lw=1.5, label=f'{name} = {r0[idx]:.2f} m')
    ax.set_aspect('equal')
    ax.set_title('First-frame LiDAR (car frame)')
    ax.set_xlabel('x forward [m]'); ax.set_ylabel('y left [m]')
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='best')
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)

    plt.tight_layout()
    out = os.path.splitext(ep_path)[0] + '_diagnose.png'
    plt.savefig(out, dpi=110)
    print(f"Saved: {out}")

    # ---- Text summary ----
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Steps: {T}   ({T*dt:.2f} s)")
    print(f"  xy travelled:   dx={poses[:,0].max()-poses[:,0].min():.2f} m, "
          f"dy={poses[:,1].max()-poses[:,1].min():.2f} m")
    path_len = np.sum(np.hypot(np.diff(poses[:,0]), np.diff(poses[:,1])))
    print(f"  Path length:    {path_len:.2f} m")
    print(f"  Steer:          mean|δ|={np.mean(np.abs(np.rad2deg(actions[:,0]))):.1f}°, "
          f"max|δ|={np.max(np.abs(np.rad2deg(actions[:,0]))):.1f}°")
    saturated = np.mean(np.abs(actions[:,0]) > 0.4) * 100
    print(f"  Steer saturated (|δ|>23°): {saturated:.1f}% of steps")
    print(f"  Speed:          mean={actions[:,1].mean():.2f}, "
          f"max={actions[:,1].max():.2f}, min={actions[:,1].min():.2f}")
    print(f"  vx (actual):    mean={vels[:,0].mean():.2f}, max={vels[:,0].max():.2f}")
    print(f"  yaw rate |wz|:  mean={np.abs(vels[:,1]).mean():.2f} rad/s")

    # First-frame beams wall_follow uses
    print()
    print("  First-frame beams used by wall_follow:")
    for beam_deg, label in [(90, 'b (straight left)'), (45, 'a (45° fwd-left)')]:
        a = np.deg2rad(beam_deg)
        idx_full = int(round((a - ANG_MIN) / ANG_INC))
        # scans are subsampled; get the subsampled index instead
        idx_sub = int(round((a - ANG_MIN) / (ANG_INC * SUBSAMPLE)))
        if 0 <= idx_sub < len(scans[0]):
            print(f"    {label:22s}  = {scans[0, idx_sub]*10:.3f} m")
    print(f"  Target distance to left wall (desired_dist): 1.02 m")


if __name__ == '__main__':
    main()