#!/usr/bin/env python3
"""
Extract track centerlines from f1tenth map PNGs via skeletonization.
Run this once before RL training to produce per-map CSV files.

Usage:
    python rl/make_centerlines.py
    python rl/make_centerlines.py --maps berlin vegas --erosion 3 --out data/centerlines
"""
import os
import sys
import argparse
import time

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import binary_erosion, label
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPS_DIR = '/home/pragya/f1tenth_gym/gym/f110_gym/envs/maps'
DEFAULT_OUT = os.path.join(PROJECT_ROOT, 'data', 'centerlines')

MAP_START_POSES = {
    'berlin':         [0.0, 0.0, 1.37],
    'stata_basement': [0.0, 0.0, 0.0],
    'vegas':          [0.0, 0.0, 0.0],
    'skirk':          [0.0, 0.0, 0.0],
    'levine':         [0.0, 0.0, 0.0],
}

MAP_EXTENSIONS = {
    'levine': '.pgm',
}

def discover_maps():
    maps = []
    for f in os.listdir(MAPS_DIR):
        if f.endswith('.yaml'):
            name = f[:-5]
            maps.append(name)
    return sorted(maps)

def load_map(name):
    """Return (binary_free_array, resolution_m_per_px, origin_xy, map_ext)."""
    base = os.path.join(MAPS_DIR, name)
    ext = MAP_EXTENSIONS.get(name)
    if not ext:
        if os.path.exists(base + '.png'):
            ext = '.png'
        elif os.path.exists(base + '.pgm'):
            ext = '.pgm'
        else:
            raise FileNotFoundError(f"No image file found for map {name}")
    
    img = np.array(Image.open(base + ext).convert('L'))
    with open(base + '.yaml') as f:
        meta = yaml.safe_load(f)
    # Strictly white (255) is usually the track. Gray (205) is background/unknown.
    return img > 250, float(meta['resolution']), meta['origin'], ext


def world_to_pixel(wx, wy, height, res, origin):
    col = (wx - origin[0]) / res
    row = height - 1.0 - (wy - origin[1]) / res
    return col, row


def pixel_to_world(rows, cols, height, res, origin):
    x = np.asarray(cols) * res + origin[0]
    y = (height - 1 - np.asarray(rows)) * res + origin[1]
    return x, y


def largest_component(skel):
    """Keep only the largest 8-connected component of the skeleton."""
    struct = np.ones((3, 3), dtype=bool)
    labeled, n = label(skel, structure=struct)
    if n <= 1:
        return skel
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == sizes.argmax()


def prune_branches(skel):
    """
    Iteratively remove degree-1 (dead-end) pixels until only the main loop remains.
    Uses a convolution-based degree count so each iteration removes all endpoints at once.
    """
    from scipy.ndimage import convolve
    kernel = np.ones((3, 3), dtype=np.int32)
    kernel[1, 1] = 0
    skel = skel.astype(bool).copy()
    for _ in range(500):
        degree = convolve(skel.astype(np.int32), kernel, mode='constant', cval=0)
        endpoints = skel & (degree <= 1)
        if not endpoints.any():
            break
        skel &= ~endpoints
    return skel


def order_skeleton(skel, seed_col, seed_row, init_dcol, init_drow):
    """
    Walk skeleton pixels into an ordered sequence starting near (seed_col, seed_row)
    and initially heading in direction (init_dcol, init_drow) in image col/row space.

    Uses direction-guided greedy traversal: at each step the neighbor requiring
    the smallest turn is preferred. Works well for smooth closed-loop tracks.
    """
    rows_px, cols_px = np.where(skel)
    pts = np.column_stack([cols_px, rows_px]).astype(np.float32)  # shape (N, 2): [col, row]

    tree = cKDTree(pts)
    _, seed_idx = tree.query([seed_col, seed_row])

    # 8-connectivity adjacency
    pixel_to_idx = {(int(r), int(c)): i for i, (c, r) in enumerate(pts)}
    adj = [[] for _ in range(len(pts))]
    for i, (cx, ry) in enumerate(pts):
        cx, ry = int(cx), int(ry)
        for dc in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if dc == 0 and dr == 0:
                    continue
                key = (ry + dr, cx + dc)
                if key in pixel_to_idx:
                    adj[i].append(pixel_to_idx[key])

    direction = np.array([init_dcol, init_drow], dtype=np.float64)
    norm = np.linalg.norm(direction)
    direction = direction / norm if norm > 1e-8 else np.array([1.0, 0.0])

    visited = np.zeros(len(pts), dtype=bool)
    path = [seed_idx]
    visited[seed_idx] = True

    while True:
        curr = path[-1]
        unvisited = [n for n in adj[curr] if not visited[n]]
        if not unvisited:
            break

        curr_xy = pts[curr]
        best, best_score = None, -np.inf
        for nb in unvisited:
            vec = pts[nb] - curr_xy
            n = np.linalg.norm(vec)
            if n < 1e-8:
                continue
            score = float(np.dot(vec / n, direction))
            if score > best_score:
                best_score = score
                best = nb

        if best is None:
            break

        new_dir = pts[best] - pts[curr]
        n = np.linalg.norm(new_dir)
        if n > 1e-8:
            direction = new_dir / n

        path.append(best)
        visited[best] = True

    path = np.array(path)
    ordered = pts[path]
    return ordered[:, 1].astype(int), ordered[:, 0].astype(int)  # rows, cols


def extract_centerline_from_data(name, data_dir, downsample=10, return_thresh=2.0):
    """
    Build a centerline from Phase 1 trajectory data (poses from .npz files).
    Takes the first episode, walks until the car returns near the start (one lap),
    then subsamples and computes arc length.
    """
    import glob
    files = sorted(glob.glob(os.path.join(data_dir, name, '*.npz')))
    if not files:
        raise RuntimeError(f"No .npz files found in {data_dir}/{name}/")

    poses = np.load(files[0])['poses']  # (T, 3): x, y, theta
    x, y = poses[:, 0], poses[:, 1]

    # Find one-lap endpoint: first return to within return_thresh m of start
    # (after the car has driven at least 20% of the episode)
    min_start = max(50, int(0.15 * len(x)))
    lap_end = len(x)
    for i in range(min_start, len(x)):
        if np.hypot(x[i] - x[0], y[i] - y[0]) < return_thresh:
            lap_end = i
            break

    x_lap = x[:lap_end:downsample]
    y_lap = y[:lap_end:downsample]

    dx = np.diff(x_lap)
    dy = np.diff(y_lap)
    s = np.concatenate([[0.0], np.cumsum(np.sqrt(dx ** 2 + dy ** 2))])
    return x_lap, y_lap, s


def extract_centerline(name, erosion_px=3, downsample=5):
    free, res, origin, ext = load_map(name)
    h = free.shape[0]

    struct = np.ones((2 * erosion_px + 1, 2 * erosion_px + 1), dtype=bool)
    eroded = binary_erosion(free, struct)
    n_eroded = int(eroded.sum())
    if n_eroded < 50:
        raise RuntimeError(
            f"Erosion with radius={erosion_px} collapsed free space ({n_eroded} px). "
            "Try reducing --erosion."
        )
    print(f"  free={int(free.sum()):,}px  after_erosion={n_eroded:,}px", end='  ')

    skel = skeletonize(eroded)
    skel = largest_component(skel)
    skel = prune_branches(skel)
    n_skel = int(skel.sum())
    print(f"skeleton={n_skel:,}px", end='  ')
    if n_skel < 10:
        raise RuntimeError("Skeleton is empty after pruning. Try reducing --erosion.")

    if name in MAP_START_POSES:
        sx, sy, stheta = MAP_START_POSES[name]
        seed_col, seed_row = world_to_pixel(sx, sy, h, res, origin)
        # Direction in (col, row) image space: col ~ world-x, row ~ -world-y
        init_dcol = np.cos(stheta)
        init_drow = -np.sin(stheta)
    else:
        # Pick any pixel from the skeleton as a seed
        rows_px, cols_px = np.where(skel)
        seed_row, seed_col = rows_px[0], cols_px[0]
        init_dcol, init_drow = 1.0, 0.0

    rows_px, cols_px = order_skeleton(skel, seed_col, seed_row, init_dcol, init_drow)

    rows_px = rows_px[::downsample]
    cols_px = cols_px[::downsample]

    x, y = pixel_to_world(rows_px, cols_px, h, res, origin)

    dx = np.gradient(x)
    dy = np.gradient(y)
    s = np.concatenate([[0.0], np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))])
    # Re-interpolate s to match x, y length if needed, but gradient is easier
    ds = np.sqrt(dx**2 + dy**2)
    s = np.cumsum(ds)
    s -= s[0]

    psi = np.arctan2(dy, dx)
    
    # Dummy curvature and speed
    kappa = np.zeros_like(s)
    v = np.ones_like(s) * 2.0 # Default 2m/s

    return s, x, y, psi, kappa, v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maps', nargs='+', default=discover_maps())
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--data-dir', default=os.path.join(PROJECT_ROOT, 'data'),
                        help='Phase 1 data directory. If a map subfolder exists here, '
                             'trajectory data is used instead of skeletonization.')
    parser.add_argument('--erosion', type=int, default=3,
                        help='Wall erosion radius in pixels (skeleton mode only)')
    parser.add_argument('--downsample-skel', type=int, default=5,
                        help='Keep every Nth skeleton pixel (skeleton mode)')
    parser.add_argument('--downsample-data', type=int, default=10,
                        help='Keep every Nth pose sample (data mode)')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    for name in args.maps:
        try:
            free, res, origin, ext = load_map(name)
        except Exception as e:
            print(f'[SKIP] {name}: {e}')
            continue

        print(f'\n{name}:')
        t0 = time.time()

        data_subdir = os.path.join(args.data_dir, name)
        use_data = os.path.isdir(data_subdir) and any(
            f.endswith('.npz') for f in os.listdir(data_subdir)
        )

        try:
            # Always try skeleton first.
            print(f'  [skeleton mode]', end=' ')
            s, x, y, psi, kappa, v = extract_centerline(name, args.erosion, args.downsample_skel)
            loop_gap = np.hypot(x[-1] - x[0], y[-1] - y[0])
            if loop_gap > 10.0 and use_data:
                print(f'\n  skeleton loop_gap={loop_gap:.1f}m — falling back to trajectory data')
                # For now fallback still uses old 3-col logic, but let's fix it too
                x, y, _s = extract_centerline_from_data(name, args.data_dir, args.downsample_data)
                dx, dy = np.gradient(x), np.gradient(y)
                s, psi, kappa, v = _s, np.arctan2(dy, dx), np.zeros_like(_s), np.ones_like(_s)*2.0
        except Exception as e:
            if use_data:
                print(f'\n  skeleton failed ({e}), trying trajectory data...')
                try:
                    x, y, s = extract_centerline_from_data(name, args.data_dir, args.downsample_data)
                    dx, dy = np.gradient(x), np.gradient(y)
                    psi, kappa, v = np.arctan2(dy, dx), np.zeros_like(s), np.ones_like(s)*2.0
                except Exception as e2:
                    print(f'  ERROR: {e2}')
                    continue
            else:
                print(f'\n  ERROR: {e}')
                continue

        out_path = os.path.join(args.out, f'{name}_centerline.csv')
        np.savetxt(out_path, np.column_stack([s, x, y, psi, kappa, v]),
                   delimiter=';', header='s_m;x_m;y_m;psi_rad;kappa_m;v_mps', comments='', fmt='%.6f')

        loop_gap = np.hypot(x[-1] - x[0], y[-1] - y[0])
        print(f'  waypoints={len(x)}  track_length={s[-1]:.1f}m  '
              f'loop_gap={loop_gap:.2f}m  ({time.time()-t0:.1f}s)')
        print(f'  -> {out_path}')


if __name__ == '__main__':
    main()



if __name__ == '__main__':
    main()
