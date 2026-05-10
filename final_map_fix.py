import os
import numpy as np
import yaml
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import label
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree

MAP_DIR = '/home/pragya/f1tenth_gym/gym/f110_gym/envs/maps'
OUT_DIR = 'data/centerlines'
os.makedirs(OUT_DIR, exist_ok=True)

# Final High-Quality Map Pool
MAP_INFO = {
    'berlin': [0.0, 0.0, 1.37],
    'Austin_map': [0.0, 0.0, 0.0],
    'vegas': [0.0, 0.0, 0.0],
    'levine': [0.0, 0.0, 0.0],
    'BrandsHatch_map': [0.0, 0.0, 0.0],
    'Budapest_map': [0.0, 0.0, 0.0],
    'Catalunya_map': [0.0, 0.0, 0.0],
    'Hockenheim_map': [0.0, 0.0, 0.0],
    'IMS_map': [0.0, 0.0, 0.0],
    'Melbourne_map': [0.0, 0.0, 0.0],
    'MexicoCity_map': [0.0, 0.0, 0.0],
    'Montreal_map': [0.0, 0.0, 0.0],
    'Monza_map': [0.0, 0.0, 0.0],
    'MoscowRaceway_map': [0.0, 0.0, 0.0],
    'Nuerburgring_map': [0.0, 0.0, 0.0],
    'Oschersleben_map': [0.0, 0.0, 0.0],
    'Sakhir_map': [0.0, 0.0, 0.0],
    'SaoPaulo_map': [0.0, 0.0, 0.0],
    'Sepang_map': [0.0, 0.0, 0.0],
    'Shanghai_map': [0.0, 0.0, 0.0],
    'Silverstone_map': [0.0, 0.0, 0.0],
    'Sochi_map': [0.0, 0.0, 0.0],
    'Spa_map': [0.0, 0.0, 0.0],
    'Spielberg_map': [0.0, 0.0, 0.0],
    'YasMarina_map': [0.0, 0.0, 0.0],
    'Zandvoort_map': [0.0, 0.0, 0.0],
}

def get_isolated_track(img, start_row, start_col):
    h, w = img.shape
    r, c = int(start_row), int(start_col)
    seed_val = int(img[r, c])
    mask = cv2.inRange(img, seed_val - 5, seed_val + 5)
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(mask, flood_mask, (c, r), 128)
    return mask == 128

def trace_and_order(skel, start_row, start_col, start_theta):
    rows, cols = np.where(skel)
    pts = np.column_stack([rows, cols])
    if len(pts) < 10: return np.zeros((0, 2))
    tree = cKDTree(pts)
    _, start_idx = tree.query([start_row, start_col])
    ordered = [pts[start_idx]]
    visited = {start_idx}
    dr, dc = -np.sin(start_theta), np.cos(start_theta)
    cdir = np.array([dr, dc])
    for _ in range(len(pts)):
        indices = [i for i in tree.query_ball_point(ordered[-1], r=3.0) if i not in visited]
        if not indices: break
        best_idx = indices[0]
        if len(indices) > 1:
            best_s = -1e9
            for idx in indices:
                v = pts[idx] - ordered[-1]
                v = v / (np.linalg.norm(v) + 1e-6)
                s = np.dot(v, cdir)
                if s > best_s: best_s, best_idx = s, idx
        new_v = pts[best_idx] - ordered[-1]
        norm = np.linalg.norm(new_v)
        if norm > 0: cdir = new_v / norm
        ordered.append(pts[best_idx])
        visited.add(best_idx)
    return np.array(ordered)

def process(name):
    ypath = os.path.join(MAP_DIR, f"{name}.yaml")
    if not os.path.exists(ypath): return
    with open(ypath) as f: meta = yaml.safe_load(f)
    res, origin = meta['resolution'], meta['origin']
    img = np.array(Image.open(os.path.join(MAP_DIR, meta['image'])).convert('L'))
    h, w = img.shape
    pose = MAP_INFO[name]
    scol, srow = (pose[0] - origin[0])/res, (h - 1) - (pose[1] - origin[1])/res
    track_mask = get_isolated_track(img, srow, scol)
    skel = skeletonize(track_mask)
    opts = trace_and_order(skel, srow, scol, pose[2])
    if len(opts) < 10: return
    rows, cols = opts[:, 0], opts[:, 1]
    wx, wy = cols * res + origin[0], (h - 1 - rows) * res + origin[1]
    dx, dy = np.gradient(wx), np.gradient(wy)
    psi = np.arctan2(dy, dx)
    s = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(wx)**2 + np.diff(wy)**2))])
    np.savetxt(f"{OUT_DIR}/{name}_centerline.csv", np.column_stack([s, wx, wy, psi, np.zeros_like(wx), np.ones_like(wx)*2.0]),
               delimiter=';', header='s_m;x_m;y_m;psi_rad;kappa_m;v_mps', comments='', fmt='%.6f')
    fimg = np.zeros_like(img)
    thick = int(3.0 / res)
    for i in range(len(cols)-1):
        cv2.line(fimg, (int(cols[i]), int(rows[i])), (int(cols[i+1]), int(rows[i+1])), 255, thick)
    cv2.line(fimg, (int(cols[-1]), int(rows[-1])), (int(cols[0]), int(rows[0])), 255, thick)
    nimg_name = meta['image'].replace('.', '_fixed.')
    Image.fromarray(fimg).save(os.path.join(MAP_DIR, nimg_name))
    meta['image'] = nimg_name
    with open(os.path.join(MAP_DIR, f"{name}_fixed.yaml"), 'w') as f: yaml.dump(meta, f)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray', origin='upper', extent=[origin[0], origin[0]+w*res, origin[1], origin[1]+h*res])
    plt.plot(wx, wy, 'r-', lw=1.5)
    plt.title(name)
    plt.savefig(f"verify_{name}.png")
    plt.close()

if __name__ == "__main__":
    for m in MAP_INFO.keys(): process(m)
