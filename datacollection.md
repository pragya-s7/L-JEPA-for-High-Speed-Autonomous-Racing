# L-JEPA Data Collection Guide

This guide gets you from a fresh clone to a working data collection setup.
Read the whole thing before starting — there are a few gotchas depending on your machine.

---

## What You're Collecting

The data collection script runs a **Follow-the-Gap** controller (no learning, no ROS2) in the
**f1tenth_gym** simulator and records LiDAR + velocity + action tuples.

Each episode is saved as `data/{map_name}/ep_NNNN.npz`.  
Each .npz contains:

| Key       | Shape          | Description                                     |
|-----------|----------------|-------------------------------------------------|
| `scans`   | `(T, 216)`     | Subsampled LiDAR (every 5th of 1080 beams), normalised to [0,1] |
| `vels`    | `(T, 2)`       | `[linear_vels_x, ang_vels_z]` in m/s and rad/s  |
| `actions` | `(T, 2)`       | `[steer (rad), speed (m/s)]` commanded to the gym |
| `poses`   | `(T, 3)`       | `[x, y, theta]` for diagnostics only             |

Aim: **≥ 20 episodes per map, across all 4 maps**.  
That gives ~200 k steps, which is enough to pretrain the encoder.

---

## Prerequisites

### 1. Python and pip

Python 3.8–3.11.  Check:
```bash
python3 --version
```

### 2. f1tenth_gym (old OpenAI gym version)

This is the simulator. Install it once:
```bash
# Clone the gym (if you don't already have it)
git clone https://github.com/f1tenth/f1tenth_gym.git ~/f1tenth_gym

# Install into the active Python environment
cd ~/f1tenth_gym
pip install -e .
```

> **Pinned dependencies** — the old gym requires `gym==0.19.0` and `numpy<=1.22`.
> If you have a newer numpy, downgrade first:
> ```bash
> pip install "numpy<1.23"
> ```

### 3. Clone this repo

```bash
git clone <repo-url> ~/ljepa
cd ~/ljepa
```

### 4. Additional Python packages for data collection

```bash
pip install pyyaml tqdm
```

The `tqdm` package is optional (prettier progress bars) but `pyyaml` is required.

---

## Telling the Script Where Your Gym Is

The script needs to find `f1tenth_gym/gym/f110_gym/`. It checks in this order:

1. `--gym-path` CLI argument  
2. `F1TENTH_GYM_PATH` environment variable  
3. `~/f1tenth_gym/gym` (default install location)

**Most people just need:**
```bash
export F1TENTH_GYM_PATH=~/f1tenth_gym/gym
```
Add that line to your `~/.bashrc` (or `~/.zshrc`) so you don't have to repeat it.

If your gym is somewhere else:
```bash
python3 collect_data.py --gym-path /path/to/f1tenth_gym/gym
```

---

## Verify Your Setup First

Always do a dry run before collecting real data:
```bash
cd ~/ljepa
python3 collect_data.py --dry-run --maps berlin --episodes 2
```

You should see output like:
```
Using gym at: /home/<you>/f1tenth_gym/gym
LiDAR: 1080 beams → subsample every 5 → 216 dims
...
Map: berlin  |  Episodes: 2  |  Start: [0.0, 0.0, 1.37]
  ep   1: 2430 steps  💥 collided  (9.2s)
  ep   2: 2443 steps  💥 collided  (8.8s)
[DRY RUN] No files will be written.
```

The "💥 collided" is expected — FTG runs clean for ~2000–2400 steps then clips a wall.
That data is all valid; we just capture the trajectory up to the collision.

---

## Running Data Collection

### Single map (quickest to start):
```bash
python3 collect_data.py --maps berlin --episodes 20
```

### All 4 maps (recommended):
```bash
python3 collect_data.py --episodes 20
```
This takes about **30–40 minutes** total.

### Resume interrupted runs:
The script is **safe to restart** — it counts existing episodes in the output
directory and continues from where it left off.
```bash
# Interrupted after 8 episodes. Just run the same command again:
python3 collect_data.py --maps berlin --episodes 20
# Will collect episodes 9–20 and skip the first 8.
```

### Collect to a custom directory (e.g. external drive):
```bash
python3 collect_data.py --output-dir /mnt/external/ljepa_data
```

---

## Maps Available

| Map             | Track style         | Expected episode length | Notes               |
|-----------------|---------------------|------------------------|---------------------|
| `berlin`        | Long sweeping track | ~2400 steps (~24 s)    | Best map to start   |
| `vegas`         | Wide, fast layout   | ~2000 steps            |                     |
| `stata_basement`| Tight indoor layout | ~2000 steps            |                     |
| `skirk`         | Technical corners   | ~750 steps             | FTG clips one corner|

All 4 maps are included by default. Use `--maps berlin vegas` etc. to run a subset.

---

## Recommended Collection Target

| Map             | Episodes | Approx steps |
|-----------------|----------|--------------|
| `berlin`        | 20       | ~48,000      |
| `vegas`         | 20       | ~40,000      |
| `stata_basement`| 20       | ~40,000      |
| `skirk`         | 20       | ~15,000      |
| **Total**       | **80**   | **~143,000** |

80 episodes takes about 40–50 min on a modern laptop. Each teammate can collect
a subset and we merge the directories.

---

## Sharing Your Data With the Team

### Option A: commit to git (small dataset)

The `data/` folder is part of the repo. After collecting:
```bash
cd ~/ljepa
git add data/
git commit -m "add berlin + vegas episodes from <yourname>"
git push
```

> Git will complain about large .npz files if they exceed GitHub's 100 MB limit.
> In that case use git-lfs:
> ```bash
> git lfs install
> git lfs track "data/**/*.npz"
> git add .gitattributes
> git commit -m "configure lfs for npz files"
> git add data/
> git commit -m "add data"
> ```

### Option B: shared drive / SCP

```bash
# Copy your data to a shared folder:
scp -r ~/ljepa/data/berlin  teammate@server:~/ljepa/data/

# Or zip and send:
tar czf berlin_episodes.tar.gz -C ~/ljepa/data berlin/
```

The other teammate unpacks into their `ljepa/data/` folder.
The pretrain script loads all `.npz` files it finds — multiple sources just work.

---

## Checking Your Data

After collection, run the built-in stats:
```bash
python3 collect_data.py --stats
```

Output:
```
Dataset at: /home/<you>/ljepa/data
  berlin              :   20 episodes,  48,200 steps
  stata_basement      :   20 episodes,  41,600 steps
  vegas               :   20 episodes,  42,400 steps
  skirk               :   20 episodes,  14,900 steps
  TOTAL               : 147,100 steps
```

For a quick sanity check of the arrays themselves:
```python
import numpy as np
d = np.load('data/berlin/ep_0000.npz')
print(d['scans'].shape)    # (T, 216)
print(d['scans'].min(), d['scans'].max())   # should be 0.0 to 1.0
print(d['vels'].shape)     # (T, 2)
print(d['actions'].shape)  # (T, 2)
```

---

## Headless / Remote Servers (No Monitor)

The gym imports `pyglet` at load time but never opens a window during
data collection (we never call `render()`). On most headless systems this
just works.

If you get an error like `pyglet.canvas.xlib.NoSuchDisplayException` or
similar display errors:

**Option 1** — Use `xvfb-run` (fastest fix, usually already installed):
```bash
xvfb-run -a python3 collect_data.py --maps berlin --episodes 20
```

**Option 2** — Install `xvfbwrapper` (auto-handled by the script):
```bash
pip install xvfbwrapper
python3 collect_data.py --maps berlin --episodes 20
# Script starts a virtual display automatically
```

**Option 3** — Use a Docker container with a virtual display.

---

## Common Problems

### `ModuleNotFoundError: No module named 'f110_gym'`
The gym isn't on your Python path. Either:
```bash
export F1TENTH_GYM_PATH=~/f1tenth_gym/gym
```
or
```bash
cd ~/f1tenth_gym && pip install -e .
```

---

### `ModuleNotFoundError: No module named 'gym'`
```bash
pip install "gym==0.19.0"
```

---

### `numpy` version errors at import or runtime
```bash
pip install "numpy<1.23"
```
Then reinstall the gym: `pip install -e ~/f1tenth_gym`

---

### `numba` JIT compilation warnings on first run
Normal — numba compiles the laser model on first call. Subsequent episodes are
much faster.

---

### Episodes ending immediately (0–30 steps)
The start pose may be spawning inside a wall for that map.
Symptoms: `collisions[0] = True` at step 0 or 1.

Try adjusting the start pose in `MAP_CATALOGUE` inside `collect_data.py`.
The berlin map is most reliable; try that first.

---

### Very slow collection (>30 s per episode)
Make sure you haven't accidentally enabled rendering. The script never calls
`env.render()`, so if it's slow check for background processes or a headless
display that is re-rendering.

---

## What Happens to This Data

The `.npz` files are loaded by `training/dataset.py`, which builds JEPA training
tuples `(context_obs, future_obs, actions)`. Each file is a single episode —
episode boundaries are never crossed when sampling, which is critical for
correct JEPA training.

After merging all team members' data, run pretraining:
```bash
python3 training/pretrain.py
```

---

## Directory Layout After Collection

```
ljepa/
└── data/
    ├── berlin/
    │   ├── ep_0000.npz   (~2400 steps, ~180 KB compressed)
    │   ├── ep_0001.npz
    │   └── ...
    ├── vegas/
    │   └── ...
    ├── stata_basement/
    │   └── ...
    └── skirk/
        └── ...
```

---

*Questions? Ping the team chat or open a GitHub issue.*
