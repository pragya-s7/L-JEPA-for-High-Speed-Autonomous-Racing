# L-JEPA Data Collection Guide (Overhauled)

The data collection pipeline has been completely overhauled to solve representation collapse. We now use a diverse set of maps, randomized spawning, and an ensemble of controllers.

---

## Key Changes
1. **20+ Maps:** We now use 28 different racetracks (Austin, Monza, Silverstone, etc.) instead of just 4.
2. **Centerline-Anchored Spawning:** The car no longer starts at `[0,0,0]`. It spawns at a random point on the track centerline with injected noise ($\pm 0.3m$ lateral, $\pm 11^\circ$ heading).
3. **Controller Ensemble:** Each episode randomly selects a controller to generate diverse driving data:
   - **Follow-The-Gap (FTG):** Reactive obstacle avoidance.
   - **Pure Pursuit:** Optimal line following (using map centerlines).
   - **Wall Follower (Left/Right):** Learns boundary features.
   - **Constant:** Noisy straight driving to capture drift/physics.
4. **Horizon $k=30$:** We now predict 0.3s into the future (up from 0.05s) to force the model to learn deep momentum and curvature features.

---

## Running Collection

### Quick Dry Run
```bash
python3 collect_data.py --episodes 1 --dry-run
```

### Full Production Run
To collect 20 episodes per map across all 28 maps (~560 episodes total):
```bash
python3 collect_data.py --episodes 20
```
This will yield approximately **1.1 million steps** of high-entropy data, which is more than enough for a robust foundation.

---

## Directory Layout
```
ljepa/
├── data/
│   ├── centerlines/      # Generated .csv files for every map
│   └── {map_name}/       # .npz episodes
```

---

*The foundation is now rock-solid. Proceed with full collection to fix the "shaky" encoder behavior.*
