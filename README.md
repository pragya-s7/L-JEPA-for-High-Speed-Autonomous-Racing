# L-JEPA for High-Speed Autonomous Racing

### Team Members
 - Pragya Singh | GitHub: pragya-s7 | LinkedIn: https://www.linkedin.com/in/pragyasingh7/
 - Qingyun Ying | GitHub: nozomi209
 - Manasa Dendukuri | GitHub: maniden | LinkedIn: https://www.linkedin.com/in/manasadendukuri/
 - Qikai Shen | GitHub: qikais-ac

### Demo Video
https://youtube.com/shorts/AudgXsTIlX4
https://youtube.com/shorts/KV5V8AaO_w0
### Brief Overview

**Problem and Motivation**
Autonomous racing on the F1/10th platform (NVIDIA Jetson Orin Nano, 20 ms budget at 50 Hz) needs a policy that is both fast and environment-aware. End-to-end RL on raw LiDAR works but learns no reusable structure. Generative world models like DreamerV3 learn rich internal models but spend most of their compute reconstructing sensor data the policy doesn't need.

**L-JEPA** asks: can you get the benefits of a world model without reconstruction overhead? Instead of decoding observations, the model learns by predicting future latent representations conditioned on past context and future actions. The pretrained encoder is then frozen and a compact PPO policy is trained on its 64-dimensional output.

**Technical Approach**

Training has two stages.

Stage 1: Self-Supervised Pretraining

Offline data is collected from Follow-the-Gap and Pure Pursuit controllers across multiple simulator tracks (Levine 2nd floor excluded as a transfer target). Each datapoint is `(past observations, future actions, future observation at t+30)`.

Three components:
- **Context encoder** `E_θ` — 3-layer 1D conv `[32, 64, 128]`, kernel 5, stride 2, followed by a temporal MLP projecting 4 consecutive 218-dim observations (216 subsampled LiDAR beams + 2 velocity) to a 64-dim latent `z_t`.
- **EMA target encoder** `E_θ̄` — identical frozen copy whose weights slowly track `E_θ` via exponential moving average (`τ=0.996`). Prevents representational collapse by providing stable regression targets.
- **Action-conditioned predictor** `P_φ` — 2-layer MLP `[256, 256]` that takes `z_t` and the next 30 actions and predicts `ẑ_{t+30}`.

Loss is MSE between predictor output and stop-gradient target:
```
L = || ẑ_{t+k} - sg(E_θ̄(o_{t+k})) ||²
```

Pretrained for 100 epochs, Adam lr=3e-4, batch 256.

Stage 2: PPO on Frozen Latents

Encoder is frozen. Compact PPO actor-critic (2-layer MLPs, hidden dim 64) trained in F1TENTH Gym on `z_t`. Reward combines centerline progress, lateral deviation, speed, steering smoothness, a continuous wall-proximity penalty (active below 0.58 m), collision penalty (−80), and lap bonus (+100). Spawn positions sampled uniformly from the track centerline during training.

Phase 2 adds domain randomization: friction, mass, tire stiffness, LiDAR noise (σ=0.028 m), 2% beam dropout, and up to 4-step action delay.

PPO: clip ε=0.2, entropy 0.02, GAE λ=0.95, 10 epochs/update, rollout 2048, lr=3e-4.

**Results**

Pretraining: Loss converges smoothly to a stable non-zero plateau over 100 epochs. Embedding variance stays high throughout; thus, no collapse.

Vegas (curriculum warmup)
Single-map policy achieved consistent lap completion across seeds, confirming that a frozen 64-dim encoder is sufficient for reliable racing in a familiar environment.

Berlin (curriculum ceiling)
Adding Berlin to the curriculum produced a hard ceiling at **57.7% lap progress** in all configurations. The policy drove straight into the same corner without braking every time. Halving the mean speed from 4.3 to 2.3 m/s made no difference.

| Configuration | Mean speed (m/s) | Max progress | Crash rate |
|---|---|---|---|
| Standard | 4.32 | 57.4% | 100% |
| + random resets | 4.16 | 59.4% | 100% |
| Speed cap 2.5 m/s | 2.35 | 59.3% | 100% |

Root cause being encoder gap, reward sparsity at that corner, or training distribution mismatch remains unresolved. We pivoted to single-map training on Levine.

**Levine 2nd Floor (8000-episode benchmark)**

| Policy | Steps | Lap rate (all conditions) | Lap rate (centerline spawn) | Multi-lap episodes |
|---|---|---|---|---|
| Phase 1 | 2.5M | 10.1% | ~20% | ~1.8% |
| Phase 2 (DR) | 5.0M | 6.7% | ~13% | 0% |

Phase 1 under best conditions (centerline spawn, no DR, deterministic): **37.6% mean lap rate, 50% peak**. Among episodes with at least one lap completion, mean forward progress is ~38.6 m for Phase 1 and ~17.9 m for Phase 2.

The near-100% collision rate is expected as the only termination conditions are collision and step timeout, so the car drives until it hits something regardless of performance. What matters is how far it gets first.

![RL training curves](checkpoints/rl_levine_2nd_phase2/loss_curve.png)
*Phase 2 training (2.5M → 5M steps). Value loss adjusts to the new reward structure; policy loss stays near zero.*

**Hardware Deployment**

**Phase 1 model:** immediately turned hard right into the nearest wall on every run. Exact cause unknown: lack of DR, observation mismatch at the start pose, and actuator differences are all plausible.

**Phase 2 (DR) model:** made meaningful forward progress on open corridor sections. Failed at doorframes bc their LiDAR geometry (narrow gap, parallel edges) resembles a branching corridor to the policy. After hitting a doorframe and being manually reset, the car immediately crashed rather than resuming; likely the context buffer retained stale observations from the doorframe encounter.

DR directionally improved hardware behavior. Partial success on open straights is a baseline to build from.

**Inference Latency (Jetson Orin Nano)**

| Component | Mean (ms) | Std (ms) |
|---|---|---|
| Encoder | [X.X] | [X.X] |
| PPO actor | [X.X] | [X.X] |
| **Total** | **[X.X]** | **[X.X]** |
| Budget (50 Hz) | 20.0 | — |

**Challenges**

**Reward hacking.** An early reward with heavy centerline progress caused degenerate behavior. Patching with penalties produced new degenerate behaviors (wall-crashing, standing still) rather than fixing the original. We scrapped the checkpoint and rebuilt the reward from scratch.

**Misleading metrics.** Shaped episode reward was a poor quality signal, slow surviving policies scored well without making real progress. Switching to arc-length forward progress revealed that several apparent improvements were actually regressions.

**Spawn bias.** Fixed-start training meant the policy never saw much of the track. Random centerline spawns fixed coverage and improved lap rates.

**Sim-to-real:** Physical doorframe LiDAR returns are absent from the simulator and outside the pretraining distribution. No physics-parameter DR can generate them. Real-world scans in the pretraining corpus are the fix. Additionally, the turning behavior remains ununderstood (occurs sometimes in DR-based models). 

**Future Work**

- **Allow zero-speed actions.** The 0.5 mph minimum speed meant the policy never learned to stop, first thing to fix before any future hardware experiments.
- **Include real LiDAR scans in pretraining.** Even a small unlabeled set would dramatically improve coverage of physical sensor geometry.
- **Probe the encoder at failure locations.** Decode geometric quantities from `z_t` at the Berlin/Levine ceiling to determine whether the bottleneck is the encoder or the reward.
- **Encoder fine-tuning.** A short joint fine-tune after PPO converges is the natural next experiment for pushing past the performance ceiling.
- **Flush context buffer on hardware reset.** One-line fix that may resolve the post-doorframe crash behavior.

---

## Build and Run

### Prerequisites

```bash
pip install torch numpy gymnasium pyyaml pandas

git clone https://github.com/f1tenth/f1tenth_gym
cd f1tenth_gym && pip install -e .

git clone https://github.com/pragya-s7/L-JEPA-for-High-Speed-Autonomous-Racing
cd L-JEPA-for-High-Speed-Autonomous-Racing && pip install -e .
```

### 1. Collect pretraining data
```bash
python training/collect_data.py --config config.yaml
```

### 2. Pretrain the encoder
```bash
python training/pretrain.py --config config.yaml
# Best checkpoint → checkpoints/pretrain/best.pt
```

### 3. Train the RL policy
```bash
# Phase 1
python training/rl_train.py --config config.yaml

# Phase 2 (resume from Phase 1 checkpoint)
python training/rl_train.py --config config.yaml \
  --resume ./checkpoints/<phase1>/rl_policy_last.pt
```

### 4. Evaluate
```bash
# Quick eval with trajectory visualization
python eval_rl_policy.py \
  --policy ./checkpoints/rl_levine_2nd_phase2/last.pt \
  --encoder ./checkpoints/pretrain/best.pt \
  --spawn centerline --stochastic \
  --trajectory-png ./eval_rollout.png

# Full benchmark grid (reproduces paper numbers)
python paper_rl_benchmark.py --config config.yaml
```

### 5. Deploy on hardware
```bash
ros2 run ljepa_node ljepa_node \
  --policy ./checkpoints/rl_levine_2nd_phase2/last.pt \
  --encoder ./checkpoints/pretrain/best.pt
```

Subscribes to `/scan` and `/odom`, publishes to `/drive`.

### Key config fields
```yaml
model:
  lidar_subsample: 5       # 1080 → 216 beams
  latent_dim: 64
  context_window: 4
  prediction_horizon: 30
pretrain:
  epochs: 100
  ema_decay: 0.996
rl:
  total_steps: 5_000_000
  spawn_curriculum:
    enabled: true
  domain_randomization:
    enabled: true          # Phase 2 only
```

---

## Repo Structure

```
├── config.yaml
├── training/
│   ├── pretrain.py
│   ├── rl_train.py
│   ├── dataset.py
│   └── models.py
├── eval_rl_policy.py
├── paper_rl_benchmark.py
├── ljepa_node/            # ROS2 deployment node
├── data/centerlines/
└── checkpoints/
    ├── pretrain/
    └── rl_levine_2nd_phase2/
```
