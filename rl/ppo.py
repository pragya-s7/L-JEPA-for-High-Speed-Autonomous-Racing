"""
Minimal PPO implementation for L-JEPA RL training.

The actor and critic operate on latent vectors z produced by the frozen
ContextEncoder, not on raw LiDAR scans. This keeps both networks small and
inference fast.

Policy output convention (IMPORTANT):
  The actor produces a raw action in unbounded R^2 sampled from
  Normal(mu, std) where std = exp(log_std). PPO computes log-probabilities,
  ratios and advantages in this *raw* space — i.e. the rollout buffer always
  stores raw actions, not env-space actions. Before being sent to the
  simulator, raw actions are transformed by `raw_to_env_action`:

      env_steer  = clamp(raw[0] * STEER_LIMIT,             -STEER_LIMIT, +STEER_LIMIT)
      env_speed  = clamp(raw[1] * SPEED_HALF + SPEED_MID,    SPEED_LOW,    SPEED_HIGH)

  At init the policy outputs raw ≈ 0 so env_speed ≈ SPEED_MID (safe mid-range)
  rather than getting saturated at the lower clip — which was the cause of the
  "agent crawls at 0.5 m/s forever" failure mode of the earlier training run.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


LOG_STD_MIN = -3.0
LOG_STD_MAX = 0.5

# Action ranges (must match the f1tenth_gym kinematic limits).
STEER_LIMIT = 0.4189            # rad
SPEED_LOW   = 0.5               # m/s
# Stage 9 curriculum: SPEED_HIGH lowered from 4.5 → 2.5 m/s. The unrestricted
# 4.5 m/s policy can't physically brake fast enough for berlin's tight
# corners, so PPO converged on "drive at 4.3 m/s into the same wall every
# episode". By capping the action space we force the policy to operate at
# speeds where the corners are physically navigable; once it learns to
# complete a lap at 2.5 m/s the cap can be raised in a follow-up stage.
SPEED_HIGH  = 2.5               # m/s — Stage 9 curriculum cap (was 4.5)
SPEED_MID   = 0.5 * (SPEED_LOW + SPEED_HIGH)        # 1.5 m/s
SPEED_HALF  = 0.5 * (SPEED_HIGH - SPEED_LOW)        # 1.0 m/s

# Pre-built tensors for the GPU path; produced lazily so a CPU-only test still works.
_TENSOR_CACHE: dict = {}


def _get_action_tensors(device, dtype):
    key = (device, dtype)
    if key not in _TENSOR_CACHE:
        _TENSOR_CACHE[key] = (
            torch.tensor([STEER_LIMIT, SPEED_HALF], device=device, dtype=dtype),
            torch.tensor([0.0,         SPEED_MID],  device=device, dtype=dtype),
            torch.tensor([-STEER_LIMIT, SPEED_LOW], device=device, dtype=dtype),
            torch.tensor([+STEER_LIMIT, SPEED_HIGH], device=device, dtype=dtype),
        )
    return _TENSOR_CACHE[key]


def raw_to_env_action(raw_action):
    """
    Map an unbounded raw action to the simulator's [steer, speed].

    Accepts either a 1D/2D torch.Tensor or a numpy array / list of length 2.
    """
    if isinstance(raw_action, torch.Tensor):
        scale, bias, low, high = _get_action_tensors(raw_action.device, raw_action.dtype)
        env_action = raw_action * scale + bias
        return torch.maximum(low, torch.minimum(env_action, high))
    arr = np.asarray(raw_action, dtype=np.float32)
    scale = np.array([STEER_LIMIT, SPEED_HALF], dtype=np.float32)
    bias  = np.array([0.0,         SPEED_MID],  dtype=np.float32)
    out = arr * scale + bias
    out[..., 0] = np.clip(out[..., 0], -STEER_LIMIT, +STEER_LIMIT)
    out[..., 1] = np.clip(out[..., 1],  SPEED_LOW,    SPEED_HIGH)
    return out


class ActorCritic(nn.Module):
    """
    Shared-trunk actor-critic that operates on latent vectors.

    The trunk is small (two hidden layers) because the encoder has already
    done the heavy lifting of compressing the raw sensor data.
    """

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden_dim, 1)

        # Orthogonal init following PPO paper recommendations
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)

    def forward(self, z):
        """
        Args:
            z: (B, latent_dim)

        Returns:
            mu:      (B, action_dim)  — mean action
            log_std: (action_dim,)    — learned log std
            value:   (B, 1)           — state value estimate
        """
        h = self.trunk(z)
        mu = self.actor_mean(h)
        log_std = torch.clamp(self.actor_log_std, LOG_STD_MIN, LOG_STD_MAX)
        value = self.critic(h)
        return mu, log_std, value

    def get_action(self, z, deterministic=False):
        """Sample (or take greedy) action and return log_prob + value."""
        mu, log_std, value = self.forward(z)
        std = log_std.exp()
        dist = Normal(mu, std)
        if deterministic:
            action = mu
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob, value

    def evaluate_actions(self, z, actions):
        """Compute log_prob, entropy, and value for given (z, action) pairs."""
        mu, log_std, value = self.forward(z)
        std = log_std.exp()
        dist = Normal(mu, std)
        log_prob = dist.log_prob(actions).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        return log_prob, entropy, value


class RolloutBuffer:
    """
    Stores one rollout of (z, action, log_prob, reward, done, value) tuples.
    After rollout collection, computes GAE advantages and returns.
    """

    def __init__(self, rollout_steps: int, latent_dim: int, action_dim: int, device):
        self.rollout_steps = rollout_steps
        self.device = device
        T = rollout_steps
        self.zs = torch.zeros(T, latent_dim, device=device)
        self.actions = torch.zeros(T, action_dim, device=device)
        self.log_probs = torch.zeros(T, 1, device=device)
        self.rewards = torch.zeros(T, 1, device=device)
        self.dones = torch.zeros(T, 1, device=device)
        self.values = torch.zeros(T, 1, device=device)
        self.ptr = 0

    def store(self, z, action, log_prob, reward, done, value):
        i = self.ptr
        self.zs[i] = z
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.dones[i] = done
        self.values[i] = value
        self.ptr += 1

    def compute_returns(self, last_value, gamma=0.99, gae_lambda=0.95):
        """Compute GAE advantages and discounted returns in-place."""
        T = self.rollout_steps
        advantages = torch.zeros(T, 1, device=self.device)
        last_gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        self.returns = advantages + self.values
        self.advantages = advantages
        self.ptr = 0

    def get_batches(self, minibatch_size: int):
        """Yield random minibatches of the rollout."""
        indices = torch.randperm(self.rollout_steps, device=self.device)
        for start in range(0, self.rollout_steps, minibatch_size):
            idx = indices[start: start + minibatch_size]
            yield (
                self.zs[idx],
                self.actions[idx],
                self.log_probs[idx],
                self.advantages[idx],
                self.returns[idx],
            )


class PPO:
    """
    Proximal Policy Optimisation for L-JEPA RL stage.

    The encoder is injected externally (frozen) so this class only owns the
    ActorCritic network.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        epochs_per_update: int = 10,
        minibatch_size: int = 64,
        max_grad_norm: float = 0.5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        rollout_steps: int = 2048,
        device: str = 'cpu',
    ):
        self.device = torch.device(device)
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs_per_update = epochs_per_update
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.rollout_steps = rollout_steps

        self.ac = ActorCritic(latent_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer(rollout_steps, latent_dim, action_dim, self.device)

    def act(self, z: torch.Tensor, deterministic=False):
        """Sample action given latent z. Returns (action, log_prob, value) as tensors."""
        with torch.no_grad():
            action, log_prob, value = self.ac.get_action(z, deterministic=deterministic)
        return action, log_prob, value

    def store(self, z, action, log_prob, reward, done, value):
        self.buffer.store(z, action, log_prob,
                          torch.tensor([[reward]], device=self.device, dtype=torch.float32),
                          torch.tensor([[float(done)]], device=self.device, dtype=torch.float32),
                          value)

    def update(self, last_value):
        """Run PPO update epochs. Returns dict of loss stats."""
        self.buffer.compute_returns(last_value, self.gamma, self.gae_lambda)

        stats = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'n_updates': 0}

        for _ in range(self.epochs_per_update):
            for zs, acts, old_lp, advs, rets in self.buffer.get_batches(self.minibatch_size):
                # Normalise advantages
                advs = (advs - advs.mean()) / (advs.std() + 1e-8)

                new_lp, entropy, value = self.ac.evaluate_actions(zs, acts)
                ratio = (new_lp - old_lp).exp()

                # Clipped policy loss
                surr1 = ratio * advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advs
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_loss = F.mse_loss(value, rets)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

                stats['policy_loss'] += policy_loss.item()
                stats['value_loss'] += value_loss.item()
                stats['entropy'] += entropy.mean().item()
                stats['n_updates'] += 1

        n = max(stats['n_updates'], 1)
        stats['policy_loss'] /= n
        stats['value_loss'] /= n
        stats['entropy'] /= n
        return stats
