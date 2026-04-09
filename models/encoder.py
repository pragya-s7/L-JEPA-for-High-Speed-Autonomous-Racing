"""
Context Encoder and Target Encoder for L-JEPA.

The ContextEncoder maps a history of (LiDAR scan, velocity) pairs to a compact
latent vector z ∈ R^latent_dim.

Architecture:
  - 1D Conv over the subsampled LiDAR scan to capture local geometry
  - Flatten + concat with velocity features
  - Small MLP head -> latent_dim

The TargetEncoder is an EMA copy of the ContextEncoder. Its weights are never
updated by gradient descent; only by EMA update after each training step.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class LiDARConvNet(nn.Module):
    """
    1D convolutional feature extractor for a single LiDAR scan.

    Input: (batch, lidar_dim) float tensor of range readings
    Output: (batch, feature_dim) feature vector
    """

    def __init__(self, lidar_dim: int, channels=(32, 64, 128), kernel_size: int = 5, stride: int = 2):
        super().__init__()
        layers = []
        in_ch = 1
        current_len = lidar_dim
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            ]
            in_ch = out_ch
            current_len = (current_len + 2 * (kernel_size // 2) - kernel_size) // stride + 1

        self.conv = nn.Sequential(*layers)
        self.out_dim = in_ch * current_len

    def forward(self, x):
        # x: (B, lidar_dim)
        x = x.unsqueeze(1)          # (B, 1, lidar_dim)
        x = self.conv(x)            # (B, C, L)
        return x.flatten(1)         # (B, C*L)


class ContextEncoder(nn.Module):
    """
    Maps a context window of (subsampled LiDAR scan + velocity) observations
    to a latent vector z ∈ R^latent_dim.

    Input shape: (batch, context_window, obs_dim)
      where obs_dim = lidar_dim + vel_dim

    Processing:
      1. For each timestep, run the LiDAR CNN on the scan portion.
      2. Concatenate all per-step CNN features + velocity features.
      3. MLP -> latent_dim with LayerNorm.
    """

    def __init__(
        self,
        lidar_dim: int,
        vel_dim: int,
        context_window: int,
        latent_dim: int,
        conv_channels=(32, 64, 128),
        conv_kernel: int = 5,
        conv_stride: int = 2,
    ):
        super().__init__()
        self.lidar_dim = lidar_dim
        self.vel_dim = vel_dim
        self.context_window = context_window
        self.latent_dim = latent_dim

        # Shared LiDAR CNN (applied to each timestep independently)
        self.lidar_cnn = LiDARConvNet(lidar_dim, conv_channels, conv_kernel, conv_stride)

        # MLP that fuses all timestep features
        fused_dim = (self.lidar_cnn.out_dim + vel_dim) * context_window
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, obs_seq):
        """
        Args:
            obs_seq: (B, context_window, lidar_dim + vel_dim)

        Returns:
            z: (B, latent_dim)
        """
        B, T, D = obs_seq.shape
        assert T == self.context_window

        lidar = obs_seq[:, :, :self.lidar_dim]   # (B, T, lidar_dim)
        vel = obs_seq[:, :, self.lidar_dim:]       # (B, T, vel_dim)

        # Apply CNN to each timestep: reshape to (B*T, lidar_dim) -> (B*T, cnn_out)
        lidar_flat = lidar.reshape(B * T, self.lidar_dim)
        cnn_feats = self.lidar_cnn(lidar_flat)      # (B*T, cnn_out)
        cnn_feats = cnn_feats.reshape(B, T, -1)     # (B, T, cnn_out)

        # Concat lidar features + velocity per step
        step_feats = torch.cat([cnn_feats, vel], dim=-1)   # (B, T, cnn_out+vel_dim)
        fused = step_feats.reshape(B, -1)                   # (B, T*(cnn_out+vel_dim))

        z = self.norm(self.mlp(fused))
        return z


class TargetEncoder(nn.Module):
    """
    EMA copy of the ContextEncoder. Weights are never optimised directly;
    they are updated via update_ema() after each training step.

    This provides stable regression targets for the L-JEPA pretraining
    objective without needing contrastive negative samples.
    """

    def __init__(self, context_encoder: ContextEncoder, ema_decay: float = 0.996):
        super().__init__()
        self.ema_decay = ema_decay
        # Deep copy; parameters frozen from optimiser's perspective
        self.encoder = copy.deepcopy(context_encoder)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_ema(self, online_encoder: ContextEncoder):
        """
        Polyak/EMA update:  θ_target ← decay * θ_target + (1 - decay) * θ_online
        """
        for param_t, param_o in zip(self.encoder.parameters(), online_encoder.parameters()):
            param_t.data.mul_(self.ema_decay).add_(param_o.data, alpha=1.0 - self.ema_decay)

    def forward(self, obs_seq):
        return self.encoder(obs_seq)
