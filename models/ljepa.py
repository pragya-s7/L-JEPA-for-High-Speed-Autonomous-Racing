"""
L-JEPA: LiDAR Joint-Embedding Predictive Architecture.

Wraps the ContextEncoder, TargetEncoder, and Predictor into a single module
with a convenience forward() that computes the pretraining loss.

Pretraining objective (non-generative):
    L = || P(z_context, a_{t:t+k-1}) - sg(z_target_{t+k}) ||^2_2

where:
    z_context  = ContextEncoder(obs_{t-h:t})
    z_target   = TargetEncoder(obs_{t+k-h:t+k})   [stop-gradient]
    P          = Predictor
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ContextEncoder, TargetEncoder
from .predictor import Predictor


class LJEPA(nn.Module):
    def __init__(
        self,
        lidar_dim: int,
        vel_dim: int,
        context_window: int,
        latent_dim: int,
        action_dim: int,
        prediction_horizon: int,
        conv_channels=(32, 64, 128),
        conv_kernel: int = 5,
        conv_stride: int = 2,
        predictor_hidden=(256, 256),
        ema_decay: float = 0.996,
    ):
        super().__init__()

        self.context_encoder = ContextEncoder(
            lidar_dim=lidar_dim,
            vel_dim=vel_dim,
            context_window=context_window,
            latent_dim=latent_dim,
            conv_channels=conv_channels,
            conv_kernel=conv_kernel,
            conv_stride=conv_stride,
        )
        self.target_encoder = TargetEncoder(self.context_encoder, ema_decay=ema_decay)
        self.predictor = Predictor(
            latent_dim=latent_dim,
            action_dim=action_dim,
            prediction_horizon=prediction_horizon,
            hidden_dims=predictor_hidden,
        )

    def encode(self, obs_seq):
        """Encode with the online (context) encoder. Used at RL time."""
        return self.context_encoder(obs_seq)

    def forward(self, context_obs, future_obs, actions):
        """
        Compute L-JEPA pretraining loss.

        Args:
            context_obs: (B, context_window, obs_dim) — observations at t-h..t
            future_obs:  (B, context_window, obs_dim) — observations at t+k-h..t+k
            actions:     (B, k, action_dim) — actions taken from t to t+k-1

        Returns:
            loss: scalar MSE between predicted and target embeddings
            z_context: (B, latent_dim) — context embeddings (for logging)
            z_pred:    (B, latent_dim) — predictor output
        """
        z_context = self.context_encoder(context_obs)          # (B, d)
        z_pred = self.predictor(z_context, actions)             # (B, d)

        with torch.no_grad():
            z_target = self.target_encoder(future_obs)          # (B, d) — no grad

        loss = F.mse_loss(z_pred, z_target)
        return loss, z_context, z_pred

    def update_ema(self):
        """Call after each optimiser step to update the target encoder."""
        self.target_encoder.update_ema(self.context_encoder)
