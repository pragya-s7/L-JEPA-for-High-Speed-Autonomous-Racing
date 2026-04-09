"""
Action-conditioned Predictor for L-JEPA.

Takes the current latent z_t and a sequence of future actions a_{t:t+k-1},
and predicts the future latent embedding z_{t+k}.

The predictor is intentionally kept *small* relative to the encoder so that
it cannot memorise the training data or act as a de-facto generative model.
"""
import torch
import torch.nn as nn


class Predictor(nn.Module):
    """
    Predicts z_{t+k} given (z_t, a_{t:t+k-1}).

    Input:
        z:       (B, latent_dim)  — current context embedding
        actions: (B, k, action_dim) — future action sequence

    Output:
        z_pred:  (B, latent_dim)  — predicted future embedding
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        prediction_horizon: int,
        hidden_dims=(256, 256),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.prediction_horizon = prediction_horizon

        # Flatten the action sequence and concatenate with z
        in_dim = latent_dim + action_dim * prediction_horizon
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        layers += [nn.Linear(prev, latent_dim)]
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z, actions):
        """
        Args:
            z:       (B, latent_dim)
            actions: (B, k, action_dim)  or  (B, k * action_dim)

        Returns:
            z_pred: (B, latent_dim)
        """
        B = z.shape[0]
        a_flat = actions.reshape(B, -1)          # (B, k*action_dim)
        x = torch.cat([z, a_flat], dim=-1)       # (B, latent_dim + k*action_dim)
        return self.norm(self.net(x))
