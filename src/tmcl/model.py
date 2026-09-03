from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    state_dim: int = 4
    action_dim: int = 1
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1


class CarBrain(nn.Module):
    """Causal sequence policy for in-context driving adaptation.

    Each token contains current state plus previous action/reward/done.
    The model predicts the current action. Hidden vehicle dynamics are not
    provided directly; useful adaptation must therefore be inferred from
    interaction history.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        token_dim = cfg.state_dim + cfg.action_dim + 2
        self.token_proj = nn.Linear(token_dim, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 512, cfg.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.d_model,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.policy = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.action_dim),
            nn.Tanh(),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return one action prediction per timestep.

        tokens: [B, T, state_dim + action_dim + 2]
        """
        _, t, _ = tokens.shape
        if t > self.pos_emb.shape[1]:
            raise ValueError(f"sequence length {t} exceeds positional limit")
        x = self.token_proj(tokens) + self.pos_emb[:, :t]
        mask = torch.triu(
            torch.ones(t, t, device=tokens.device, dtype=torch.bool), diagonal=1
        )
        h = self.transformer(x, mask=mask)
        return self.policy(self.norm(h))

    @torch.no_grad()
    def act(self, tokens: torch.Tensor) -> torch.Tensor:
        """Predict only the final action for a context window."""
        return self(tokens)[:, -1]
