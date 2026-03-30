from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PatientSpecificDecayTransformer(nn.Module):
    def __init__(
        self,
        num_dynamic_features: int,
        num_static_features: int,
        num_tasks: int,
        num_time_bins: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_dynamic_features = num_dynamic_features
        self.num_tasks = num_tasks

        self.static_encoder = nn.Sequential(
            nn.Linear(num_static_features, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.decay_projector = nn.Linear(d_model, num_dynamic_features)
        self.input_projection = nn.Linear(num_dynamic_features * 4 + d_model, d_model)
        self.position_embedding = nn.Embedding(num_time_bins + 1, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_tasks)

    def _decay_fill(
        self,
        values: torch.Tensor,
        masks: torch.Tensor,
        deltas: torch.Tensor,
        static_context: torch.Tensor,
    ) -> torch.Tensor:
        rates = F.softplus(self.decay_projector(static_context)) + 1e-4
        state = torch.zeros_like(values[:, 0, :])
        outputs = []
        for t in range(values.size(1)):
            gamma = torch.exp(-rates * deltas[:, t, :])
            imputed = masks[:, t, :] * values[:, t, :] + (1.0 - masks[:, t, :]) * (gamma * state)
            state = imputed
            outputs.append(imputed)
        return torch.stack(outputs, dim=1)

    def forward(
        self,
        values: torch.Tensor,
        masks: torch.Tensor,
        counts: torch.Tensor,
        deltas: torch.Tensor,
        static_features: torch.Tensor,
    ) -> torch.Tensor:
        static_context = self.static_encoder(static_features)
        decayed = self._decay_fill(values, masks, deltas, static_context)
        count_signal = torch.log1p(counts)
        static_tokens = static_context.unsqueeze(1).expand(-1, values.size(1), -1)
        token_inputs = torch.cat([decayed, masks, count_signal, deltas, static_tokens], dim=-1)
        token_embeddings = self.input_projection(token_inputs)

        positions = torch.arange(values.size(1), device=values.device).unsqueeze(0)
        token_embeddings = token_embeddings + self.position_embedding(positions)

        cls = self.cls_token.expand(values.size(0), -1, -1)
        cls = cls + static_context.unsqueeze(1)
        encoded = self.encoder(self.dropout(torch.cat([cls, token_embeddings], dim=1)))
        pooled = encoded[:, 0, :]
        return self.head(pooled)

