import math
from typing import Optional

import torch
from torch import nn


class TabularFeatureTokenizer(nn.Module):
    """Tokenize each scalar descriptor as one Transformer token."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim, hidden_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.feature_embedding = nn.Parameter(torch.zeros(1, input_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.weight)
        nn.init.normal_(self.feature_embedding, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        tokens = features.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        tokens = tokens + self.feature_embedding
        cls = self.cls_token.expand(features.size(0), -1, -1)
        return self.dropout(torch.cat([cls, tokens], dim=1))


class TabularTransformerBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        num_layers: int = 4,
        attention_heads: int = 8,
        ffn_dim: Optional[int] = None,
    ):
        super().__init__()
        if hidden_dim % attention_heads != 0:
            raise ValueError("--hidden-dim must be divisible by --attention-heads")
        ffn_dim = int(ffn_dim or hidden_dim * 4)
        self.tokenizer = TabularFeatureTokenizer(input_dim, hidden_dim, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(features)
        encoded = self.encoder(tokens)
        cls = encoded[:, 0]
        pooled = encoded[:, 1:].mean(dim=1)
        return self.final_norm(cls + pooled)


class TabularTransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        output_dim: int = 2,
        transformer_layers: int = 4,
        attention_heads: int = 8,
        ffn_dim: Optional[int] = None,
    ):
        super().__init__()
        if output_dim not in {1, 2}:
            raise ValueError("TabularTransformerRegressor expects output_dim=1 or output_dim=2")
        self.output_dim = output_dim
        self.backbone = TabularTransformerBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=transformer_layers,
            attention_heads=attention_heads,
            ffn_dim=ffn_dim,
        )
        self.co2_tower = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.x_tower = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head0 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.head1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(features.float())
        co2_pred = self.head0(self.co2_tower(hidden))
        if self.output_dim == 1:
            return co2_pred
        return torch.cat([co2_pred, self.head1(self.x_tower(hidden))], dim=-1)
