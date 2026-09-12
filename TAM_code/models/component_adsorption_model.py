import torch
from torch import nn
from typing import List, Optional

from .mof_encoder_utils import load_partial_mof_encoder, split_mof_feature_indices


class AdsorptionRegressor(nn.Module):
    """Single MLP regressor for component adsorption loading."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        output_dim: int = 2,
        coupled_co2_head: bool = False,
        feature_cols: Optional[List[str]] = None,
        use_mof_encoder: bool = False,
        mof_pretrained_encoder: str = "",
    ):
        super().__init__()
        self.coupled_co2_head = coupled_co2_head
        self.use_mof_encoder = use_mof_encoder
        if self.use_mof_encoder:
            if feature_cols is None:
                raise ValueError("feature_cols is required when use_mof_encoder=True")
            mof_indices, other_indices, mof_cols, _ = split_mof_feature_indices(feature_cols)
            self.register_buffer("mof_feature_indices", torch.as_tensor(mof_indices, dtype=torch.long), persistent=False)
            self.register_buffer(
                "other_feature_indices",
                torch.as_tensor(other_indices, dtype=torch.long),
                persistent=False,
            )
            self.mof_encoder = nn.Sequential(
                nn.Linear(len(mof_indices), hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.other_encoder = nn.Sequential(
                nn.Linear(len(other_indices), hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.backbone = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            if mof_pretrained_encoder:
                load_partial_mof_encoder(self.mof_encoder, mof_pretrained_encoder, mof_cols)
        else:
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
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
        self.head1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        head0_input_dim = hidden_dim * 2 + 1 if coupled_co2_head else hidden_dim
        self.head0 = nn.Sequential(
            nn.Linear(head0_input_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        if output_dim != 2:
            raise ValueError("AdsorptionRegressor currently expects output_dim=2")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_mof_encoder:
            mof_features = x.index_select(dim=1, index=self.mof_feature_indices)
            other_features = x.index_select(dim=1, index=self.other_feature_indices)
            hidden = self.backbone(torch.cat([self.mof_encoder(mof_features), self.other_encoder(other_features)], dim=-1))
        else:
            hidden = self.backbone(x)
        co2_hidden = self.co2_tower(hidden)
        x_hidden = self.x_tower(hidden)
        x_pred = self.head1(x_hidden)
        if self.coupled_co2_head:
            co2_input = torch.cat([co2_hidden, x_hidden, x_pred], dim=-1)
        else:
            co2_input = co2_hidden
        return torch.cat([self.head0(co2_input), x_pred], dim=-1)
