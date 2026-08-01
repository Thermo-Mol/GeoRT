from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .chem import EXPLICIT_DESCRIPTOR_NAMES


class RBFDistance(nn.Module):
    def __init__(self, n_centers: int, cutoff: float, gamma: float):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, n_centers)
        self.register_buffer("centers", centers)
        self.gamma = gamma
        self.cutoff = cutoff

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        d = distance.unsqueeze(-1)
        rbf = torch.exp(-self.gamma * (d - self.centers) ** 2)
        cutoff_envelope = 0.5 * (torch.cos(torch.clamp(distance, max=self.cutoff) * math.pi / self.cutoff) + 1.0)
        cutoff_envelope = cutoff_envelope * (distance <= self.cutoff).float()
        return rbf * cutoff_envelope.unsqueeze(-1)


class PairBiasedSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + attn_bias
        key_mask = mask[:, None, None, :].bool()
        fill_value = -1.0e4 if scores.dtype in (torch.float16, torch.bfloat16) else -1.0e9
        scores = scores.masked_fill(~key_mask, fill_value)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        out = self.out(out)
        return out * mask.unsqueeze(-1).float()


class GeometryTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, ffn_multiplier: int, condition_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = PairBiasedSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_multiplier, d_model),
            nn.Dropout(dropout),
        )
        self.film = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, d_model * 2))

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor, mask: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(condition).chunk(2, dim=-1)
        x_norm = self.norm1(x)
        x_mod = x_norm * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        x = x + self.attn(x_mod, attn_bias, mask)
        x = x + self.ffn(self.norm2(x)) * mask.unsqueeze(-1).float()
        return x


class AttentivePool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)
        fill_value = -1.0e4 if scores.dtype in (torch.float16, torch.bfloat16) else -1.0e9
        scores = scores.masked_fill(~mask.bool(), fill_value)
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class GeometryAwareStateTransformer(nn.Module):
    """Invariant 3D molecular Transformer conditioned on temperature and pressure.

    Coordinates are used through pairwise distances and bond-aware attention bias.
    Therefore translation and rotation of the molecule cannot change the prediction.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        mol_cfg = config["molecule"]
        model_cfg = config["model"]
        d_model = int(model_cfg["d_model"])
        n_heads = int(model_cfg["n_heads"])
        condition_dim = int(model_cfg["condition_dim"])
        dropout = float(model_cfg["dropout"])
        condition_input_dim = len(config["data"].get("condition_cols", [config["data"]["temperature_col"], config["data"]["pressure_col"]]))
        self.use_explicit_descriptors = bool(mol_cfg.get("use_explicit_graph_descriptors", False))
        self.use_group_features = bool(model_cfg.get("use_group_features", False))
        descriptor_dim = int(model_cfg.get("descriptor_dim", 128))
        group_input_dim = len(config["data"].get("group_cols", []))
        group_feature_dim = int(model_cfg.get("group_feature_dim", 128))
        if self.use_group_features and group_input_dim == 0:
            raise ValueError("model.use_group_features is true, but data.group_cols is empty.")

        self.n_heads = n_heads
        self.rbf = RBFDistance(
            n_centers=int(mol_cfg["rbf_centers"]),
            cutoff=float(mol_cfg["distance_cutoff_angstrom"]),
            gamma=float(mol_cfg["rbf_gamma"]),
        )
        self.atom_z_emb = nn.Embedding(119, d_model, padding_idx=0)
        self.degree_emb = nn.Embedding(9, d_model, padding_idx=0)
        self.charge_emb = nn.Embedding(11, d_model, padding_idx=5)
        self.hybrid_emb = nn.Embedding(8, d_model, padding_idx=0)
        self.binary_emb = nn.Embedding(2, d_model)
        self.mass_proj = nn.Linear(1, d_model)

        pair_dim = int(mol_cfg["rbf_centers"]) + 16
        self.bond_emb = nn.Embedding(5, 16, padding_idx=0)
        self.pair_bias = nn.Sequential(
            nn.Linear(pair_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_heads),
        )

        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_input_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
            nn.SiLU(),
        )
        self.condition_to_token = nn.Linear(condition_dim, d_model)
        self.descriptor_encoder = nn.Sequential(
            nn.Linear(len(EXPLICIT_DESCRIPTOR_NAMES), descriptor_dim),
            nn.LayerNorm(descriptor_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(descriptor_dim, descriptor_dim),
            nn.SiLU(),
        )
        self.group_encoder = nn.Sequential(
            nn.Linear(group_input_dim, group_feature_dim),
            nn.LayerNorm(group_feature_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(group_feature_dim, group_feature_dim),
            nn.SiLU(),
        ) if self.use_group_features else None

        self.layers = nn.ModuleList(
            [
                GeometryTransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    ffn_multiplier=int(model_cfg["ffn_multiplier"]),
                    condition_dim=condition_dim,
                )
                for _ in range(int(model_cfg["n_layers"]))
            ]
        )
        self.pooling = model_cfg.get("pooling", "attentive")
        self.pool = AttentivePool(d_model)
        fused_dim = (
            d_model * 2
            + (descriptor_dim if self.use_explicit_descriptors else 0)
            + (group_feature_dim if self.use_group_features else 0)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.critical_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

    def _atom_embedding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.atom_z_emb(batch["atom_z"].clamp(0, 118))
        x = x + self.degree_emb(batch["degree"].clamp(0, 8))
        x = x + self.charge_emb(batch["formal_charge"].clamp(0, 10))
        x = x + self.hybrid_emb(batch["hybridization"].clamp(0, 7))
        x = x + self.binary_emb(batch["aromatic"].clamp(0, 1))
        x = x + self.binary_emb(batch["in_ring"].clamp(0, 1))
        x = x + self.mass_proj(batch["mass"].unsqueeze(-1))
        return x * batch["atom_mask"].unsqueeze(-1).float()

    def _attention_bias(self, batch: dict[str, torch.Tensor], seq_len: int) -> torch.Tensor:
        distance_rbf = self.rbf(batch["distance"])
        bond = self.bond_emb(batch["bond_type"].clamp(0, 4))
        pair = torch.cat([distance_rbf, bond], dim=-1)
        atom_bias = self.pair_bias(pair).permute(0, 3, 1, 2)

        bsz, _, n_atoms, _ = atom_bias.shape
        bias = atom_bias.new_zeros((bsz, self.n_heads, seq_len, seq_len))
        bias[:, :, :n_atoms, :n_atoms] = atom_bias
        return bias

    def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        atom_mask = batch["atom_mask"].bool()
        atom_x = self._atom_embedding(batch)
        condition = self.condition_encoder(batch["condition"])
        condition_token = self.condition_to_token(condition).unsqueeze(1)

        x = torch.cat([atom_x, condition_token], dim=1)
        state_mask = torch.ones((x.size(0), 1), dtype=torch.bool, device=x.device)
        mask = torch.cat([atom_mask, state_mask], dim=1)
        attn_bias = self._attention_bias(batch, seq_len=x.size(1))

        for layer in self.layers:
            x = layer(x, attn_bias=attn_bias, mask=mask, condition=condition)

        atom_final = x[:, :-1, :]
        state_final = x[:, -1, :]
        mol_pool = self.pool(atom_final, atom_mask)
        return {
            "atom_final": atom_final,
            "state_final": state_final,
            "mol_pool": mol_pool,
            "condition": condition,
            "atom_mask": atom_mask,
        }

    def forward(self, batch: dict[str, torch.Tensor], return_aux: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        encoded = self.encode(batch)
        mol_pool = encoded["mol_pool"]
        state_final = encoded["state_final"]
        fused = [mol_pool, state_final]
        if self.use_explicit_descriptors:
            fused.append(self.descriptor_encoder(batch["explicit_descriptor"]))
        if self.use_group_features:
            if self.group_encoder is None:
                raise RuntimeError("Group encoder was not initialized.")
            fused.append(self.group_encoder(batch["group_feature"]))
        pred = self.head(torch.cat(fused, dim=-1)).squeeze(-1)
        if return_aux:
            return {"target": pred, "critical": self.critical_head(mol_pool)}
        return pred


class MultibranchGeometryStateTransformer(GeometryAwareStateTransformer):
    """Geometry Transformer with separate state and molecular descriptor branches.

    The geometry branch learns atom, bond, and pairwise-distance representations.
    The state branch keeps the low-dimensional corresponding-states signal
    explicit. The descriptor branch injects graph statistics extracted from
    `.mol` files. A learned gate decides how much each branch should contribute.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        model_cfg = config["model"]
        d_model = int(model_cfg["d_model"])
        condition_dim = int(model_cfg["condition_dim"])
        dropout = float(model_cfg["dropout"])
        branch_dim = int(model_cfg.get("hybrid_branch_dim", 128))
        descriptor_dim = int(model_cfg.get("descriptor_dim", 128))

        self.geometry_branch = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, branch_dim),
            nn.GELU(),
        )
        self.state_branch = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(branch_dim, branch_dim),
            nn.GELU(),
        )
        self.descriptor_encoder = nn.Sequential(
            nn.Linear(len(EXPLICIT_DESCRIPTOR_NAMES), descriptor_dim),
            nn.LayerNorm(descriptor_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(descriptor_dim, branch_dim),
            nn.GELU(),
        )

        fused_dim = branch_dim * 3
        self.branch_gate = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor], return_aux: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        atom_mask = batch["atom_mask"].bool()
        atom_x = self._atom_embedding(batch)
        condition = self.condition_encoder(batch["condition"])
        condition_token = self.condition_to_token(condition).unsqueeze(1)

        x = torch.cat([atom_x, condition_token], dim=1)
        state_mask = torch.ones((x.size(0), 1), dtype=torch.bool, device=x.device)
        mask = torch.cat([atom_mask, state_mask], dim=1)
        attn_bias = self._attention_bias(batch, seq_len=x.size(1))

        for layer in self.layers:
            x = layer(x, attn_bias=attn_bias, mask=mask, condition=condition)

        atom_final = x[:, :-1, :]
        state_final = x[:, -1, :]
        mol_pool = self.pool(atom_final, atom_mask)

        geometry_emb = self.geometry_branch(torch.cat([mol_pool, state_final], dim=-1))
        state_emb = self.state_branch(condition)
        descriptor_emb = self.descriptor_encoder(batch["explicit_descriptor"])
        fused = torch.cat([geometry_emb, state_emb, descriptor_emb], dim=-1)
        fused = fused * self.branch_gate(fused)
        pred = self.head(fused).squeeze(-1)
        if return_aux:
            return {"target": pred, "critical": self.critical_head(mol_pool)}
        return pred


class ResidualDescriptorCorrectedTransformer(GeometryAwareStateTransformer):
    """Geometry Transformer plus a bounded descriptor/state residual correction.

    The main path remains the original geometry Transformer. The descriptor/state
    branch predicts only a small residual in transformed target space, so it can
    improve difficult extrapolation cases without dominating the global fit.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        model_cfg = config["model"]
        d_model = int(model_cfg["d_model"])
        condition_dim = int(model_cfg["condition_dim"])
        descriptor_dim = int(model_cfg.get("descriptor_dim", 128))
        residual_dim = int(model_cfg.get("residual_branch_dim", 128))
        dropout = float(model_cfg["dropout"])
        self.residual_scale = float(model_cfg.get("residual_scale", 0.25))

        self.base_head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.descriptor_encoder = nn.Sequential(
            nn.Linear(len(EXPLICIT_DESCRIPTOR_NAMES), descriptor_dim),
            nn.LayerNorm(descriptor_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(descriptor_dim, residual_dim),
            nn.SiLU(),
        )
        self.state_residual_encoder = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, residual_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(residual_dim, residual_dim),
            nn.SiLU(),
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(residual_dim * 2),
            nn.Linear(residual_dim * 2, residual_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(residual_dim, 1),
        )
        last = self.residual_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, batch: dict[str, torch.Tensor], return_aux: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        atom_mask = batch["atom_mask"].bool()
        atom_x = self._atom_embedding(batch)
        condition = self.condition_encoder(batch["condition"])
        condition_token = self.condition_to_token(condition).unsqueeze(1)

        x = torch.cat([atom_x, condition_token], dim=1)
        state_mask = torch.ones((x.size(0), 1), dtype=torch.bool, device=x.device)
        mask = torch.cat([atom_mask, state_mask], dim=1)
        attn_bias = self._attention_bias(batch, seq_len=x.size(1))

        for layer in self.layers:
            x = layer(x, attn_bias=attn_bias, mask=mask, condition=condition)

        atom_final = x[:, :-1, :]
        state_final = x[:, -1, :]
        mol_pool = self.pool(atom_final, atom_mask)

        base_pred = self.base_head(torch.cat([mol_pool, state_final], dim=-1)).squeeze(-1)
        state_residual = self.state_residual_encoder(condition)
        descriptor_residual = self.descriptor_encoder(batch["explicit_descriptor"])
        residual_raw = self.residual_head(torch.cat([state_residual, descriptor_residual], dim=-1)).squeeze(-1)
        pred = base_pred + self.residual_scale * torch.tanh(residual_raw)
        if return_aux:
            return {"target": pred, "critical": self.critical_head(mol_pool)}
        return pred
