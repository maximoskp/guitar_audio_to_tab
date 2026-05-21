#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hidden-onset TabEstimator architecture.

This model keeps the legacy frame/tab + note/tab streams from Tab-estimator-hand
and adds a hidden onset branch that conditions the note-level stream without
using thresholded onset decisions at inference time.

Key design rules:
  - onset branch uses audio-only encoder memory, never hand-fused memory;
  - onset logits are auxiliary/diagnostic only;
  - note prediction is conditioned by hidden onset features, not thresholded onsets;
  - note-level hand prior is rest-preserving by default.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from espnet.nets.asr_interface import ASRInterface
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder

REST_CLASS = 20


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _length_mask(
    lengths: torch.Tensor,
    max_len: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return a bool mask of shape (B, max_len)."""
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.as_tensor(lengths, dtype=torch.long, device=device)
    lengths = lengths.to(device=device, dtype=torch.long)
    max_len = int(max_len)
    if max_len <= 0:
        raise ValueError(f"max_len must be positive, got {max_len}")
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def _crop_time_pair(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Crop pred/target along time dimension to a shared length.

    This keeps the loss robust if an encoder variant produces T_enc slightly
    different from the original feature length.
    """
    if pred.size(1) == target.size(1):
        if lengths is not None:
            lengths = torch.clamp(lengths, max=pred.size(1))
        return pred, target, lengths

    target_len = min(int(pred.size(1)), int(target.size(1)))
    pred = pred[:, :target_len]
    target = target[:, :target_len]
    if lengths is not None:
        lengths = torch.clamp(lengths, max=target_len)
    return pred, target, lengths


def _masked_mean(loss: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Mean over valid timesteps and all non-time trailing dimensions.

    loss shape: (B, T, ...)
    lengths shape: (B,)
    """
    if loss.dim() < 2:
        raise ValueError(f"Expected loss shape (B,T,...), got {tuple(loss.shape)}")

    T = int(loss.size(1))
    lengths = torch.clamp(lengths.to(loss.device, dtype=torch.long), min=0, max=T)
    mask = _length_mask(lengths, T, loss.device).to(loss.dtype)

    while mask.dim() < loss.dim():
        mask = mask.unsqueeze(-1)

    masked = loss * mask
    trailing = int(torch.tensor(loss.shape[2:]).prod().item()) if loss.dim() > 2 else 1
    denom = lengths.to(loss.dtype).sum().clamp_min(1.0) * float(trailing)
    return masked.sum() / denom


def _resize_sequence_time(sequence: torch.Tensor, target_len: int) -> torch.Tensor:
    """Resize a (B,T,F) sequence along time with linear interpolation."""
    target_len = int(target_len)
    if sequence.size(1) == target_len:
        return sequence
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")
    y = sequence.transpose(1, 2)
    y = F.interpolate(y, size=target_len, mode="linear", align_corners=False)
    return y.transpose(1, 2)


# -----------------------------------------------------------------------------
# Audio frontend
# -----------------------------------------------------------------------------


class ConvStack(nn.Module):
    def __init__(self, input_features: int, output_features: int, input_ch: int):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(input_ch, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )

        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cnn(x)
        y = y.transpose(1, 2).flatten(-2)
        y = self.fc(y)
        return y


# -----------------------------------------------------------------------------
# Soft hand-position modules
# -----------------------------------------------------------------------------


class SoftHandPositionFusion(nn.Module):
    """
    Add soft hand-position information into a hidden sequence.

    hidden shape: (B, T_hidden, hidden_dim)
    hand_pos shape: (B, T_hand, hand_pos_dim)
    """

    def __init__(self, hand_pos_dim: int, hidden_dim: int, dropout: float = 0.1, gate_init: float = 0.5):
        super().__init__()
        self.hand_pos_dim = int(hand_pos_dim)
        self.hidden_dim = int(hidden_dim)

        self.proj = nn.Sequential(
            nn.Linear(self.hand_pos_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(float(dropout)),
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))

    @staticmethod
    def resize_time(x: Optional[torch.Tensor], target_len: int) -> Optional[torch.Tensor]:
        if x is None:
            return None
        return _resize_sequence_time(x, int(target_len))

    def forward(self, hidden: torch.Tensor, hand_pos: Optional[torch.Tensor]) -> torch.Tensor:
        if hand_pos is None:
            return hidden

        if hand_pos.dim() == 2:
            hand_pos = hand_pos.unsqueeze(0)

        if hand_pos.dim() != 3:
            raise ValueError(
                f"Expected hand_pos shape (B, T, {self.hand_pos_dim}), got {tuple(hand_pos.shape)}"
            )

        if hand_pos.size(-1) != self.hand_pos_dim:
            raise ValueError(f"Expected hand_pos_dim={self.hand_pos_dim}, got {hand_pos.size(-1)}")

        hand_pos = hand_pos.to(device=hidden.device, dtype=hidden.dtype)
        hand_pos = self.resize_time(hand_pos, hidden.size(1))
        return hidden + self.gate.to(dtype=hidden.dtype) * self.proj(hand_pos)


class HandPositionPrior(nn.Module):
    """
    Converts a soft hand-position distribution into a tablature prior.

    hand_pos shape: (B, T, n_positions)
    prior output shape: (B, T, 6, 21)
    """

    def __init__(
        self,
        n_positions: int = 20,
        n_strings: int = 6,
        n_tab_classes: int = 21,
        hand_span: int = 4,
        rest_class: int = REST_CLASS,
        open_weight: float = 0.55,
        rest_weight: float = 0.50,
        playable_weight: float = 1.00,
        floor: float = 1e-5,
    ):
        super().__init__()
        self.n_positions = int(n_positions)
        self.n_strings = int(n_strings)
        self.n_tab_classes = int(n_tab_classes)
        self.hand_span = int(hand_span)
        self.rest_class = int(rest_class)
        self.open_weight = float(open_weight)
        self.rest_weight = float(rest_weight)
        self.playable_weight = float(playable_weight)
        self.floor = float(floor)

        prior_table = self._build_prior_table()
        self.register_buffer("prior_table", prior_table, persistent=False)

    def _build_prior_table(self) -> torch.Tensor:
        table = torch.full(
            (self.n_positions, self.n_tab_classes),
            self.floor,
            dtype=torch.float32,
        )

        for p in range(self.n_positions):
            table[p, 0] = self.open_weight

            if p == 0:
                low = 1
                high = self.hand_span - 1
            else:
                low = p
                high = p + self.hand_span - 1

            low = max(1, low)
            high = min(self.n_tab_classes - 2, high)

            if high >= low:
                table[p, low : high + 1] = self.playable_weight

            table[p, self.rest_class] = self.rest_weight

        table = table / table.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return table

    @staticmethod
    def resize_time(x: torch.Tensor, target_len: int) -> torch.Tensor:
        return _resize_sequence_time(x, int(target_len))

    def forward(self, hand_pos: Optional[torch.Tensor], target_len: Optional[int] = None) -> Optional[torch.Tensor]:
        if hand_pos is None:
            return None

        if hand_pos.dim() == 2:
            hand_pos = hand_pos.unsqueeze(0)

        if hand_pos.size(-1) != self.n_positions:
            raise ValueError(
                f"Expected hand_pos last dim {self.n_positions}, got {hand_pos.size(-1)}"
            )

        if target_len is not None:
            hand_pos = self.resize_time(hand_pos, int(target_len))

        table = self.prior_table.to(device=hand_pos.device, dtype=hand_pos.dtype)
        fret_prior = torch.matmul(hand_pos, table)
        fret_prior = fret_prior.unsqueeze(2).repeat(1, 1, self.n_strings, 1)
        fret_prior = fret_prior / fret_prior.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return fret_prior


def apply_hand_position_prior(
    tab_probs: torch.Tensor,
    hand_pos: Optional[torch.Tensor],
    prior_module: HandPositionPrior,
    strength: float = 0.0,
) -> torch.Tensor:
    if hand_pos is None or strength <= 0.0:
        return tab_probs

    prior = prior_module(hand_pos, target_len=tab_probs.size(1))
    prior = prior.to(device=tab_probs.device, dtype=tab_probs.dtype)
    adjusted = tab_probs * torch.pow(prior.clamp_min(1e-8), float(strength))
    adjusted = adjusted / adjusted.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return adjusted


def apply_hand_position_prior_rest_preserving(
    tab_probs: torch.Tensor,
    hand_pos: Optional[torch.Tensor],
    prior_module: HandPositionPrior,
    strength: float = 0.0,
    rest_class: int = REST_CLASS,
) -> torch.Tensor:
    """
    Apply hand prior without changing the model/audio rest probability.

    P'_rest = P_rest
    P'_fret = (1 - P_rest) * normalize(P_fret * prior_fret^strength)
    """
    if hand_pos is None or strength <= 0.0:
        return tab_probs

    if rest_class != tab_probs.size(-1) - 1:
        raise ValueError(
            "apply_hand_position_prior_rest_preserving assumes the rest class "
            f"is the last class. Got rest_class={rest_class}, num_classes={tab_probs.size(-1)}"
        )

    prior = prior_module(hand_pos, target_len=tab_probs.size(1))
    prior = prior.to(device=tab_probs.device, dtype=tab_probs.dtype)

    rest_prob = tab_probs[..., rest_class : rest_class + 1]
    note_mass = 1.0 - rest_prob

    fret_probs = tab_probs[..., :rest_class]
    fret_prior = prior[..., :rest_class]

    adjusted_frets = fret_probs * torch.pow(fret_prior.clamp_min(1e-8), float(strength))
    adjusted_frets = adjusted_frets / adjusted_frets.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    adjusted_frets = adjusted_frets * note_mass

    return torch.cat([adjusted_frets, rest_prob], dim=-1)


# -----------------------------------------------------------------------------
# Non-causal gated TCN onset head
# -----------------------------------------------------------------------------


class GatedTemporalBlock(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.25):
        super().__init__()

        kernel_size = int(kernel_size)
        dilation = int(dilation)

        if kernel_size % 2 == 0:
            raise ValueError("Use an odd onset TCN kernel_size to preserve sequence length.")

        padding = (kernel_size // 2) * dilation

        self.filter_conv = weight_norm(
            nn.Conv1d(
                int(n_inputs),
                int(n_outputs),
                kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
            )
        )
        self.gate_conv = weight_norm(
            nn.Conv1d(
                int(n_inputs),
                int(n_outputs),
                kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
            )
        )
        self.dropout = nn.Dropout(float(dropout))
        self.downsample = nn.Conv1d(int(n_inputs), int(n_outputs), kernel_size=1) if int(n_inputs) != int(n_outputs) else None
        self.out_activation = nn.ReLU()
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.normal_(self.filter_conv.weight, 0.0, 0.01)
        nn.init.normal_(self.gate_conv.weight, 0.0, 0.01)
        if self.filter_conv.bias is not None:
            nn.init.zeros_(self.filter_conv.bias)
        if self.gate_conv.bias is not None:
            nn.init.zeros_(self.gate_conv.bias)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0.0, 0.01)
            if self.downsample.bias is not None:
                nn.init.zeros_(self.downsample.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        filt = torch.tanh(self.filter_conv(x))
        gate = torch.sigmoid(self.gate_conv(x))
        out = self.dropout(filt * gate)

        res = x if self.downsample is None else self.downsample(x)
        if out.shape[-1] != res.shape[-1]:
            min_len = min(out.shape[-1], res.shape[-1])
            out = out[..., :min_len]
            res = res[..., :min_len]

        return self.out_activation(out + res)


class OnsetTCNHead(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int = 6,
        num_channels: Any = 64,
        kernel_size: int = 5,
        dropout: float = 0.25,
    ):
        super().__init__()

        if isinstance(num_channels, int):
            num_channels = (int(num_channels),)
        num_channels = tuple(int(c) for c in num_channels)
        if not num_channels:
            raise ValueError("num_channels must contain at least one channel size.")

        layers = []
        for i, out_channels in enumerate(num_channels):
            in_channels = int(input_size) if i == 0 else num_channels[i - 1]
            dilation = 2 ** i
            layers.append(
                GatedTemporalBlock(
                    n_inputs=in_channels,
                    n_outputs=out_channels,
                    kernel_size=int(kernel_size),
                    dilation=dilation,
                    dropout=float(dropout),
                )
            )

        self.tcn = nn.Sequential(*layers)
        self.output = nn.Linear(num_channels[-1], int(output_size))
        self.feature_dim = int(num_channels[-1])

    def forward(self, hidden: torch.Tensor, return_features: bool = False):
        z = hidden.transpose(1, 2)
        z = self.tcn(z)
        z = z.transpose(1, 2)
        logits = self.output(z)
        if return_features:
            return logits, z
        return logits


class NoteOnsetConditioning(nn.Module):
    def __init__(self, onset_feature_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(int(onset_feature_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.Dropout(float(dropout)),
        )
        self.gate = nn.Sequential(
            nn.Linear(int(onset_feature_dim) + int(hidden_dim), int(hidden_dim)),
            nn.Sigmoid(),
        )

    def forward(self, note_hidden: torch.Tensor, onset_note_features: torch.Tensor) -> torch.Tensor:
        if onset_note_features.size(1) != note_hidden.size(1):
            onset_note_features = _resize_sequence_time(onset_note_features, note_hidden.size(1))
        onset_proj = self.proj(onset_note_features)
        gate = self.gate(torch.cat([note_hidden, onset_note_features], dim=-1))
        return note_hidden + gate * onset_proj


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------


class CustomLoss(nn.Module):
    """
    Loss for hidonset tablature training.

    Expects probability outputs for frame_tab_pred/note_tab_pred and raw logits
    for onset heads.
    """

    def __init__(
        self,
        onset_loss_weight: float = 0.25,
        onset_positive_weight: float = 10.0,
        global_onset_loss_weight: float = 0.25,
        global_onset_positive_weight: float = 10.0,
        tab_loss_weight: float = 1.0,
        use_galoss: bool = False,
    ):
        super().__init__()
        self.onset_loss_weight = float(onset_loss_weight)
        self.onset_positive_weight = float(onset_positive_weight)
        self.global_onset_loss_weight = float(global_onset_loss_weight)
        self.global_onset_positive_weight = float(global_onset_positive_weight)
        self.tab_loss_weight = float(tab_loss_weight)
        self.use_galoss = bool(use_galoss)
        self.GALoss = GuidedAttentionLoss(sigma=0.4, alpha=1.0)

    @staticmethod
    def categorical_tab_loss(
        pred_probs: torch.Tensor,
        gt_onehot: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        pred_probs, gt_onehot, lengths = _crop_time_pair(pred_probs, gt_onehot, lengths)
        # Sum over 21 tab classes -> loss shape (B,T,6), then mask over time.
        loss = -(gt_onehot * torch.log(pred_probs.clamp_min(1e-7))).sum(dim=-1)
        return _masked_mean(loss, lengths)

    def forward(
        self,
        frame_tab_pred: torch.Tensor,
        frame_tab_gt: torch.Tensor,
        note_tab_pred: torch.Tensor,
        note_tab_gt: torch.Tensor,
        frame_onset_pred: Optional[torch.Tensor],
        global_onset_pred: Optional[torch.Tensor],
        frame_onset_gt: Optional[torch.Tensor],
        attn: Optional[torch.Tensor],
        olens: torch.Tensor,
        note_len: torch.Tensor,
    ):
        frame_loss = self.categorical_tab_loss(frame_tab_pred, frame_tab_gt, olens)
        note_loss = self.categorical_tab_loss(note_tab_pred, note_tab_gt, note_len)

        if frame_onset_pred is not None and frame_onset_gt is not None and self.onset_loss_weight != 0.0:
            frame_onset_pred, frame_onset_gt, onset_lens = _crop_time_pair(
                frame_onset_pred,
                frame_onset_gt,
                olens,
            )
            pos_weight = torch.tensor(
                self.onset_positive_weight,
                dtype=frame_onset_pred.dtype,
                device=frame_onset_pred.device,
            )
            onset_raw = F.binary_cross_entropy_with_logits(
                frame_onset_pred,
                frame_onset_gt.to(dtype=frame_onset_pred.dtype),
                reduction="none",
                pos_weight=pos_weight,
            )
            onset_loss = _masked_mean(onset_raw, onset_lens)
        else:
            onset_loss = frame_tab_pred.new_tensor(0.0)

        if global_onset_pred is not None and frame_onset_gt is not None and self.global_onset_loss_weight != 0.0:
            global_onset_gt = torch.max(frame_onset_gt, dim=-1).values
            global_onset_pred, global_onset_gt, global_lens = _crop_time_pair(
                global_onset_pred,
                global_onset_gt,
                olens,
            )
            pos_weight = torch.tensor(
                self.global_onset_positive_weight,
                dtype=global_onset_pred.dtype,
                device=global_onset_pred.device,
            )
            global_raw = F.binary_cross_entropy_with_logits(
                global_onset_pred,
                global_onset_gt.to(dtype=global_onset_pred.dtype),
                reduction="none",
                pos_weight=pos_weight,
            )
            global_onset_loss = _masked_mean(global_raw, global_lens)
        else:
            global_onset_loss = frame_tab_pred.new_tensor(0.0)

        attn_loss = frame_tab_pred.new_tensor(0.0)
        if self.use_galoss and attn is not None:
            # attn is commonly shaped (layers*B, heads, T, T) or similar.
            if attn.dim() >= 4:
                for head in range(attn.shape[1]):
                    attn_loss = attn_loss + self.GALoss(attn[:, head], olens, olens)
            elif attn.dim() == 3:
                attn_loss = attn_loss + self.GALoss(attn, olens, olens)

        loss = (
            self.tab_loss_weight * (frame_loss + note_loss)
            + self.onset_loss_weight * onset_loss
            + self.global_onset_loss_weight * global_onset_loss
            + attn_loss
        )

        return loss, {
            "frame_loss": float(frame_loss.detach().item()),
            "note_loss": float(note_loss.detach().item()),
            "onset_loss": float(onset_loss.detach().item()),
            "global_onset_loss": float(global_onset_loss.detach().item()),
            "attn_loss": float(attn_loss.detach().item()),
        }


class GuidedAttentionLoss(nn.Module):
    def __init__(self, sigma: float = 0.2, alpha: float = 1.0, reset_always: bool = True):
        super().__init__()
        self.sigma = float(sigma)
        self.alpha = float(alpha)
        self.reset_always = bool(reset_always)
        self.guided_attn_masks = None
        self.masks = None

    @staticmethod
    def _lengths_to_list(lengths):
        if isinstance(lengths, torch.Tensor):
            return [int(x) for x in lengths.detach().cpu().tolist()]
        return [int(x) for x in lengths]

    def forward(self, att_ws: torch.Tensor, ilens: torch.Tensor, olens: torch.Tensor) -> torch.Tensor:
        if self.guided_attn_masks is None:
            self.guided_attn_masks = self._make_guided_attention_masks(ilens, olens, device=att_ws.device)
        if self.masks is None:
            self.masks = self._make_masks(ilens, olens).to(att_ws.device)

        # Crop masks if attention has extra/broadcasted batch rows.
        n = min(att_ws.size(0), self.guided_attn_masks.size(0))
        t1 = min(att_ws.size(-2), self.guided_attn_masks.size(-2))
        t2 = min(att_ws.size(-1), self.guided_attn_masks.size(-1))

        att_crop = att_ws[:n, :t1, :t2]
        guide = self.guided_attn_masks[:n, :t1, :t2]
        mask = self.masks[:n, :t1, :t2]

        losses = guide * att_crop
        if mask.any():
            loss = torch.mean(losses.masked_select(mask))
        else:
            loss = losses.mean() * 0.0

        if self.reset_always:
            self.guided_attn_masks = None
            self.masks = None

        return self.alpha * loss

    def _make_guided_attention_masks(self, ilens, olens, device=None) -> torch.Tensor:
        ilens_list = self._lengths_to_list(ilens)
        olens_list = self._lengths_to_list(olens)
        n_batches = len(ilens_list)
        max_ilen = max(ilens_list)
        max_olen = max(olens_list)

        guided_attn_masks = torch.zeros((n_batches, max_olen, max_ilen), device=device)
        for idx, (ilen, olen) in enumerate(zip(ilens_list, olens_list)):
            guided_attn_masks[idx, :olen, :ilen] = self._make_guided_attention_mask(
                ilen,
                olen,
                self.sigma,
                device=device,
            )
        return guided_attn_masks

    @staticmethod
    def _make_guided_attention_mask(ilen: int, olen: int, sigma: float, device=None) -> torch.Tensor:
        grid_x, grid_y = torch.meshgrid(
            torch.arange(int(olen), device=device),
            torch.arange(int(ilen), device=device),
            indexing="ij",
        )
        grid_x = grid_x.float()
        grid_y = grid_y.float()
        ilen_f = float(max(1, int(ilen)))
        olen_f = float(max(1, int(olen)))
        return 1.0 - torch.exp(-((grid_y / ilen_f - grid_x / olen_f) ** 2) / (2 * (float(sigma) ** 2)))

    @staticmethod
    def _make_masks(ilens: torch.Tensor, olens: torch.Tensor) -> torch.Tensor:
        ilens = ilens.to(dtype=torch.long)
        olens = olens.to(dtype=torch.long)
        max_ilen = int(ilens.max().item())
        max_olen = int(olens.max().item())
        in_masks = torch.arange(max_ilen, device=ilens.device).unsqueeze(0) < ilens.unsqueeze(1)
        out_masks = torch.arange(max_olen, device=olens.device).unsqueeze(0) < olens.unsqueeze(1)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)


# -----------------------------------------------------------------------------
# Hidden-Onset TabEstimator
# -----------------------------------------------------------------------------


class TabEstimator(ASRInterface, nn.Module):
    def __init__(
        self,
        mode: str,
        encoder_type: str,
        use_custom_decimation_func: bool,
        use_conv_stack: bool,
        n_bins: int,
        hop_length: int,
        sr: int,
        encoder_heads: int = 1,
        encoder_layers: int = 1,
        normalize_before: bool = True,
        use_hand_position: bool = False,
        hand_pos_dim: int = 20,
        hand_position_fusion: str = "hidden+prior",
        hand_hidden_gate_init: float = 0.5,
        hand_prior_strength: float = 0.35,
        hand_span: int = 4,
        note_target_length: int = 64,
        onset_hidden_dim: int = 64,
        onset_dropout: float = 0.25,
        onset_kernel_size: int = 3,
        onset_tcn_levels: int = 4,
        onset_use_raw_features: bool = True,
        onset_raw_proj_dim: int = 64,
        onset_raw_dropout: float = 0.10,
        onset_input_mode: str = "full",
        use_hidden_onset_to_note: bool = True,
        onset_note_fusion: str = "gated_add",
        detach_onset_features_for_note: bool = False,
        note_hidden_hand_fusion: bool = False,
        note_rest_preserving_prior: bool = True,
    ):
        super().__init__()

        self.mode = str(mode)
        self.use_custom_decimation_func = bool(use_custom_decimation_func)
        self.use_conv_stack = bool(use_conv_stack)
        self.hop_length = int(hop_length)
        self.sr = int(sr)
        self.encoder_output_size = 64
        self.n_encoder_ffn = 64
        self.encoder_attn_dropout = 0.0
        self.encoder_pos_dropout = 0.1
        self.conv_output_features = 16 * 32
        self.note_target_length = int(note_target_length)

        self.use_hand_position = bool(use_hand_position)
        self.hand_pos_dim = int(hand_pos_dim)
        self.hand_position_fusion = str(hand_position_fusion)
        self.hand_hidden_gate_init = float(hand_hidden_gate_init)
        self.hand_prior_strength = float(hand_prior_strength)
        self.hand_span = int(hand_span)
        self.note_hidden_hand_fusion = bool(note_hidden_hand_fusion)
        self.note_rest_preserving_prior = bool(note_rest_preserving_prior)

        self.onset_hidden_dim = int(onset_hidden_dim)
        self.onset_dropout = float(onset_dropout)
        self.onset_kernel_size = int(onset_kernel_size)
        self.onset_tcn_levels = int(onset_tcn_levels)
        self.onset_use_raw_features = bool(onset_use_raw_features)
        self.onset_raw_proj_dim = int(onset_raw_proj_dim)
        self.onset_raw_dropout = float(onset_raw_dropout)
        self.onset_input_mode = str(onset_input_mode)
        self.use_hidden_onset_to_note = bool(use_hidden_onset_to_note)
        self.onset_note_fusion = str(onset_note_fusion)
        self.detach_onset_features_for_note = bool(detach_onset_features_for_note)

        if self.mode != "tab":
            raise ValueError("Tab-estimator-hand-hidonset supports mode='tab' only.")

        if self.hand_position_fusion not in ["hidden", "prior", "hidden+prior", "none"]:
            raise ValueError("hand_position_fusion must be one of: 'hidden', 'prior', 'hidden+prior', 'none'")

        if self.onset_input_mode not in ["full", "encoder", "raw"]:
            raise ValueError("onset_input_mode must be one of: 'full', 'encoder', 'raw'")

        if self.onset_input_mode in ["full", "raw"] and not self.onset_use_raw_features:
            raise ValueError("onset_input_mode='full' or 'raw' requires onset raw feature projection.")

        if self.onset_note_fusion != "gated_add":
            raise ValueError("Currently only onset_note_fusion='gated_add' is supported.")

        if self.onset_tcn_levels <= 0:
            raise ValueError("onset_tcn_levels must be positive.")

        if self.use_conv_stack:
            self.convstack = ConvStack(int(n_bins), self.conv_output_features, 1)

        encoder_input_size = self.conv_output_features if self.use_conv_stack else int(n_bins)

        if encoder_type == "transformer":
            self.encoder = TransformerEncoder(
                encoder_input_size,
                output_size=self.encoder_output_size,
                attention_heads=int(encoder_heads),
                linear_units=self.n_encoder_ffn,
                num_blocks=int(encoder_layers),
                positional_dropout_rate=self.encoder_pos_dropout,
                attention_dropout_rate=self.encoder_attn_dropout,
                input_layer="linear",
                positionwise_layer_type="conv1d",
                normalize_before=normalize_before,
            )
        elif encoder_type == "conformer":
            self.encoder = ConformerEncoder(
                encoder_input_size,
                output_size=self.encoder_output_size,
                attention_heads=int(encoder_heads),
                linear_units=self.n_encoder_ffn,
                num_blocks=int(encoder_layers),
                attention_dropout_rate=self.encoder_attn_dropout,
                input_layer="linear",
                positionwise_layer_type="conv1d",
                positionwise_conv_kernel_size=3,
                normalize_before=normalize_before,
                macaron_style=False,
                rel_pos_type="latest",
                pos_enc_layer_type="rel_pos",
                selfattention_layer_type="rel_selfattn",
                cnn_module_kernel=3,
            )
        else:
            raise ValueError("encoder_type must be either 'transformer' or 'conformer'")

        self.frame_tab_output_layer = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.encoder_output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 126),
        )
        self.note_tab_output_layer = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.encoder_output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 126),
        )
        self.softmax_by_string = nn.Softmax(dim=3)

        self.note_encoder = ConformerEncoder(
            self.encoder_output_size,
            output_size=self.encoder_output_size,
            attention_heads=int(encoder_heads),
            linear_units=self.n_encoder_ffn,
            num_blocks=int(encoder_layers),
            attention_dropout_rate=self.encoder_attn_dropout,
            input_layer="linear",
            positionwise_layer_type="conv1d",
            positionwise_conv_kernel_size=3,
            normalize_before=normalize_before,
            macaron_style=False,
            rel_pos_type="latest",
            pos_enc_layer_type="rel_pos",
            selfattention_layer_type="rel_selfattn",
            cnn_module_kernel=3,
        )

        if self.use_hand_position:
            if self.hand_position_fusion in ["hidden", "hidden+prior"]:
                self.frame_hand_fusion = SoftHandPositionFusion(
                    hand_pos_dim=self.hand_pos_dim,
                    hidden_dim=self.encoder_output_size,
                    dropout=0.1,
                    gate_init=self.hand_hidden_gate_init,
                )
                self.note_hand_fusion = (
                    SoftHandPositionFusion(
                        hand_pos_dim=self.hand_pos_dim,
                        hidden_dim=self.encoder_output_size,
                        dropout=0.1,
                        gate_init=self.hand_hidden_gate_init,
                    )
                    if self.note_hidden_hand_fusion
                    else None
                )
            else:
                self.frame_hand_fusion = None
                self.note_hand_fusion = None

            if self.hand_position_fusion in ["prior", "hidden+prior"]:
                self.hand_prior = HandPositionPrior(
                    n_positions=self.hand_pos_dim,
                    n_strings=6,
                    n_tab_classes=21,
                    hand_span=self.hand_span,
                    rest_class=REST_CLASS,
                )
            else:
                self.hand_prior = None
        else:
            self.frame_hand_fusion = None
            self.note_hand_fusion = None
            self.hand_prior = None

        if self.onset_use_raw_features or self.onset_input_mode in ["raw", "full"]:
            self.onset_raw_feature_proj = nn.Sequential(
                nn.Linear(int(n_bins), self.onset_raw_proj_dim),
                nn.LayerNorm(self.onset_raw_proj_dim),
                nn.Dropout(self.onset_raw_dropout),
            )
        else:
            self.onset_raw_feature_proj = None

        if self.onset_input_mode == "encoder":
            onset_input_dim = self.encoder_output_size
        elif self.onset_input_mode == "raw":
            onset_input_dim = self.onset_raw_proj_dim
        else:
            onset_input_dim = self.encoder_output_size + self.onset_raw_proj_dim

        onset_channels = (self.onset_hidden_dim,) * self.onset_tcn_levels

        self.frame_onset_output_layer = OnsetTCNHead(
            onset_input_dim,
            output_size=6,
            num_channels=onset_channels,
            kernel_size=self.onset_kernel_size,
            dropout=self.onset_dropout,
        )
        self.global_onset_output_layer = OnsetTCNHead(
            onset_input_dim,
            output_size=1,
            num_channels=onset_channels,
            kernel_size=self.onset_kernel_size,
            dropout=self.onset_dropout,
        )


        if self.use_hidden_onset_to_note:
            self.note_onset_conditioning = NoteOnsetConditioning(
                onset_feature_dim=2 * self.onset_hidden_dim,
                hidden_dim=self.encoder_output_size,
                dropout=0.1,
            )
        else:
            self.note_onset_conditioning = None

    def _build_onset_input(self, audio_memory: torch.Tensor, raw_features: torch.Tensor) -> torch.Tensor:
        raw_proj = None
        if self.onset_raw_feature_proj is not None:
            raw_proj = self.onset_raw_feature_proj(raw_features)
            raw_proj = _resize_sequence_time(raw_proj, audio_memory.size(1))

        if self.onset_input_mode == "encoder":
            return audio_memory
        if self.onset_input_mode == "raw":
            if raw_proj is None:
                raise RuntimeError("onset_input_mode='raw' requires raw feature projection.")
            return raw_proj
        if self.onset_input_mode == "full":
            if raw_proj is None:
                raise RuntimeError("onset_input_mode='full' requires raw feature projection.")
            return torch.cat([audio_memory, raw_proj], dim=-1)
        raise ValueError(f"Unknown onset_input_mode: {self.onset_input_mode}")

    def _interpolate_to_note_grid(self, memory: torch.Tensor, olens: torch.Tensor) -> torch.Tensor:
        """
        Differentiable fallback decimation when custom BPM decimation is disabled.
        """
        batch_size, _, feature_size = memory.shape
        output = memory.new_zeros(batch_size, self.note_target_length, feature_size)

        for n_batch in range(batch_size):
            valid_len = int(olens[n_batch].item())
            valid_len = max(1, min(valid_len, memory.size(1)))
            y = memory[n_batch : n_batch + 1, :valid_len].transpose(1, 2)
            y = F.interpolate(
                y,
                size=self.note_target_length,
                mode="linear",
                align_corners=False,
            )
            output[n_batch] = y.transpose(1, 2).squeeze(0)

        return output

    def forward(
        self,
        src_pad: torch.Tensor,
        src_len: torch.Tensor,
        note_len: torch.Tensor,
        bpm: torch.Tensor,
        frame_hand_pos: Optional[torch.Tensor] = None,
        note_hand_pos: Optional[torch.Tensor] = None,
    ):
        batch_size = src_pad.shape[0]
        raw_features = src_pad

        if self.use_conv_stack:
            encoder_in = self.convstack(src_pad.unsqueeze(1))
        else:
            encoder_in = src_pad

        memory, olens, _ = self.encoder(encoder_in, src_len)
        audio_memory = memory

        # Frame/tab stream may use hand-fused hidden memory.
        frame_memory = audio_memory
        if self.use_hand_position and self.frame_hand_fusion is not None and frame_hand_pos is not None:
            frame_memory = self.frame_hand_fusion(frame_memory, frame_hand_pos)

        frame_tab_pred = self.frame_tab_output_layer(frame_memory)
        frame_tab_pred = frame_tab_pred.view(batch_size, -1, 6, 21)
        frame_tab_pred = self.softmax_by_string(frame_tab_pred)

        if self.use_hand_position and self.hand_prior is not None and frame_hand_pos is not None:
            frame_tab_pred = apply_hand_position_prior(
                frame_tab_pred,
                frame_hand_pos,
                self.hand_prior,
                strength=self.hand_prior_strength,
            )

        # Onset branch uses audio-only encoder memory + optional raw features.
        onset_input = self._build_onset_input(audio_memory, raw_features)

        frame_onset_logits, string_onset_features = self.frame_onset_output_layer(
            onset_input,
            return_features=True,
        )
        global_onset_logits, global_onset_features = self.global_onset_output_layer(
            onset_input,
            return_features=True,
        )
        global_onset_logits = global_onset_logits.squeeze(-1)

        # Note stream starts from audio-only memory. Do NOT wrap this in no_grad:
        # note loss must be able to train the audio encoder.
        if self.use_custom_decimation_func:
            decimated_memory = self.notelevel_decimation(audio_memory, bpm)
        else:
            decimated_memory = self._interpolate_to_note_grid(audio_memory, olens)

        # Hidden onset conditioning. No thresholding, no sigmoid scores.
        onset_features = torch.cat([string_onset_features, global_onset_features], dim=-1)
        if self.detach_onset_features_for_note:
            onset_features = onset_features.detach()

        onset_note_features = self.decimate_sequence_to_note_grid(
            onset_features,
            bpm,
            target_length=decimated_memory.size(1),
        )

        if self.note_onset_conditioning is not None:
            decimated_memory = self.note_onset_conditioning(
                decimated_memory,
                onset_note_features,
            )

        # Optional note hidden hand fusion is disabled by default.
        if self.use_hand_position and self.note_hand_fusion is not None and note_hand_pos is not None:
            decimated_memory = self.note_hand_fusion(decimated_memory, note_hand_pos)

        note_lens_for_encoder = torch.clamp(note_len.to(device=decimated_memory.device), max=decimated_memory.size(1))
        decimated_memory, _, _ = self.note_encoder(decimated_memory, note_lens_for_encoder)

        note_tab_pred = self.note_tab_output_layer(decimated_memory)
        note_tab_pred = note_tab_pred.view(batch_size, -1, 6, 21)
        note_tab_pred = self.softmax_by_string(note_tab_pred)

        if self.use_hand_position and self.hand_prior is not None and note_hand_pos is not None:
            if self.note_rest_preserving_prior:
                note_tab_pred = apply_hand_position_prior_rest_preserving(
                    note_tab_pred,
                    note_hand_pos,
                    self.hand_prior,
                    strength=self.hand_prior_strength,
                    rest_class=REST_CLASS,
                )
            else:
                note_tab_pred = apply_hand_position_prior(
                    note_tab_pred,
                    note_hand_pos,
                    self.hand_prior,
                    strength=self.hand_prior_strength,
                )

        return (
            frame_tab_pred,
            note_tab_pred,
            frame_onset_logits,
            global_onset_logits,
            olens,
        )

    def notelevel_decimation(self, memory: torch.Tensor, bpm: torch.Tensor) -> torch.Tensor:
        """
        BPM-aware decimation from frame grid to note grid.

        Works for the encoder hidden dimension and keeps gradients to memory.
        """
        return self.decimate_sequence_to_note_grid(
            memory,
            bpm,
            target_length=self.note_target_length,
        )

    def decimate_sequence_to_note_grid(
        self,
        sequence: torch.Tensor,
        bpm: torch.Tensor,
        target_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        BPM-aware decimation for arbitrary hidden feature dimensions.

        sequence shape: (B, T_frame, D)
        output shape:   (B, target_length, D)
        """
        target_length = int(target_length) if target_length is not None else self.note_target_length
        padded_sequence = F.pad(sequence, (0, 0, 0, 10))
        batch_size = sequence.shape[0]
        feature_size = sequence.shape[2]
        max_frame_idx = padded_sequence.shape[1] - 1

        output = sequence.new_zeros(batch_size, target_length, feature_size)

        bpm = bpm.to(device=sequence.device, dtype=sequence.dtype)

        for n_batch in range(batch_size):
            frames_per_note = (float(self.sr) * 60.0) / (float(self.hop_length) * 4.0 * bpm[n_batch].clamp_min(1e-6))

            for n_note in range(target_length):
                frame_start = n_note * frames_per_note
                frame_end = (n_note + 1) * frames_per_note

                start_floor = int(torch.floor(frame_start).detach().item())
                start_ceil = int(torch.ceil(frame_start).detach().item())
                end_floor = int(torch.floor(frame_end).detach().item())

                start_floor = max(0, min(start_floor, max_frame_idx))
                start_ceil = max(0, min(start_ceil, max_frame_idx))
                end_floor = max(0, min(end_floor, max_frame_idx))
                if end_floor < start_ceil:
                    end_floor = start_ceil

                start_weight = torch.clamp(start_ceil - frame_start, min=0.0, max=1.0)
                end_weight = torch.clamp(frame_end - end_floor, min=0.0, max=1.0)

                sum_feat = padded_sequence[n_batch, start_floor, :] * start_weight

                if end_floor > start_ceil:
                    sum_feat = sum_feat + torch.sum(
                        padded_sequence[n_batch, start_ceil:end_floor, :],
                        dim=0,
                    )

                sum_feat = sum_feat + padded_sequence[n_batch, end_floor, :] * end_weight
                mean_feat = sum_feat / torch.clamp(frames_per_note, min=1e-6)
                output[n_batch, n_note] = mean_feat

        return output
