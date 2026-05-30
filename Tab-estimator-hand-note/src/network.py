#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
network.py

Tab-Estimator-Hand-Onset + chord-aware event assembly head.

This variant keeps the BPM-free frame-level design:
  - frame_tab_pred:          (B, T, 6, 21)
  - frame_onset_logits:     (B, T, 6)
  - global_onset_logits:    (B, T)

and adds a learned event assembly head:
  - event_logits:           (B, T)
  - event_string_logits:    (B, T, 6)
  - event_fret_logits:      (B, T, 6, 21)
  - event_type_logits:      (B, T, 4)

The event assembly head is non-causal and temporal. It receives shared encoder
states, frame-tab hidden evidence, and onset-TCN hidden evidence. It is meant to
learn the symbolic assembly step that hand-written decoders struggle with:
single notes vs chords vs arpeggios.

No causal convolutions are used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask


REST_CLASS = 20
EVENT_TYPE_NONE = 0
EVENT_TYPE_SINGLE = 1
EVENT_TYPE_CHORD = 2
EVENT_TYPE_ARPEGGIO = 3
N_EVENT_TYPES = 4


# -----------------------------------------------------------------------------
# Audio frontend
# -----------------------------------------------------------------------------


class ConvStack(nn.Module):
    def __init__(self, input_features, output_features, input_ch):
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

    def forward(self, x):
        y = self.cnn(x)
        y = y.transpose(1, 2).flatten(-2)
        y = self.fc(y)
        return y


# -----------------------------------------------------------------------------
# Optional frame-level hand-position conditioning
# -----------------------------------------------------------------------------


class SoftHandPositionFusion(nn.Module):
    """
    Add soft hand-position information into frame-level hidden states.

    hidden:   (B, T, hidden_dim)
    hand_pos: (B, T_hand, hand_pos_dim)
    """

    def __init__(self, hand_pos_dim, hidden_dim, dropout=0.1, gate_init=0.5):
        super().__init__()
        self.hand_pos_dim = int(hand_pos_dim)
        self.hidden_dim = int(hidden_dim)

        self.proj = nn.Sequential(
            nn.Linear(self.hand_pos_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(dropout),
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))

    @staticmethod
    def resize_time(x, target_len):
        if x is None:
            return None
        if x.size(1) == target_len:
            return x
        x = x.transpose(1, 2)
        x = F.interpolate(x, size=int(target_len), mode="linear", align_corners=False)
        x = x.transpose(1, 2)
        return x

    def forward(self, hidden, hand_pos):
        if hand_pos is None:
            return hidden
        if hand_pos.dim() == 2:
            hand_pos = hand_pos.unsqueeze(0)
        if hand_pos.dim() != 3:
            raise ValueError(
                f"Expected hand_pos shape (B, T, {self.hand_pos_dim}), got {tuple(hand_pos.shape)}"
            )
        if hand_pos.size(-1) != self.hand_pos_dim:
            raise ValueError(
                f"Expected hand_pos_dim={self.hand_pos_dim}, got {hand_pos.size(-1)}"
            )

        hand_pos = hand_pos.to(device=hidden.device, dtype=hidden.dtype)
        hand_pos = self.resize_time(hand_pos, hidden.size(1))
        return hidden + self.gate * self.proj(hand_pos)


class HandPositionPrior(nn.Module):
    """
    Convert soft hand-position distribution into a fret prior.

    hand_pos: (B, T, n_positions)
    prior:    (B, T, 6, 21)
    """

    def __init__(
        self,
        n_positions=20,
        n_strings=6,
        n_tab_classes=21,
        hand_span=4,
        rest_class=REST_CLASS,
        open_weight=0.55,
        rest_weight=0.50,
        playable_weight=1.00,
        floor=1e-5,
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
        self.register_buffer("prior_table", self._build_prior_table(), persistent=False)

    def _build_prior_table(self):
        table = torch.full((self.n_positions, self.n_tab_classes), self.floor, dtype=torch.float32)

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
    def resize_time(x, target_len):
        if x.size(1) == target_len:
            return x
        x = x.transpose(1, 2)
        x = F.interpolate(x, size=int(target_len), mode="linear", align_corners=False)
        x = x.transpose(1, 2)
        return x

    def forward(self, hand_pos, target_len=None):
        if hand_pos is None:
            return None
        if hand_pos.dim() == 2:
            hand_pos = hand_pos.unsqueeze(0)
        if hand_pos.size(-1) != self.n_positions:
            raise ValueError(
                f"Expected hand_pos last dim {self.n_positions}, got {hand_pos.size(-1)}"
            )

        hand_pos = hand_pos.float()
        if target_len is not None:
            hand_pos = self.resize_time(hand_pos, target_len)

        table = self.prior_table.to(device=hand_pos.device, dtype=hand_pos.dtype)
        fret_prior = torch.matmul(hand_pos, table)
        fret_prior = fret_prior.unsqueeze(2).repeat(1, 1, self.n_strings, 1)
        fret_prior = fret_prior / fret_prior.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return fret_prior


def apply_hand_position_prior(tab_probs, hand_pos, prior_module, strength=0.0):
    if hand_pos is None or strength <= 0.0:
        return tab_probs
    prior = prior_module(hand_pos, target_len=tab_probs.size(1))
    prior = prior.to(device=tab_probs.device, dtype=tab_probs.dtype)
    adjusted = tab_probs * torch.pow(prior.clamp_min(1e-8), float(strength))
    adjusted = adjusted / adjusted.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return adjusted


# -----------------------------------------------------------------------------
# Non-causal gated TCN blocks
# -----------------------------------------------------------------------------


class GatedTemporalBlock(nn.Module):
    """
    Non-causal gated residual temporal block.

    Input/output layout:
        x: (B, C, T)

    Symmetric padding is used. This is intentionally non-causal.
    """

    def __init__(self, n_inputs, n_outputs, kernel_size=5, dilation=1, dropout=0.25):
        super().__init__()

        kernel_size = int(kernel_size)
        dilation = int(dilation)

        if kernel_size % 2 == 0:
            raise ValueError("Use an odd TCN kernel_size to preserve sequence length.")

        padding = (kernel_size // 2) * dilation

        self.filter_conv = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
            )
        )
        self.gate_conv = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
            )
        )
        self.dropout = nn.Dropout(float(dropout))
        self.downsample = nn.Conv1d(n_inputs, n_outputs, kernel_size=1) if n_inputs != n_outputs else None
        self.out_activation = nn.ReLU()
        self.init_weights()

    def init_weights(self):
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

    def forward(self, x):
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
    """
    Non-causal TCN onset head.

    Returns raw logits. Can also return penultimate hidden features.
    """

    def __init__(
        self,
        input_size,
        output_size=6,
        num_channels=(64, 64, 64, 64),
        kernel_size=5,
        dropout=0.25,
    ):
        super().__init__()

        if isinstance(num_channels, int):
            num_channels = (int(num_channels),)
        num_channels = tuple(int(c) for c in num_channels)
        if len(num_channels) < 4:
            raise ValueError("OnsetTCNHead must use at least 4 non-causal TCN blocks.")

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

    def forward(self, hidden, return_features=False):
        z = hidden.transpose(1, 2)
        z = self.tcn(z)
        z = z.transpose(1, 2)
        logits = self.output(z)
        if return_features:
            return logits, z
        return logits


class ChordAwareEventAssemblyHead(nn.Module):
    """
    Learned note-event assembly head.

    This head learns the decision normally made by threshold rules:
      - event existence
      - selected strings
      - fret per selected string
      - event type: none / single / chord / arpeggio context

    It is a non-causal temporal TCN with at least 4 layers.
    """

    def __init__(
        self,
        input_size,
        hidden_dim=128,
        num_layers=6,
        kernel_size=5,
        dropout=0.25,
        n_strings=6,
        n_tab_classes=21,
        n_event_types=N_EVENT_TYPES,
    ):
        super().__init__()

        num_layers = int(num_layers)
        if num_layers < 4:
            raise ValueError("ChordAwareEventAssemblyHead must use at least 4 non-causal TCN blocks.")

        hidden_dim = int(hidden_dim)
        layers = []
        for i in range(num_layers):
            in_channels = int(input_size) if i == 0 else hidden_dim
            dilation = 2 ** i
            layers.append(
                GatedTemporalBlock(
                    n_inputs=in_channels,
                    n_outputs=hidden_dim,
                    kernel_size=int(kernel_size),
                    dilation=dilation,
                    dropout=float(dropout),
                )
            )

        self.tcn = nn.Sequential(*layers)
        self.event_output = nn.Linear(hidden_dim, 1)
        self.string_output = nn.Linear(hidden_dim, int(n_strings))
        self.fret_output = nn.Linear(hidden_dim, int(n_strings) * int(n_tab_classes))
        self.type_output = nn.Linear(hidden_dim, int(n_event_types))
        self.n_strings = int(n_strings)
        self.n_tab_classes = int(n_tab_classes)

    def forward(self, x, return_features=False):
        z = x.transpose(1, 2)
        z = self.tcn(z)
        z = z.transpose(1, 2)

        event_logits = self.event_output(z).squeeze(-1)
        string_logits = self.string_output(z)
        fret_logits = self.fret_output(z).view(z.size(0), z.size(1), self.n_strings, self.n_tab_classes)
        type_logits = self.type_output(z)

        if return_features:
            return event_logits, string_logits, fret_logits, type_logits, z

        return event_logits, string_logits, fret_logits, type_logits


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------


class CustomLoss(nn.Module):
    """
    BPM-free frame-level + onset + chord-aware event assembly loss.
    """

    def __init__(
        self,
        onset_loss_weight=0.25,
        onset_positive_weight=10.0,
        global_onset_loss_weight=0.25,
        global_onset_positive_weight=10.0,
        tab_loss_weight=1.0,
        event_loss_weight=0.50,
        event_positive_weight=10.0,
        event_string_loss_weight=0.50,
        event_string_positive_weight=10.0,
        event_fret_loss_weight=1.0,
        event_type_loss_weight=0.10,
    ):
        super().__init__()
        self.onset_loss_weight = float(onset_loss_weight)
        self.onset_positive_weight = float(onset_positive_weight)
        self.global_onset_loss_weight = float(global_onset_loss_weight)
        self.global_onset_positive_weight = float(global_onset_positive_weight)
        self.tab_loss_weight = float(tab_loss_weight)
        self.event_loss_weight = float(event_loss_weight)
        self.event_positive_weight = float(event_positive_weight)
        self.event_string_loss_weight = float(event_string_loss_weight)
        self.event_string_positive_weight = float(event_string_positive_weight)
        self.event_fret_loss_weight = float(event_fret_loss_weight)
        self.event_type_loss_weight = float(event_type_loss_weight)

    @staticmethod
    def _resize_2d_target(x, target_len):
        if x.size(1) == target_len:
            return x
        x = x.transpose(1, 2)
        x = F.interpolate(x, size=target_len, mode="nearest")
        x = x.transpose(1, 2)
        return x

    @staticmethod
    def _resize_3d_target(x, target_len):
        if x.size(1) == target_len:
            return x
        x = x.permute(0, 2, 3, 1)
        x = F.interpolate(x, size=target_len, mode="nearest")
        x = x.permute(0, 3, 1, 2)
        return x

    def forward(
        self,
        frame_tab_pred,
        frame_tab_gt,
        frame_onset_pred,
        global_onset_pred,
        frame_onset_gt,
        event_logits,
        event_string_logits,
        event_fret_logits,
        event_type_logits,
        event_type_gt,
        olens,
    ):
        frame_tab_gt = frame_tab_gt.to(device=frame_tab_pred.device, dtype=frame_tab_pred.dtype)
        frame_onset_gt = frame_onset_gt.to(device=frame_onset_pred.device, dtype=frame_onset_pred.dtype)
        event_type_gt = event_type_gt.to(device=frame_tab_pred.device, dtype=torch.long)

        T = int(frame_tab_pred.size(1))

        frame_tab_gt = self._resize_3d_target(frame_tab_gt, T)
        frame_onset_gt = self._resize_2d_target(frame_onset_gt, T)

        if event_type_gt.size(1) != T:
            event_type_float = event_type_gt.float().unsqueeze(1)
            event_type_gt = F.interpolate(event_type_float, size=T, mode="nearest").squeeze(1).long()

        if global_onset_pred.dim() == 3 and global_onset_pred.size(-1) == 1:
            global_onset_pred = global_onset_pred.squeeze(-1)

        olens = olens.to(device=frame_tab_pred.device)
        frame_mask = make_non_pad_mask(olens).to(frame_tab_pred.device)

        # Frame tab loss.
        tab_loss = -frame_tab_gt * torch.log(frame_tab_pred.clamp_min(1e-7))
        tab_loss = tab_loss * frame_mask[:, :, None, None]
        tab_denom = frame_mask.sum().clamp_min(1).float() * 6.0
        tab_loss = tab_loss.sum() / tab_denom

        # Per-string onset auxiliary loss.
        pos_weight = torch.tensor(
            float(self.onset_positive_weight),
            device=frame_onset_pred.device,
            dtype=frame_onset_pred.dtype,
        )
        onset_loss = F.binary_cross_entropy_with_logits(
            frame_onset_pred,
            frame_onset_gt,
            pos_weight=pos_weight,
            reduction="none",
        )
        onset_loss = onset_loss * frame_mask[:, :, None]
        onset_denom = frame_mask.sum().clamp_min(1).float() * 6.0
        onset_loss = onset_loss.sum() / onset_denom

        # Global onset auxiliary loss.
        global_onset_gt = torch.max(frame_onset_gt, dim=2).values
        global_pos_weight = torch.tensor(
            float(self.global_onset_positive_weight),
            device=global_onset_pred.device,
            dtype=global_onset_pred.dtype,
        )
        global_onset_loss = F.binary_cross_entropy_with_logits(
            global_onset_pred,
            global_onset_gt,
            pos_weight=global_pos_weight,
            reduction="none",
        )
        global_onset_loss = global_onset_loss * frame_mask
        global_onset_loss = global_onset_loss.sum() / frame_mask.sum().clamp_min(1).float()

        # Event existence loss.
        event_gt = global_onset_gt
        event_pos_weight = torch.tensor(
            float(self.event_positive_weight),
            device=event_logits.device,
            dtype=event_logits.dtype,
        )
        event_loss = F.binary_cross_entropy_with_logits(
            event_logits,
            event_gt,
            pos_weight=event_pos_weight,
            reduction="none",
        )
        event_loss = event_loss * frame_mask
        event_loss = event_loss.sum() / frame_mask.sum().clamp_min(1).float()

        # Event string mask loss.
        event_string_pos_weight = torch.tensor(
            float(self.event_string_positive_weight),
            device=event_string_logits.device,
            dtype=event_string_logits.dtype,
        )
        event_string_loss = F.binary_cross_entropy_with_logits(
            event_string_logits,
            frame_onset_gt,
            pos_weight=event_string_pos_weight,
            reduction="none",
        )
        event_string_loss = event_string_loss * frame_mask[:, :, None]
        event_string_loss = event_string_loss.sum() / (frame_mask.sum().clamp_min(1).float() * 6.0)

        # Event fret loss only on strings that have an event onset.
        log_fret = F.log_softmax(event_fret_logits, dim=-1)
        event_fret_ce = -frame_tab_gt * log_fret
        event_fret_ce = event_fret_ce.sum(dim=-1)
        event_string_mask = (frame_onset_gt > 0).to(event_fret_ce.dtype) * frame_mask[:, :, None].to(event_fret_ce.dtype)
        event_fret_loss = (event_fret_ce * event_string_mask).sum() / event_string_mask.sum().clamp_min(1.0)

        # Event type loss. This auxiliary loss helps distinguish single/chord/arpeggio contexts.
        event_type_loss = F.cross_entropy(
            event_type_logits.transpose(1, 2),
            event_type_gt,
            reduction="none",
        )
        event_type_loss = event_type_loss * frame_mask
        event_type_loss = event_type_loss.sum() / frame_mask.sum().clamp_min(1).float()

        total = (
            self.tab_loss_weight * tab_loss
            + self.onset_loss_weight * onset_loss
            + self.global_onset_loss_weight * global_onset_loss
            + self.event_loss_weight * event_loss
            + self.event_string_loss_weight * event_string_loss
            + self.event_fret_loss_weight * event_fret_loss
            + self.event_type_loss_weight * event_type_loss
        )

        return total


# -----------------------------------------------------------------------------
# TabEstimator
# -----------------------------------------------------------------------------


class TabEstimator(torch.nn.Module):
    """
    BPM-free frame-level tablature + onset + event assembly estimator.

    Forward:
        (
            frame_tab_pred,
            frame_onset_logits,
            global_onset_logits,
            event_logits,
            event_string_logits,
            event_fret_logits,
            event_type_logits,
            olens,
        ) = model(src_pad, src_len, frame_hand_pos=None)

    src_pad: (B, T_frame, n_bins)
    src_len: (B,)
    """

    def __init__(
        self,
        mode,
        encoder_type,
        use_custom_decimation_func,
        use_conv_stack,
        n_bins,
        hop_length,
        sr,
        encoder_heads=1,
        encoder_layers=1,
        normalize_before=True,
        use_hand_position=False,
        hand_pos_dim=20,
        hand_position_fusion="hidden+prior",
        hand_hidden_gate_init=0.5,
        hand_prior_strength=0.35,
        hand_span=4,
        onset_hidden_dim=64,
        onset_dropout=0.25,
        onset_kernel_size=3,
        onset_tcn_levels=4,
        onset_use_raw_features=True,
        onset_raw_proj_dim=64,
        onset_raw_dropout=0.10,
        onset_input_mode="full",
        event_head_hidden_dim=128,
        event_head_tcn_levels=6,
        event_head_kernel_size=5,
        event_head_dropout=0.25,
        event_input_uses_tab_probs=True,
        detach_tab_probs_for_event=False,
        **unused_kwargs,
    ):
        super().__init__()

        if mode != "tab":
            raise ValueError("This BPM-free network supports mode='tab' only.")

        self.mode = "tab"
        self.use_conv_stack = bool(use_conv_stack)
        self.use_custom_decimation_func = False
        self.hop_length = int(hop_length)
        self.sr = int(sr)

        self.encoder_output_size = 64
        self.n_encoder_ffn = 64
        self.encoder_attn_dropout = 0.0
        self.encoder_pos_dropout = 0.1
        self.conv_output_features = 16 * 32

        self.use_hand_position = bool(use_hand_position)
        self.hand_pos_dim = int(hand_pos_dim)
        self.hand_position_fusion = str(hand_position_fusion)
        self.hand_prior_strength = float(hand_prior_strength)
        self.hand_span = int(hand_span)

        self.onset_use_raw_features = bool(onset_use_raw_features)
        self.onset_raw_proj_dim = int(onset_raw_proj_dim)
        self.onset_raw_dropout = float(onset_raw_dropout)
        self.onset_input_mode = str(onset_input_mode)
        if self.onset_input_mode not in ["full", "encoder", "raw"]:
            raise ValueError("onset_input_mode must be one of: full, encoder, raw")

        self.event_input_uses_tab_probs = bool(event_input_uses_tab_probs)
        self.detach_tab_probs_for_event = bool(detach_tab_probs_for_event)

        if self.hand_position_fusion not in ["hidden", "prior", "hidden+prior", "none"]:
            raise ValueError(
                "hand_position_fusion must be one of: 'hidden', 'prior', 'hidden+prior', 'none'"
            )

        if int(onset_tcn_levels) < 4:
            raise ValueError("onset_tcn_levels must be at least 4.")
        if int(event_head_tcn_levels) < 4:
            raise ValueError("event_head_tcn_levels must be at least 4.")
        if int(onset_kernel_size) % 2 == 0 or int(event_head_kernel_size) % 2 == 0:
            raise ValueError("TCN kernel sizes must be odd for non-causal same-length padding.")

        if self.use_conv_stack:
            self.convstack = ConvStack(n_bins, self.conv_output_features, 1)

        encoder_input_size = self.conv_output_features if self.use_conv_stack else n_bins

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

        self.frame_tab_feature_layer = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.encoder_output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.frame_tab_output_layer = nn.Linear(128, 6 * 21)
        self.softmax_by_string = nn.Softmax(dim=3)

        # Raw feature projection for onset/event evidence.
        if self.onset_use_raw_features or self.onset_input_mode in ["full", "raw"]:
            self.onset_raw_feature_proj = nn.Sequential(
                nn.Linear(int(n_bins), self.onset_raw_proj_dim),
                nn.LayerNorm(self.onset_raw_proj_dim),
                nn.Dropout(self.onset_raw_dropout),
            )
        else:
            self.onset_raw_feature_proj = None

        if self.onset_input_mode == "encoder":
            onset_input_size = self.encoder_output_size
        elif self.onset_input_mode == "raw":
            if self.onset_raw_feature_proj is None:
                raise ValueError("onset_input_mode='raw' requires raw feature projection.")
            onset_input_size = self.onset_raw_proj_dim
        else:
            if self.onset_raw_feature_proj is None:
                raise ValueError("onset_input_mode='full' requires raw feature projection.")
            onset_input_size = self.encoder_output_size + self.onset_raw_proj_dim

        onset_channels = tuple([int(onset_hidden_dim)] * int(onset_tcn_levels))
        self.frame_onset_output_layer = OnsetTCNHead(
            input_size=onset_input_size,
            output_size=6,
            num_channels=onset_channels,
            kernel_size=int(onset_kernel_size),
            dropout=float(onset_dropout),
        )
        self.global_onset_output_layer = OnsetTCNHead(
            input_size=onset_input_size,
            output_size=1,
            num_channels=onset_channels,
            kernel_size=int(onset_kernel_size),
            dropout=float(onset_dropout),
        )

        event_input_size = (
            self.encoder_output_size
            + 128
            + int(onset_hidden_dim)
            + int(onset_hidden_dim)
        )
        if self.event_input_uses_tab_probs:
            event_input_size += 6 * 21

        self.event_assembly_head = ChordAwareEventAssemblyHead(
            input_size=event_input_size,
            hidden_dim=int(event_head_hidden_dim),
            num_layers=int(event_head_tcn_levels),
            kernel_size=int(event_head_kernel_size),
            dropout=float(event_head_dropout),
            n_strings=6,
            n_tab_classes=21,
            n_event_types=N_EVENT_TYPES,
        )

        if self.use_hand_position:
            if self.hand_position_fusion in ["hidden", "hidden+prior"]:
                self.frame_hand_fusion = SoftHandPositionFusion(
                    hand_pos_dim=self.hand_pos_dim,
                    hidden_dim=self.encoder_output_size,
                    dropout=0.1,
                    gate_init=hand_hidden_gate_init,
                )
            else:
                self.frame_hand_fusion = None

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
            self.hand_prior = None

    @staticmethod
    def _resize_time(x, target_len):
        if x.size(1) == target_len:
            return x
        x = x.transpose(1, 2)
        x = F.interpolate(x, size=int(target_len), mode="linear", align_corners=False)
        x = x.transpose(1, 2)
        return x

    def _project_raw_features(self, raw_features, target_len, dtype, device):
        if self.onset_raw_feature_proj is None:
            return None
        raw_proj = self.onset_raw_feature_proj(raw_features.to(device=device, dtype=dtype))
        raw_proj = self._resize_time(raw_proj, target_len)
        return raw_proj

    def _build_onset_input(self, audio_memory, raw_proj):
        if self.onset_input_mode == "encoder":
            return audio_memory
        if self.onset_input_mode == "raw":
            if raw_proj is None:
                raise RuntimeError("raw onset features are required but unavailable.")
            return raw_proj
        if raw_proj is None:
            raise RuntimeError("full onset input requires raw projected features.")
        return torch.cat([audio_memory, raw_proj], dim=-1)

    def forward(self, src_pad, src_len, frame_hand_pos=None):
        batch_size = src_pad.shape[0]
        raw_features = src_pad

        if self.use_conv_stack:
            encoder_in = self.convstack(torch.unsqueeze(src_pad, dim=1))
        else:
            encoder_in = src_pad

        audio_memory, olens, _ = self.encoder(encoder_in, src_len)

        # The onset branch remains audio-dominant; frame hand fusion is applied
        # only to the frame/tab evidence path.
        frame_memory = audio_memory
        if self.use_hand_position and self.frame_hand_fusion is not None and frame_hand_pos is not None:
            frame_memory = self.frame_hand_fusion(frame_memory, frame_hand_pos)

        frame_tab_hidden = self.frame_tab_feature_layer(frame_memory)
        frame_tab_logits = self.frame_tab_output_layer(frame_tab_hidden)
        frame_tab_pred = frame_tab_logits.view(batch_size, -1, 6, 21)
        frame_tab_pred = self.softmax_by_string(frame_tab_pred)

        if self.use_hand_position and self.hand_prior is not None and frame_hand_pos is not None:
            frame_tab_pred = apply_hand_position_prior(
                frame_tab_pred,
                frame_hand_pos,
                self.hand_prior,
                strength=self.hand_prior_strength,
            )

        raw_proj = self._project_raw_features(
            raw_features,
            target_len=audio_memory.size(1),
            dtype=audio_memory.dtype,
            device=audio_memory.device,
        )

        onset_input = self._build_onset_input(audio_memory, raw_proj)

        frame_onset_logits, string_onset_features = self.frame_onset_output_layer(
            onset_input,
            return_features=True,
        )
        global_onset_raw, global_onset_features = self.global_onset_output_layer(
            onset_input,
            return_features=True,
        )
        global_onset_logits = global_onset_raw.squeeze(-1)

        event_inputs = [
            frame_memory,
            frame_tab_hidden,
            string_onset_features,
            global_onset_features,
        ]
        if self.event_input_uses_tab_probs:
            tab_probs_for_event = frame_tab_pred
            if self.detach_tab_probs_for_event:
                tab_probs_for_event = tab_probs_for_event.detach()
            event_inputs.append(tab_probs_for_event.reshape(batch_size, frame_tab_pred.size(1), 6 * 21))

        event_input = torch.cat(event_inputs, dim=-1)

        event_logits, event_string_logits, event_fret_logits, event_type_logits = self.event_assembly_head(
            event_input,
            return_features=False,
        )

        return (
            frame_tab_pred,
            frame_onset_logits,
            global_onset_logits,
            event_logits,
            event_string_logits,
            event_fret_logits,
            event_type_logits,
            olens,
        )
