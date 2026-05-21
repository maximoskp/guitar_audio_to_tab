#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
network.py

Audio-only onset detection network.

This is a simplified version of Tab-Estimator-Hand-Onset that focuses only
on predicting audio-based onsets without tablature estimation or hand-position
input.

The model predicts:
  - frame_onset_logits:  (B, T_frame, 6), raw per-string onset logits
  - global_onset_logits: (B, T_frame), raw global/no-string onset logits

The onset heads receive both encoder states and an optional learned projection
of the raw input features, so transient CQT/mel information is available.

The intended runtime decoder is:
  sigmoid(global_onset_logits) gives global note-start timing;
  sigmoid(frame_onset_logits) gives optional per-string onset support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask


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
# Non-causal gated TCN onset head
# -----------------------------------------------------------------------------


class GatedTemporalBlock(nn.Module):
    """
    Non-causal gated residual temporal block.

    Input/output layout:
        x: (B, C, T)

    This block uses symmetric padding, so it can look slightly before and after
    the current frame. That is intentional for offline transcription and onset
    localization. It is not causal.
    """

    def __init__(self, n_inputs, n_outputs, kernel_size=5, dilation=1, dropout=0.25):
        super().__init__()

        kernel_size = int(kernel_size)
        dilation = int(dilation)

        if kernel_size % 2 == 0:
            raise ValueError("Use an odd onset TCN kernel_size to preserve sequence length.")

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
        # x: (B, C, T)
        filt = torch.tanh(self.filter_conv(x))
        gate = torch.sigmoid(self.gate_conv(x))
        out = self.dropout(filt * gate)

        res = x if self.downsample is None else self.downsample(x)

        # Protect against future changes to padding/kernel configuration.
        if out.shape[-1] != res.shape[-1]:
            min_len = min(out.shape[-1], res.shape[-1])
            out = out[..., :min_len]
            res = res[..., :min_len]

        return self.out_activation(out + res)


class OnsetTCNHead(nn.Module):
    """
    Non-causal TCN onset head.

    Input:
        hidden: (B, T, D)

    Output:
        logits: (B, T, output_size)

    The output is raw logits, not sigmoid probabilities. Train with
    BCEWithLogitsLoss. Apply torch.sigmoid(logits) only for thresholding during
    inference.
    """

    def __init__(
        self,
        input_size,
        output_size=6,
        num_channels=(64, 64, 64),
        kernel_size=5,
        dropout=0.25,
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

    def forward(self, hidden):
        # hidden: (B, T, D)
        z = hidden.transpose(1, 2)       # (B, D, T)
        z = self.tcn(z)                  # (B, C, T)
        z = z.transpose(1, 2)            # (B, T, C)
        return self.output(z)            # raw logits, (B, T, output_size)


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------


class OnsetOnlyLoss(nn.Module):
    """
    Audio-only onset detection loss.

    frame_onset_pred:        (B, T, 6), raw per-string onset logits
    global_onset_pred:       (B, T), raw global/no-string onset logits
    frame_onset_gt:          (B, T, 6), binary or soft widened labels
    olens:                   (B,), valid frame lengths after encoder

    The global target is derived as max(frame_onset_gt, string_dim). If training
    labels have been widened, the global labels are widened too.
    """

    def __init__(
        self,
        onset_loss_weight=0.25,
        onset_positive_weight=10.0,
        global_onset_loss_weight=0.25,
        global_onset_positive_weight=10.0,
    ):
        super().__init__()
        self.onset_loss_weight = float(onset_loss_weight)
        self.onset_positive_weight = float(onset_positive_weight)
        self.global_onset_loss_weight = float(global_onset_loss_weight)
        self.global_onset_positive_weight = float(global_onset_positive_weight)

    def forward(
        self,
        frame_onset_pred,
        global_onset_pred,
        frame_onset_gt,
        olens,
    ):
        frame_onset_gt = frame_onset_gt.to(device=frame_onset_pred.device, dtype=frame_onset_pred.dtype)

        if frame_onset_gt.size(1) != frame_onset_pred.size(1):
            frame_onset_gt = frame_onset_gt.transpose(1, 2)
            frame_onset_gt = F.interpolate(frame_onset_gt, size=frame_onset_pred.size(1), mode="nearest")
            frame_onset_gt = frame_onset_gt.transpose(1, 2)

        if global_onset_pred.dim() == 3 and global_onset_pred.size(-1) == 1:
            global_onset_pred = global_onset_pred.squeeze(-1)

        olens = olens.to(device=frame_onset_pred.device)
        frame_mask = make_non_pad_mask(olens).to(frame_onset_pred.device)

        # Per-string onset loss.
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

        # Global onset loss. The target is active if any string has an onset.
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
        global_denom = frame_mask.sum().clamp_min(1).float()
        global_onset_loss = global_onset_loss.sum() / global_denom

        return (
            self.onset_loss_weight * onset_loss
            + self.global_onset_loss_weight * global_onset_loss
        )


# -----------------------------------------------------------------------------
# Audio-only onset detector
# -----------------------------------------------------------------------------


class OnsetDetector(nn.Module):
    """
    Audio-only onset detection network.

    Forward signature:
        frame_onset_logits, global_onset_logits, olens = model(
            src_pad,
            src_len,
        )

    src_pad: (B, T_frame, n_bins)
    src_len: (B,)
    """

    def __init__(
        self,
        encoder_type,
        use_conv_stack,
        n_bins,
        hop_length,
        sr,
        encoder_heads=1,
        encoder_layers=1,
        normalize_before=True,
        onset_hidden_dim=64,
        onset_dropout=0.25,
        onset_kernel_size=3,
        onset_tcn_levels=4,
        onset_use_raw_features=True,
        onset_raw_proj_dim=64,
        onset_raw_dropout=0.10,
        onset_input_mode="full",
        **unused_kwargs,
    ):
        super().__init__()

        self.use_conv_stack = bool(use_conv_stack)
        self.hop_length = int(hop_length)
        self.sr = int(sr)
        self.n_bins = int(n_bins)

        self.encoder_output_size = 64
        self.n_encoder_ffn = 64
        self.encoder_attn_dropout = 0.0
        self.encoder_pos_dropout = 0.1
        self.conv_output_features = 16 * 32

        self.onset_use_raw_features = bool(onset_use_raw_features)
        self.onset_raw_proj_dim = int(onset_raw_proj_dim)
        self.onset_raw_dropout = float(onset_raw_dropout)
        self.onset_input_mode = str(onset_input_mode)

        if self.onset_input_mode not in ["full", "encoder", "raw"]:
            raise ValueError(
                f"onset_input_mode must be one of: 'full', 'encoder', 'raw', got {self.onset_input_mode}"
            )

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

        # Onset input projection
        if self.onset_use_raw_features:
            self.onset_raw_feature_proj = nn.Sequential(
                nn.Linear(int(n_bins), self.onset_raw_proj_dim),
                nn.LayerNorm(self.onset_raw_proj_dim),
                nn.Dropout(self.onset_raw_dropout),
            )
        else:
            self.onset_raw_feature_proj = None

        # Determine onset input size based on mode
        if self.onset_input_mode == "encoder":
            onset_input_size = self.encoder_output_size
        elif self.onset_input_mode == "raw":
            onset_input_size = self.onset_raw_proj_dim
        elif self.onset_input_mode == "full":
            onset_input_size = self.encoder_output_size + self.onset_raw_proj_dim
        else:
            raise ValueError(f"Unknown onset_input_mode: {self.onset_input_mode}")

        # Validate that we have the necessary components
        if self.onset_input_mode == "raw" and not self.onset_use_raw_features:
            raise ValueError("onset_input_mode='raw' requires onset_use_raw_features=True")

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

    def forward(self, src_pad, src_len):
        raw_features = src_pad

        if self.use_conv_stack:
            encoder_in = self.convstack(torch.unsqueeze(src_pad, dim=1))
        else:
            encoder_in = src_pad

        memory, olens, _ = self.encoder(encoder_in, src_len)

        # Build onset input based on mode
        if self.onset_input_mode == "encoder":
            onset_input = memory
        elif self.onset_input_mode == "raw":
            if self.onset_raw_feature_proj is None:
                raise RuntimeError("onset_input_mode='raw' but onset_raw_feature_proj is None")
            raw_proj = self.onset_raw_feature_proj(
                raw_features.to(device=memory.device, dtype=memory.dtype)
            )
            if raw_proj.size(1) != memory.size(1):
                raw_proj = raw_proj.transpose(1, 2)
                raw_proj = F.interpolate(
                    raw_proj,
                    size=int(memory.size(1)),
                    mode="linear",
                    align_corners=False,
                )
                raw_proj = raw_proj.transpose(1, 2)
            onset_input = raw_proj
        elif self.onset_input_mode == "full":
            if self.onset_raw_feature_proj is None:
                raise RuntimeError("onset_input_mode='full' but onset_raw_feature_proj is None")
            raw_proj = self.onset_raw_feature_proj(
                raw_features.to(device=memory.device, dtype=memory.dtype)
            )
            if raw_proj.size(1) != memory.size(1):
                raw_proj = raw_proj.transpose(1, 2)
                raw_proj = F.interpolate(
                    raw_proj,
                    size=int(memory.size(1)),
                    mode="linear",
                    align_corners=False,
                )
                raw_proj = raw_proj.transpose(1, 2)
            onset_input = torch.cat([memory, raw_proj], dim=-1)
        else:
            raise ValueError(f"Unknown onset_input_mode: {self.onset_input_mode}")

        frame_onset_logits = self.frame_onset_output_layer(onset_input)
        global_onset_logits = self.global_onset_output_layer(onset_input).squeeze(-1)

        return frame_onset_logits, global_onset_logits, olens
