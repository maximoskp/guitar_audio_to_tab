#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
network.py

BPM-free, frame-level TabEstimator variant.

This version intentionally removes the legacy beat-synchronous note-level path:
  - no bpm input
  - no note_len input
  - no note_pred output
  - no note_encoder
  - no note-level decimation

The model predicts directly on the audio frame timeline:
  - frame_tab_pred:   (B, T_frame, 6, 21)
  - frame_onset_logits: (B, T_frame, 6), raw onset logits

The intended runtime decoder is:
  sigmoid(frame_onset_logits) gives note-start probabilities;
  frame_tab_pred gives string/fret labels at those times;
  nearby onsets can be grouped into chords as post-processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask


REST_CLASS = 20


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

        # This should not be needed for odd kernels, but protects against future
        # changes to padding/kernel configuration.
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
        logits: (B, T, 6)

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
        return self.output(z)            # raw logits, (B, T, 6)

# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------


class CustomLoss(nn.Module):
    """
    BPM-free frame-level loss.

    frame_tab_pred:   (B, T, 6, 21), probabilities after softmax over frets
    frame_tab_gt:     (B, T, 6, 21), one-hot labels
    frame_onset_pred: (B, T, 6), raw logits from the onset TCN head
    frame_onset_gt:   (B, T, 6), binary labels
    olens:            (B,), valid frame lengths after encoder
    """

    def __init__(
        self,
        onset_loss_weight=0.25,
        onset_positive_weight=10.0,
        tab_loss_weight=1.0,
    ):
        super().__init__()
        self.onset_loss_weight = float(onset_loss_weight)
        self.onset_positive_weight = float(onset_positive_weight)
        self.tab_loss_weight = float(tab_loss_weight)

    @staticmethod
    def _resize_time_like(target, pred):
        if target.size(1) == pred.size(1):
            return target
        x = target.transpose(1, -1)
        x = F.interpolate(x, size=pred.size(1), mode="nearest")
        x = x.transpose(1, -1)
        return x

    def forward(self, frame_tab_pred, frame_tab_gt, frame_onset_pred, frame_onset_gt, olens):
        frame_tab_gt = frame_tab_gt.to(device=frame_tab_pred.device, dtype=frame_tab_pred.dtype)
        frame_onset_gt = frame_onset_gt.to(device=frame_onset_pred.device, dtype=frame_onset_pred.dtype)

        if frame_tab_gt.size(1) != frame_tab_pred.size(1):
            # Nearest resize is a fallback for unusual encoder length changes.
            frame_tab_gt = frame_tab_gt.permute(0, 2, 3, 1)
            frame_tab_gt = F.interpolate(frame_tab_gt, size=frame_tab_pred.size(1), mode="nearest")
            frame_tab_gt = frame_tab_gt.permute(0, 3, 1, 2)

        if frame_onset_gt.size(1) != frame_onset_pred.size(1):
            frame_onset_gt = frame_onset_gt.transpose(1, 2)
            frame_onset_gt = F.interpolate(frame_onset_gt, size=frame_onset_pred.size(1), mode="nearest")
            frame_onset_gt = frame_onset_gt.transpose(1, 2)

        olens = olens.to(device=frame_tab_pred.device)
        frame_mask = make_non_pad_mask(olens).to(frame_tab_pred.device)

        # Tab loss: original TabEstimator-style binary cross entropy over one-hot tab grid.
        tab_loss = -frame_tab_gt * torch.log(frame_tab_pred.clamp_min(1e-7))
        tab_loss = tab_loss * frame_mask[:, :, None, None]
        tab_denom = frame_mask.sum().clamp_min(1).float() * 6.0
        tab_loss = tab_loss.sum() / tab_denom

        # Onset loss: weighted BCE on raw logits because onsets are sparse.
        # A scalar pos_weight broadcasts over (B, T, 6). This avoids applying a
        # sigmoid inside the model and is numerically more stable than manual BCE.
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

        return self.tab_loss_weight * tab_loss + self.onset_loss_weight * onset_loss


# -----------------------------------------------------------------------------
# BPM-free frame/onset TabEstimator
# -----------------------------------------------------------------------------


class TabEstimator(torch.nn.Module):
    """
    BPM-free frame-level tablature + onset estimator.

    Forward signature:
        frame_tab_pred, frame_onset_logits, olens = model(
            src_pad,
            src_len,
            frame_hand_pos=None,
        )

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
        onset_kernel_size=5,
        onset_tcn_levels=3,
        **unused_kwargs,
    ):
        super().__init__()

        if mode != "tab":
            raise ValueError("This BPM-free network supports mode='tab' only.")

        self.mode = "tab"
        self.use_conv_stack = bool(use_conv_stack)
        self.use_custom_decimation_func = False  # intentionally ignored
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

        if self.hand_position_fusion not in ["hidden", "prior", "hidden+prior", "none"]:
            raise ValueError(
                "hand_position_fusion must be one of: 'hidden', 'prior', 'hidden+prior', 'none'"
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

        self.frame_tab_output_layer = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.encoder_output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 6 * 21),
        )
        self.softmax_by_string = nn.Softmax(dim=3)

        onset_channels = tuple([int(onset_hidden_dim)] * int(onset_tcn_levels))
        self.frame_onset_output_layer = OnsetTCNHead(
            input_size=self.encoder_output_size,
            output_size=6,
            num_channels=onset_channels,
            kernel_size=int(onset_kernel_size),
            dropout=float(onset_dropout),
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

    def forward(self, src_pad, src_len, frame_hand_pos=None):
        batch_size = src_pad.shape[0]

        if self.use_conv_stack:
            encoder_in = self.convstack(torch.unsqueeze(src_pad, dim=1))
        else:
            encoder_in = src_pad

        memory, olens, _ = self.encoder(encoder_in, src_len)

        if self.use_hand_position and self.frame_hand_fusion is not None and frame_hand_pos is not None:
            memory = self.frame_hand_fusion(memory, frame_hand_pos)

        frame_tab_pred = self.frame_tab_output_layer(memory)
        frame_tab_pred = frame_tab_pred.view(batch_size, -1, 6, 21)
        frame_tab_pred = self.softmax_by_string(frame_tab_pred)

        if self.use_hand_position and self.hand_prior is not None and frame_hand_pos is not None:
            frame_tab_pred = apply_hand_position_prior(
                frame_tab_pred,
                frame_hand_pos,
                self.hand_prior,
                strength=self.hand_prior_strength,
            )

        frame_onset_logits = self.frame_onset_output_layer(memory)

        return frame_tab_pred, frame_onset_logits, olens
