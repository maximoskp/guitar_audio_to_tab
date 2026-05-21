#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py

Training script for Tab-estimator-hand-hidonset.

This version keeps the original frame-level + note-level tab objectives and adds
onset supervision as an auxiliary hidden stream. The onset logits are trained for
diagnostics/auxiliary learning, but final prediction remains note_tab_pred.

Expected checkpoint layout:
    model/<run_tag>/<run_name>/config.yaml
    model/<run_tag>/<run_name>/run_metadata.yaml
    model/<run_tag>/<run_name>/testNoXX/epochNNN.model

That layout matches the accompanying hidonset-aware predict.py.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import os
import random
import re
import shutil
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

try:
    import torch_optimizer as torch_optim  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch_optim = None

try:
    from tensorboardX import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore
    except Exception:  # pragma: no cover
        SummaryWriter = None  # type: ignore

from network import TabEstimator

try:
    from network import REST_CLASS as NETWORK_REST_CLASS
except Exception:  # pragma: no cover - older network.py files may not define it
    NETWORK_REST_CLASS = 20

REST_CLASS = int(NETWORK_REST_CLASS)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def dataset_name_from_dir(dataset_dir: str) -> str:
    return os.path.basename(os.path.normpath(dataset_dir)).lower()


def default_npz_dir(dataset_dir: str, note_resolution: int, quantized: bool) -> str:
    dataset_name = dataset_name_from_dir(dataset_dir)
    if quantized:
        return os.path.join(
            "data",
            "npz",
            f"auto_quantized_{note_resolution}_{dataset_name}",
            "split",
        )
    return os.path.join("data", "npz", dataset_name, "split")


def npz_tag_from_path(npz_dir: str) -> str:
    p = os.path.normpath(npz_dir)
    parts = p.split(os.sep)
    if len(parts) >= 2 and parts[-1] == "split":
        return parts[-2]
    return os.path.basename(p).lower()


def safe_run_name(s: Any) -> str:
    s = str(s)
    s = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)
    return s.strip("_") or "run"


def config_bool(config: Dict[str, Any], key: str, default: bool) -> bool:
    return bool(config[key]) if key in config and config[key] is not None else bool(default)


def config_int(config: Dict[str, Any], key: str, default: int) -> int:
    return int(config[key]) if key in config and config[key] is not None else int(default)


def config_float(config: Dict[str, Any], key: str, default: float) -> float:
    return float(config[key]) if key in config and config[key] is not None else float(default)


def config_str(config: Dict[str, Any], key: str, default: str) -> str:
    return str(config[key]) if key in config and config[key] is not None else str(default)


def as_scalar_float(x: Any) -> float:
    arr = np.asarray(x)
    if arr.size == 0:
        raise ValueError("Cannot convert an empty array to float.")
    return float(arr.reshape(-1)[0])


def fit_time_length_2d(x: np.ndarray, target_len: int) -> np.ndarray:
    """Pad or truncate a 2D time-major array to target_len."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array, got {x.shape}")
    if x.shape[0] == target_len:
        return x
    if x.shape[0] > target_len:
        return x[:target_len]
    pad = np.zeros((target_len - x.shape[0], x.shape[1]), dtype=np.float32)
    return np.concatenate([x, pad], axis=0)


# -----------------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------------


def _epoch_number_from_checkpoint(path: str) -> int:
    match = re.search(r"epoch(\d+)\.model$", os.path.basename(path))
    return int(match.group(1)) if match else -1


def safe_torch_load(path: str, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # older PyTorch
        return torch.load(path, map_location=map_location)


def unwrap_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint is not a state_dict-like dict: {type(checkpoint)!r}")
    return checkpoint


def resolve_pretrained_model_path(
    pretrained_model: Optional[str],
    model_root: str,
    test_num: Optional[int] = None,
) -> Optional[str]:
    if pretrained_model is None:
        return None

    raw = str(pretrained_model)
    path = os.path.expanduser(raw)

    if os.path.isfile(path):
        return path

    if os.path.isdir(path):
        candidates = glob.glob(os.path.join(path, "*.model"))
        if not candidates:
            candidates = glob.glob(os.path.join(path, "**", "*.model"), recursive=True)
        if candidates:
            return sorted(
                candidates,
                key=lambda p: (_epoch_number_from_checkpoint(p), os.path.getmtime(p)),
            )[-1]

    candidates: List[str] = []
    if test_num is not None:
        candidates.extend(
            glob.glob(
                os.path.join(model_root, "*", raw, f"testNo{test_num:02d}", "*.model")
            )
        )
        candidates.extend(
            glob.glob(
                os.path.join(model_root, raw, f"testNo{test_num:02d}", "*.model")
            )
        )

    candidates.extend(glob.glob(os.path.join(model_root, "*", raw, "**", "*.model"), recursive=True))
    candidates.extend(glob.glob(os.path.join(model_root, raw, "**", "*.model"), recursive=True))

    if candidates:
        return sorted(
            candidates,
            key=lambda p: (_epoch_number_from_checkpoint(p), os.path.getmtime(p)),
        )[-1]

    raise FileNotFoundError(f"Could not resolve pretrained model: {pretrained_model}")


def load_pretrained_weights(model: nn.Module, checkpoint_path: str, strict: bool = True) -> None:
    checkpoint = unwrap_state_dict(safe_torch_load(checkpoint_path, map_location="cpu"))

    if strict:
        model.load_state_dict(checkpoint, strict=True)
        print(f"Loaded pretrained checkpoint strictly: {checkpoint_path}")
        return

    model_state = model.state_dict()
    compatible = {
        k: v
        for k, v in checkpoint.items()
        if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)
    }
    skipped = [k for k in checkpoint.keys() if k not in compatible]
    model_state.update(compatible)
    model.load_state_dict(model_state, strict=True)

    print(f"Loaded pretrained checkpoint partially: {checkpoint_path}")
    print(f"Loaded tensors: {len(compatible)}")
    print(f"Skipped tensors: {len(skipped)}")
    if skipped:
        print("First skipped tensors:")
        for key in skipped[:20]:
            print("  ", key)


# -----------------------------------------------------------------------------
# Onset target helpers
# -----------------------------------------------------------------------------


def derive_frame_onsets_from_frame_tab(frame_tab: np.ndarray, rest_class: int = REST_CLASS) -> np.ndarray:
    """
    Derive per-string onset targets from frame-level tab labels.

    A string onset occurs when the current class is a played fret and it differs
    from the previous frame's class. This catches rest->fret and fretA->fretB.
    """
    frame_tab = np.asarray(frame_tab)
    if frame_tab.ndim != 3 or frame_tab.shape[1] != 6:
        raise ValueError(f"Expected frame_tab shape (T, 6, C), got {frame_tab.shape}")

    classes = np.argmax(frame_tab, axis=2)
    onset = np.zeros(classes.shape, dtype=np.float32)

    for s in range(classes.shape[1]):
        prev = int(rest_class)
        for t in range(classes.shape[0]):
            cur = int(classes[t, s])
            if cur != rest_class and cur != prev:
                onset[t, s] = 1.0
            prev = cur

    return onset


def load_frame_onset_from_npz(data: np.lib.npyio.NpzFile, frame_tab: np.ndarray) -> np.ndarray:
    """
    Prefer explicit per-frame onset keys. If none exist, derive from frame_tab.
    Accepted explicit key shapes:
      - (T, 6)
      - (T, 6, 21), collapsed over fret classes 0..REST_CLASS-1
    """
    for key in ["frame_onset", "frame_onsets", "onset", "onsets", "frame_tab_onset"]:
        if key not in data.files:
            continue
        x = np.asarray(data[key]).astype(np.float32)
        if x.ndim == 3:
            if x.shape[1] != 6:
                raise ValueError(f"NPZ onset key {key!r} has unsupported shape {x.shape}")
            x = np.max(x[:, :, :REST_CLASS], axis=2)
        if x.ndim != 2 or x.shape[1] != 6:
            raise ValueError(f"NPZ onset key {key!r} has unsupported shape {x.shape}")
        return x

    return derive_frame_onsets_from_frame_tab(frame_tab)


def ms_to_frames(milliseconds: float, hop_length: int, sample_rate: int) -> int:
    frame_ms = 1000.0 * float(hop_length) / float(sample_rate)
    if frame_ms <= 0:
        raise ValueError(f"Invalid frame step: {frame_ms} ms")
    return max(0, int(round(float(milliseconds) / frame_ms)))


def widen_binary_onset_targets(
    onset: np.ndarray,
    radius_frames: int = 0,
    mode: str = "hard",
    sigma_frames: float = 1.0,
) -> np.ndarray:
    """Widen sparse onset targets for BCE supervision."""
    onset = np.asarray(onset, dtype=np.float32)
    radius_frames = int(radius_frames)
    if radius_frames <= 0:
        return onset.copy()

    mode = str(mode).lower()
    sigma_frames = max(1e-6, float(sigma_frames))
    widened = np.zeros_like(onset, dtype=np.float32)

    def value_for_offset(dt: int) -> float:
        if mode == "hard":
            return 1.0
        if mode == "triangular":
            return 1.0 - abs(dt) / float(radius_frames + 1)
        if mode == "gaussian":
            return float(np.exp(-0.5 * (dt / sigma_frames) ** 2))
        raise ValueError(f"Unknown onset widening mode: {mode}")

    if onset.ndim == 1:
        active = np.where(onset > 0)[0]
        for t in active:
            for dt in range(-radius_frames, radius_frames + 1):
                u = int(t + dt)
                if 0 <= u < onset.shape[0]:
                    widened[u] = max(float(widened[u]), value_for_offset(dt))
        return widened

    if onset.ndim == 2:
        T, S = onset.shape
        for s in range(S):
            active = np.where(onset[:, s] > 0)[0]
            for t in active:
                for dt in range(-radius_frames, radius_frames + 1):
                    u = int(t + dt)
                    if 0 <= u < T:
                        widened[u, s] = max(float(widened[u, s]), value_for_offset(dt))
        return widened

    raise ValueError(f"Expected onset target shape (T,) or (T, 6), got {onset.shape}")


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class CustomDataset(Dataset):
    def __init__(
        self,
        data_list: Sequence[str],
        input_feature_type: str,
        use_hand_position: bool = False,
        hand_pos_dim: int = 20,
        allow_missing_hand_pos: bool = False,
        onset_target_radius_frames: int = 0,
        onset_target_mode: str = "hard",
        onset_target_sigma_frames: float = 1.0,
    ) -> None:
        self.data_list = list(data_list)
        self.input_feature_type = str(input_feature_type)
        self.use_hand_position = bool(use_hand_position)
        self.hand_pos_dim = int(hand_pos_dim)
        self.allow_missing_hand_pos = bool(allow_missing_hand_pos)
        self.onset_target_radius_frames = int(onset_target_radius_frames)
        self.onset_target_mode = str(onset_target_mode)
        self.onset_target_sigma_frames = float(onset_target_sigma_frames)

    def __len__(self) -> int:
        return len(self.data_list)

    def _missing_hand_pos(self, length: int) -> np.ndarray:
        return np.ones((int(length), self.hand_pos_dim), dtype=np.float32) / float(self.hand_pos_dim)

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        npz_path = self.data_list[index]
        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception as exc:
            raise RuntimeError(f"Could not load NPZ: {npz_path} ; {repr(exc)}") from exc

        if self.input_feature_type == "cqt":
            input_features = data["cqt"].astype(np.float32)
        elif self.input_feature_type == "melspec":
            input_features = data["mel_spec"].astype(np.float32)
        else:
            raise ValueError(f"Unknown input_feature_type: {self.input_feature_type}")

        if "frame_tab" not in data.files:
            raise KeyError(f"{npz_path} does not contain frame_tab")
        if "tab" not in data.files:
            raise KeyError(f"{npz_path} does not contain tab")

        frame_tab = data["frame_tab"].astype(np.float32)
        note_tab = data["tab"].astype(np.float32)
        frame_onset = load_frame_onset_from_npz(data, frame_tab).astype(np.float32)

        # Keep all frame-aligned arrays the same time length.
        target_frame_len = int(min(input_features.shape[0], frame_tab.shape[0], frame_onset.shape[0]))
        input_features = input_features[:target_frame_len]
        frame_tab = frame_tab[:target_frame_len]
        frame_onset = frame_onset[:target_frame_len]

        frame_onset = widen_binary_onset_targets(
            frame_onset,
            radius_frames=self.onset_target_radius_frames,
            mode=self.onset_target_mode,
            sigma_frames=self.onset_target_sigma_frames,
        )

        frame_len = int(target_frame_len)
        note_len = int(note_tab.shape[0])
        bpm = as_scalar_float(data["tempo"])

        frame_hand_pos = None
        note_hand_pos = None
        if self.use_hand_position:
            has_frame_hp = "frame_hand_pos" in data.files
            has_note_hp = "hand_pos" in data.files or "note_hand_pos" in data.files

            if has_frame_hp and has_note_hp:
                frame_hand_pos = data["frame_hand_pos"].astype(np.float32)
                note_key = "hand_pos" if "hand_pos" in data.files else "note_hand_pos"
                note_hand_pos = data[note_key].astype(np.float32)
            elif self.allow_missing_hand_pos:
                frame_hand_pos = self._missing_hand_pos(frame_len)
                note_hand_pos = self._missing_hand_pos(note_len)
            else:
                raise KeyError(
                    f"{npz_path} does not contain required hand-position keys: "
                    "'frame_hand_pos' and 'hand_pos' or 'note_hand_pos'."
                )

            frame_hand_pos = fit_time_length_2d(frame_hand_pos, frame_len)
            note_hand_pos = fit_time_length_2d(note_hand_pos, note_len)

            if frame_hand_pos.shape[-1] != self.hand_pos_dim:
                raise ValueError(
                    f"{npz_path}: frame_hand_pos dim is {frame_hand_pos.shape[-1]}, "
                    f"expected {self.hand_pos_dim}."
                )
            if note_hand_pos.shape[-1] != self.hand_pos_dim:
                raise ValueError(
                    f"{npz_path}: hand_pos dim is {note_hand_pos.shape[-1]}, "
                    f"expected {self.hand_pos_dim}."
                )

        return (
            input_features,
            frame_tab,
            note_tab,
            frame_onset,
            frame_len,
            note_len,
            bpm,
            frame_hand_pos,
            note_hand_pos,
        )


# -----------------------------------------------------------------------------
# Collate helpers
# -----------------------------------------------------------------------------


def _pad_2d(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    pad_len = int(target_len) - int(x.shape[0])
    if pad_len < 0:
        return x[:target_len]
    return np.pad(x, [(0, pad_len), (0, 0)], mode="constant")


def _pad_3d(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    pad_len = int(target_len) - int(x.shape[0])
    if pad_len < 0:
        return x[:target_len]
    return np.pad(x, [(0, pad_len), (0, 0), (0, 0)], mode="constant")


def _collate_optional_hand_pos(
    frame_hand_pos: Sequence[Optional[np.ndarray]],
    note_hand_pos: Sequence[Optional[np.ndarray]],
    unsorted_frame_len: Sequence[int],
    unsorted_note_len: Sequence[int],
    sort_idx: np.ndarray,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    any_frame_hp = any(x is not None for x in frame_hand_pos)
    any_note_hp = any(x is not None for x in note_hand_pos)

    if not any_frame_hp and not any_note_hp:
        return None, None
    if not all(x is not None for x in frame_hand_pos) or not all(x is not None for x in note_hand_pos):
        raise ValueError("Some batch items have hand-position arrays and others do not.")

    frame_maxlen = int(max(unsorted_frame_len))
    note_maxlen = int(max(unsorted_note_len))

    padded_frame_hand_pos = np.asarray(
        [_pad_2d(x, frame_maxlen) for x in frame_hand_pos], dtype=np.float32
    )
    padded_note_hand_pos = np.asarray(
        [_pad_2d(x, note_maxlen) for x in note_hand_pos], dtype=np.float32
    )

    padded_frame_hand_pos = np.take(padded_frame_hand_pos, sort_idx, axis=0)
    padded_note_hand_pos = np.take(padded_note_hand_pos, sort_idx, axis=0)

    return torch.from_numpy(padded_frame_hand_pos), torch.from_numpy(padded_note_hand_pos)


def tab_pad_collate(batch: Sequence[Tuple[Any, ...]]) -> Tuple[Any, ...]:
    (
        input_features,
        frame_tab,
        note_tab,
        frame_onset,
        frame_len,
        note_len,
        bpm,
        frame_hand_pos,
        note_hand_pos,
    ) = zip(*batch)

    frame_len_np = np.asarray(frame_len, dtype=np.int64)
    note_len_np = np.asarray(note_len, dtype=np.int64)
    bpm_np = np.asarray(bpm, dtype=np.float32)

    frame_maxlen = int(max(frame_len_np))
    note_maxlen = int(max(note_len_np))

    padded_input_features = np.asarray([_pad_2d(x, frame_maxlen) for x in input_features], dtype=np.float32)
    padded_frame_tab = np.asarray([_pad_3d(x, frame_maxlen) for x in frame_tab], dtype=np.float32)
    padded_note_tab = np.asarray([_pad_3d(x, note_maxlen) for x in note_tab], dtype=np.float32)
    padded_frame_onset = np.asarray([_pad_2d(x, frame_maxlen) for x in frame_onset], dtype=np.float32)

    # Sort by frame length for encoders that use packed lengths.
    sort_idx = np.argsort(frame_len_np)[::-1].copy()

    padded_input_features = np.take(padded_input_features, sort_idx, axis=0)
    padded_frame_tab = np.take(padded_frame_tab, sort_idx, axis=0)
    padded_note_tab = np.take(padded_note_tab, sort_idx, axis=0)
    padded_frame_onset = np.take(padded_frame_onset, sort_idx, axis=0)
    sorted_frame_len = np.take(frame_len_np, sort_idx, axis=0)
    sorted_note_len = np.take(note_len_np, sort_idx, axis=0)
    sorted_bpm = np.take(bpm_np, sort_idx, axis=0)

    padded_frame_hand_pos, padded_note_hand_pos = _collate_optional_hand_pos(
        frame_hand_pos,
        note_hand_pos,
        frame_len_np,
        note_len_np,
        sort_idx,
    )

    return (
        torch.from_numpy(padded_input_features),
        torch.from_numpy(padded_frame_tab),
        torch.from_numpy(padded_note_tab),
        torch.from_numpy(padded_frame_onset),
        torch.from_numpy(sorted_frame_len),
        torch.from_numpy(sorted_note_len),
        torch.from_numpy(sorted_bpm),
        padded_frame_hand_pos,
        padded_note_hand_pos,
    )


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------


def sequence_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    lengths = lengths.to(dtype=torch.long)
    lengths = torch.clamp(lengths, min=0, max=max_len)
    arange = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return arange < lengths.unsqueeze(1)


def resize_time_tensor(x: torch.Tensor, target_len: int, mode: str = "nearest") -> torch.Tensor:
    """Resize a (B, T, ...) tensor along T."""
    if x.size(1) == target_len:
        return x
    B = x.size(0)
    tail = tuple(x.shape[2:])
    x_flat = x.reshape(B, x.size(1), -1).transpose(1, 2)
    if mode == "nearest":
        y = F.interpolate(x_flat, size=target_len, mode="nearest")
    else:
        y = F.interpolate(x_flat, size=target_len, mode="linear", align_corners=False)
    return y.transpose(1, 2).reshape(B, target_len, *tail)


class CustomLoss(nn.Module):
    """
    Hidonset training loss.

    Tab losses consume probability outputs shaped (B, T, 6, 21). Onset losses
    consume logits shaped (B, T, 6) and (B, T). All losses are masked by valid
    encoder/note lengths.
    """

    def __init__(
        self,
        onset_loss_weight: float = 0.25,
        onset_positive_weight: float = 10.0,
        global_onset_loss_weight: float = 0.25,
        global_onset_positive_weight: float = 10.0,
        tab_loss_weight: float = 1.0,
        use_galoss: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.onset_loss_weight = float(onset_loss_weight)
        self.onset_positive_weight = float(onset_positive_weight)
        self.global_onset_loss_weight = float(global_onset_loss_weight)
        self.global_onset_positive_weight = float(global_onset_positive_weight)
        self.tab_loss_weight = float(tab_loss_weight)
        self.use_galoss = bool(use_galoss)
        self.eps = float(eps)

    def _tab_loss(self, pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if pred.ndim != 4 or target.ndim != 4:
            raise ValueError(f"Expected tab tensors (B,T,6,C), got pred={pred.shape}, target={target.shape}")
        target = resize_time_tensor(target.float(), pred.size(1), mode="nearest")
        lengths = torch.clamp(lengths.to(pred.device), max=pred.size(1))
        mask = sequence_mask(lengths, pred.size(1)).float().unsqueeze(-1)  # (B,T,1)
        pred = pred.clamp(min=self.eps, max=1.0)
        nll = -(target * torch.log(pred)).sum(dim=-1)  # (B,T,6)
        denom = (mask.sum() * pred.size(2)).clamp_min(1.0)
        return (nll * mask).sum() / denom

    def _onset_loss(
        self,
        logits: Optional[torch.Tensor],
        target: torch.Tensor,
        lengths: torch.Tensor,
        pos_weight: float,
    ) -> torch.Tensor:
        if logits is None:
            return target.new_tensor(0.0)
        if logits.ndim == 3 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        target = target.float().to(logits.device)
        target = resize_time_tensor(target, logits.size(1), mode="nearest")
        lengths = torch.clamp(lengths.to(logits.device), max=logits.size(1))

        if logits.ndim == 3:
            mask = sequence_mask(lengths, logits.size(1)).float().unsqueeze(-1)
            pos = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
            raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pos)
            denom = (mask.sum() * logits.size(2)).clamp_min(1.0)
            return (raw * mask).sum() / denom

        if logits.ndim == 2:
            if target.ndim == 3:
                target = torch.max(target, dim=2).values
            mask = sequence_mask(lengths, logits.size(1)).float()
            pos = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
            raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pos)
            denom = mask.sum().clamp_min(1.0)
            return (raw * mask).sum() / denom

        raise ValueError(f"Unsupported onset logits shape: {logits.shape}")

    def forward(
        self,
        frame_tab_pred: torch.Tensor,
        frame_tab_gt: torch.Tensor,
        note_tab_pred: torch.Tensor,
        note_tab_gt: torch.Tensor,
        frame_onset_pred: Optional[torch.Tensor],
        global_onset_pred: Optional[torch.Tensor],
        frame_onset_gt: torch.Tensor,
        attn: Optional[torch.Tensor],
        olens: torch.Tensor,
        note_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        frame_lengths = olens.to(frame_tab_pred.device) if olens is not None else torch.full(
            (frame_tab_pred.size(0),), frame_tab_pred.size(1), device=frame_tab_pred.device, dtype=torch.long
        )
        note_lengths = note_len.to(note_tab_pred.device)

        frame_loss = self._tab_loss(frame_tab_pred, frame_tab_gt.to(frame_tab_pred.device), frame_lengths)
        note_loss = self._tab_loss(note_tab_pred, note_tab_gt.to(note_tab_pred.device), note_lengths)

        onset_loss = self._onset_loss(
            frame_onset_pred,
            frame_onset_gt.to(frame_tab_pred.device),
            frame_lengths,
            self.onset_positive_weight,
        )

        global_target = torch.max(frame_onset_gt.to(frame_tab_pred.device), dim=2).values
        global_onset_loss = self._onset_loss(
            global_onset_pred,
            global_target,
            frame_lengths,
            self.global_onset_positive_weight,
        )

        # Keep the hook for old configs, but do not force a fragile attention loss.
        guided_attention_loss = frame_loss.new_tensor(0.0)
        _ = attn  # explicit: attention is optional in this script.

        loss = (
            self.tab_loss_weight * (frame_loss + note_loss)
            + self.onset_loss_weight * onset_loss
            + self.global_onset_loss_weight * global_onset_loss
            + guided_attention_loss
        )

        terms = {
            "frame_loss": float(frame_loss.detach().cpu().item()),
            "note_loss": float(note_loss.detach().cpu().item()),
            "onset_loss": float(onset_loss.detach().cpu().item()),
            "global_onset_loss": float(global_onset_loss.detach().cpu().item()),
            "guided_attention_loss": float(guided_attention_loss.detach().cpu().item()),
        }
        return loss, terms


# -----------------------------------------------------------------------------
# Model helpers used by both train.py and predict.py
# -----------------------------------------------------------------------------


def get_encoder_attention(model: nn.Module, encoder_layers: int) -> Optional[torch.Tensor]:
    try:
        attn_map = model.encoder.encoders._modules["0"]._modules["self_attn"].attn
        for n_layer in range(1, int(encoder_layers)):
            attn_map = torch.cat(
                (
                    attn_map,
                    model.encoder.encoders._modules[f"{n_layer}"]._modules["self_attn"].attn,
                ),
                dim=0,
            )
        return attn_map
    except Exception:
        return None


def model_forward(
    model: nn.Module,
    padded_input_features: torch.Tensor,
    frame_len: torch.Tensor,
    note_len: torch.Tensor,
    bpm: torch.Tensor,
    frame_hand_pos: Optional[torch.Tensor] = None,
    note_hand_pos: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, ...]:
    if frame_hand_pos is None and note_hand_pos is None:
        return model(padded_input_features.float(), frame_len, note_len, bpm)
    return model(
        padded_input_features.float(),
        frame_len,
        note_len,
        bpm,
        frame_hand_pos=frame_hand_pos,
        note_hand_pos=note_hand_pos,
    )


def build_tab_estimator(
    mode: str,
    input_feature_type: str,
    encoder_type: str,
    use_custom_decimation_func: bool,
    use_conv_stack: bool,
    n_bins: int,
    hop_length: int,
    sr: int,
    encoder_heads: int,
    encoder_layers: int,
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
    detach_onset_features_for_note: bool = False,
    note_hidden_hand_fusion: bool = False,
    note_rest_preserving_prior: bool = True,
) -> TabEstimator:
    kwargs: Dict[str, Any] = dict(
        encoder_heads=encoder_heads,
        encoder_layers=encoder_layers,
        onset_hidden_dim=onset_hidden_dim,
        onset_dropout=onset_dropout,
        onset_kernel_size=onset_kernel_size,
        onset_tcn_levels=onset_tcn_levels,
        onset_use_raw_features=onset_use_raw_features,
        onset_raw_proj_dim=onset_raw_proj_dim,
        onset_raw_dropout=onset_raw_dropout,
        onset_input_mode=onset_input_mode,
        use_hidden_onset_to_note=use_hidden_onset_to_note,
        detach_onset_features_for_note=detach_onset_features_for_note,
        note_hidden_hand_fusion=note_hidden_hand_fusion,
        note_rest_preserving_prior=note_rest_preserving_prior,
        note_target_length=note_target_length,
    )

    if use_hand_position:
        kwargs.update(
            dict(
                use_hand_position=True,
                hand_pos_dim=hand_pos_dim,
                hand_position_fusion=hand_position_fusion,
                hand_hidden_gate_init=hand_hidden_gate_init,
                hand_prior_strength=hand_prior_strength,
                hand_span=hand_span,
            )
        )

    return TabEstimator(
        mode,
        encoder_type,
        use_custom_decimation_func,
        use_conv_stack,
        n_bins,
        hop_length,
        sr,
        **kwargs,
    )


def freeze_model_encoder(model: nn.Module) -> None:
    for param in model.encoder.parameters():
        param.requires_grad = False
    print("Encoder frozen.")


def freeze_audio_frontend(model: nn.Module) -> None:
    if hasattr(model, "convstack") and model.convstack is not None:
        for param in model.convstack.parameters():
            param.requires_grad = False
        print("ConvStack frozen.")


def make_optimizer(params: Iterable[nn.Parameter], lr: float) -> torch.optim.Optimizer:
    params = list(params)
    if torch_optim is not None and hasattr(torch_optim, "RAdam"):
        return torch_optim.RAdam(params, lr=float(lr))
    if hasattr(torch.optim, "RAdam"):
        return torch.optim.RAdam(params, lr=float(lr))
    return torch.optim.AdamW(params, lr=float(lr))


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: CustomLoss,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    encoder_layers: int,
    freeze_encoder: bool = False,
    freeze_frontend: bool = False,
    train_mode: bool = True,
) -> Dict[str, float]:
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_frame = 0.0
    total_note = 0.0
    total_onset = 0.0
    total_global_onset = 0.0
    total_batches = 0

    for batch in loader:
        (
            padded_input_features,
            frame_gt,
            note_gt,
            frame_onset,
            frame_len,
            note_len,
            bpm,
            frame_hand_pos,
            note_hand_pos,
        ) = batch

        padded_input_features = padded_input_features.to(device, non_blocking=True)
        frame_gt = frame_gt.to(device, non_blocking=True)
        note_gt = note_gt.to(device, non_blocking=True)
        frame_onset = frame_onset.to(device, non_blocking=True)
        frame_len = frame_len.to(device, non_blocking=True)
        note_len = note_len.to(device, non_blocking=True)
        bpm = bpm.to(device, non_blocking=True)
        if frame_hand_pos is not None:
            frame_hand_pos = frame_hand_pos.to(device, non_blocking=True)
        if note_hand_pos is not None:
            note_hand_pos = note_hand_pos.to(device, non_blocking=True)

        if train_mode:
            if optimizer is None:
                raise ValueError("optimizer must not be None when train_mode=True")
            if freeze_encoder:
                model.encoder.eval()
            if freeze_frontend and hasattr(model, "convstack") and model.convstack is not None:
                model.convstack.eval()
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            outputs = model_forward(
                model,
                padded_input_features,
                frame_len,
                note_len,
                bpm,
                frame_hand_pos=frame_hand_pos,
                note_hand_pos=note_hand_pos,
            )

            if len(outputs) == 5:
                frame_pred, note_pred, frame_onset_logits, global_onset_logits, olens = outputs
            elif len(outputs) == 3:
                # Compatibility path for older audio-only experiments. Hidonset training
                # should normally use the 5-output model.
                frame_pred, note_pred, olens = outputs
                frame_onset_logits = None
                global_onset_logits = None
            else:
                raise RuntimeError(f"Unexpected model output arity: {len(outputs)}")

            attn_map = get_encoder_attention(model, encoder_layers)
            loss, loss_terms = criterion(
                frame_pred,
                frame_gt,
                note_pred,
                note_gt,
                frame_onset_logits,
                global_onset_logits,
                frame_onset,
                attn_map,
                olens,
                note_len,
            )

            if train_mode:
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()

        total_loss += float(loss.detach().cpu().item())
        total_frame += float(loss_terms["frame_loss"])
        total_note += float(loss_terms["note_loss"])
        total_onset += float(loss_terms["onset_loss"])
        total_global_onset += float(loss_terms["global_onset_loss"])
        total_batches += 1

    denom = max(1, total_batches)
    return {
        "loss": total_loss / denom,
        "frame_loss": total_frame / denom,
        "note_loss": total_note / denom,
        "onset_loss": total_onset / denom,
        "global_onset_loss": total_global_onset / denom,
    }


def make_fold_lists(
    data_list: Sequence[str],
    test_num: Optional[int],
    n_folds: int,
    train_ratio: float,
    seed_value: int,
) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed_value + (int(test_num) if test_num is not None else 0))

    if n_folds <= 1 or test_num is None:
        data = list(data_list)
        rng.shuffle(data)
        split_idx = int(round(len(data) * float(train_ratio)))
        return data[:split_idx], data[split_idx:]

    fold_prefix = f"{int(test_num):02d}_"
    dev_data_list = [p for p in data_list if not os.path.basename(p).startswith(fold_prefix)]

    # If the files are not fold-prefixed, fall back to random train/valid split.
    if len(dev_data_list) == len(data_list):
        data = list(data_list)
        rng.shuffle(data)
        split_idx = int(round(len(data) * float(train_ratio)))
        return data[:split_idx], data[split_idx:]

    rng.shuffle(dev_data_list)
    split_idx = int(round(len(dev_data_list) * float(train_ratio)))
    return dev_data_list[:split_idx], dev_data_list[split_idx:]


# -----------------------------------------------------------------------------
# Runtime option resolution
# -----------------------------------------------------------------------------


def resolve_runtime_options(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    use_hand_position = bool(args.use_hand_position or config_bool(config, "use_hand_position", False))

    hand_position_fusion = (
        args.hand_position_fusion
        if args.hand_position_fusion is not None
        else config_str(config, "hand_position_fusion", "hidden+prior")
    )

    opts: Dict[str, Any] = {
        "use_hand_position": use_hand_position,
        "hand_pos_dim": int(args.hand_pos_dim if args.hand_pos_dim is not None else config_int(config, "hand_pos_dim", 20)),
        "hand_position_fusion": str(hand_position_fusion),
        "hand_hidden_gate_init": float(
            args.hand_hidden_gate_init
            if args.hand_hidden_gate_init is not None
            else config_float(config, "hand_hidden_gate_init", 0.5)
        ),
        "hand_prior_strength": float(
            args.hand_prior_strength
            if args.hand_prior_strength is not None
            else config_float(config, "hand_prior_strength", 0.35)
        ),
        "hand_span": int(args.hand_span if args.hand_span is not None else config_int(config, "hand_span", 4)),
        "note_target_length": int(
            args.note_target_length
            if args.note_target_length is not None
            else config_int(config, "note_target_length", 64)
        ),
        "onset_hidden_dim": int(args.onset_hidden_dim),
        "onset_dropout": float(args.onset_dropout),
        "onset_kernel_size": int(args.onset_kernel_size),
        "onset_tcn_levels": int(args.onset_tcn_levels),
        "onset_use_raw_features": not bool(args.no_onset_raw_features),
        "onset_raw_proj_dim": int(args.onset_raw_proj_dim),
        "onset_raw_dropout": float(args.onset_raw_dropout),
        "onset_input_mode": str(args.onset_input_mode),
        "use_hidden_onset_to_note": not bool(args.no_hidden_onset_to_note),
        "detach_onset_features_for_note": bool(args.detach_onset_features_for_note),
        "note_hidden_hand_fusion": bool(args.note_hidden_hand_fusion),
        "note_rest_preserving_prior": not bool(args.no_note_rest_preserving_prior),
        "use_custom_decimation_func": config_bool(config, "use_custom_decimation_func", False),
    }

    if opts["onset_input_mode"] in {"full", "raw"} and not opts["onset_use_raw_features"]:
        raise ValueError(
            "--no-onset-raw-features is incompatible with --onset-input-mode full/raw. "
            "Use --onset-input-mode encoder or keep raw features enabled."
        )

    return opts


def build_metadata(
    config: Dict[str, Any],
    args: argparse.Namespace,
    opts: Dict[str, Any],
    npz_dir: str,
    run_tag: str,
    run_name: str,
) -> Dict[str, Any]:
    return {
        "architecture": "tab_estimator_hand_hidonset",
        "npz_dir": str(npz_dir),
        "run_tag": str(run_tag),
        "run_name": str(run_name),
        "uses_note_level_stream": True,
        "uses_frame_tab": True,
        "uses_hidden_onset_to_note": bool(opts["use_hidden_onset_to_note"]),
        "uses_thresholded_onset_for_prediction": False,
        "onset_input_mode": str(opts["onset_input_mode"]),
        "onset_use_raw_features": bool(opts["onset_use_raw_features"]),
        "onset_hidden_dim": int(opts["onset_hidden_dim"]),
        "onset_tcn_levels": int(opts["onset_tcn_levels"]),
        "onset_dropout": float(opts["onset_dropout"]),
        "onset_kernel_size": int(opts["onset_kernel_size"]),
        "onset_raw_proj_dim": int(opts["onset_raw_proj_dim"]),
        "onset_raw_dropout": float(opts["onset_raw_dropout"]),
        "detach_onset_features_for_note": bool(opts["detach_onset_features_for_note"]),
        "note_hidden_hand_fusion": bool(opts["note_hidden_hand_fusion"]),
        "note_rest_preserving_prior": bool(opts["note_rest_preserving_prior"]),
        "use_hand_position": bool(opts["use_hand_position"]),
        "hand_pos_dim": int(opts["hand_pos_dim"]),
        "hand_position_fusion": str(opts["hand_position_fusion"]),
        "hand_hidden_gate_init": float(opts["hand_hidden_gate_init"]),
        "hand_prior_strength": float(opts["hand_prior_strength"]),
        "hand_span": int(opts["hand_span"]),
        "note_target_length": int(opts["note_target_length"]),
        "onset_loss_weight": float(args.onset_loss_weight),
        "onset_positive_weight": float(args.onset_positive_weight),
        "global_onset_loss_weight": float(args.global_onset_loss_weight),
        "global_onset_positive_weight": float(args.global_onset_positive_weight),
        "onset_target_radius_ms": float(args.onset_target_radius_ms),
        "onset_target_mode": str(args.onset_target_mode),
        "onset_target_sigma_ms": None if args.onset_target_sigma_ms is None else float(args.onset_target_sigma_ms),
        "mode": str(config.get("mode", "tab")),
        "input_feature_type": str(config.get("input_feature_type", "cqt")),
        "encoder_type": str(config.get("encoder_type", "transformer")),
        "n_folds": int(args.n_folds),
        "seed": int(args.seed if args.seed is not None else config_int(config, "seed_", 0)),
    }


# -----------------------------------------------------------------------------
# Main train logic
# -----------------------------------------------------------------------------


def train(
    config: Dict[str, Any],
    args: argparse.Namespace,
    opts: Dict[str, Any],
    test_num: Optional[int],
    train_data_list: Sequence[str],
    valid_data_list: Sequence[str],
    tensorboard_dir: str,
    model_dir: str,
) -> None:
    hop_length = int(config["hop_length"])
    sr = int(config["down_sampling_rate"])
    cqt_n_bins = int(config["cqt_n_bins"])
    encoder_heads = int(config["encoder_heads"])
    encoder_layers = int(config["encoder_layers"])
    input_feature_type = str(config["input_feature_type"])
    mode = str(config["mode"])
    encoder_type = str(config["encoder_type"])
    use_conv_stack = bool(config["use_conv_stack"])
    n_bins = cqt_n_bins if input_feature_type == "cqt" else 128

    if mode != "tab":
        raise ValueError("This hidonset train.py currently supports mode='tab' only.")

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        raise EnvironmentError(f"CUDA was requested but is not available: {device_str}")
    device = torch.device(device_str)

    model = build_tab_estimator(
        mode=mode,
        input_feature_type=input_feature_type,
        encoder_type=encoder_type,
        use_custom_decimation_func=bool(opts["use_custom_decimation_func"]),
        use_conv_stack=use_conv_stack,
        n_bins=n_bins,
        hop_length=hop_length,
        sr=sr,
        encoder_heads=encoder_heads,
        encoder_layers=encoder_layers,
        use_hand_position=bool(opts["use_hand_position"]),
        hand_pos_dim=int(opts["hand_pos_dim"]),
        hand_position_fusion=str(opts["hand_position_fusion"]),
        hand_hidden_gate_init=float(opts["hand_hidden_gate_init"]),
        hand_prior_strength=float(opts["hand_prior_strength"]),
        hand_span=int(opts["hand_span"]),
        note_target_length=int(opts["note_target_length"]),
        onset_hidden_dim=int(opts["onset_hidden_dim"]),
        onset_dropout=float(opts["onset_dropout"]),
        onset_kernel_size=int(opts["onset_kernel_size"]),
        onset_tcn_levels=int(opts["onset_tcn_levels"]),
        onset_use_raw_features=bool(opts["onset_use_raw_features"]),
        onset_raw_proj_dim=int(opts["onset_raw_proj_dim"]),
        onset_raw_dropout=float(opts["onset_raw_dropout"]),
        onset_input_mode=str(opts["onset_input_mode"]),
        use_hidden_onset_to_note=bool(opts["use_hidden_onset_to_note"]),
        detach_onset_features_for_note=bool(opts["detach_onset_features_for_note"]),
        note_hidden_hand_fusion=bool(opts["note_hidden_hand_fusion"]),
        note_rest_preserving_prior=bool(opts["note_rest_preserving_prior"]),
    )

    if not bool(args.no_reinit):
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    pretrained_model_path = (
        resolve_pretrained_model_path(args.pretrained_model, args.model_root, test_num=test_num)
        if args.pretrained_model
        else None
    )
    if pretrained_model_path:
        load_pretrained_weights(model, pretrained_model_path, strict=not args.pretrained_allow_partial)

    if args.freeze_frontend:
        freeze_audio_frontend(model)
    if args.freeze_encoder:
        freeze_model_encoder(model)

    model.to(device)

    criterion = CustomLoss(
        onset_loss_weight=float(args.onset_loss_weight),
        onset_positive_weight=float(args.onset_positive_weight),
        global_onset_loss_weight=float(args.global_onset_loss_weight),
        global_onset_positive_weight=float(args.global_onset_positive_weight),
        tab_loss_weight=float(args.tab_loss_weight),
        use_galoss=bool(args.use_galoss),
    ).to(device)

    onset_radius_frames = ms_to_frames(args.onset_target_radius_ms, hop_length=hop_length, sample_rate=sr)
    if args.onset_target_sigma_ms is None:
        onset_sigma_frames = max(1.0, onset_radius_frames / 2.0)
    else:
        onset_sigma_frames = max(1e-6, float(args.onset_target_sigma_ms) * float(sr) / (1000.0 * float(hop_length)))

    train_dataset = CustomDataset(
        train_data_list,
        input_feature_type=input_feature_type,
        use_hand_position=bool(opts["use_hand_position"]),
        hand_pos_dim=int(opts["hand_pos_dim"]),
        allow_missing_hand_pos=bool(args.allow_missing_hand_pos),
        onset_target_radius_frames=onset_radius_frames,
        onset_target_mode=args.onset_target_mode,
        onset_target_sigma_frames=onset_sigma_frames,
    )

    valid_dataset = CustomDataset(
        valid_data_list,
        input_feature_type=input_feature_type,
        use_hand_position=bool(opts["use_hand_position"]),
        hand_pos_dim=int(opts["hand_pos_dim"]),
        allow_missing_hand_pos=bool(args.allow_missing_hand_pos),
        onset_target_radius_frames=onset_radius_frames,
        onset_target_mode=args.onset_target_mode,
        onset_target_sigma_frames=onset_sigma_frames,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=tab_pad_collate,
        num_workers=int(args.n_cores),
        pin_memory=bool(args.pin_memory),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=tab_pad_collate,
        num_workers=int(args.n_cores),
        pin_memory=bool(args.pin_memory),
    )

    writer = SummaryWriter(tensorboard_dir) if SummaryWriter is not None else None
    os.makedirs(model_dir, exist_ok=True)

    lr = float(args.lr if args.lr is not None else config_float(config, "lr", 1e-3))
    optimizer = make_optimizer([p for p in model.parameters() if p.requires_grad], lr=lr)

    max_epochs = int(args.epoch if args.epoch is not None else config_int(config, "epoch", 192))
    save_every = int(args.save_every)

    print("device:", device)
    print("train files:", len(train_dataset), "valid files:", len(valid_dataset))
    print("model_dir:", model_dir)
    print("hidden onset to note:", bool(opts["use_hidden_onset_to_note"]))
    print("onset input mode:", str(opts["onset_input_mode"]))
    print("note hidden hand fusion:", bool(opts["note_hidden_hand_fusion"]))
    print("note rest-preserving prior:", bool(opts["note_rest_preserving_prior"]))

    for epoch in range(1, max_epochs + 1):
        train_stats = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            encoder_layers,
            freeze_encoder=bool(args.freeze_encoder),
            freeze_frontend=bool(args.freeze_frontend),
            train_mode=True,
        )

        valid_stats = train_epoch(
            model,
            valid_loader,
            criterion,
            None,
            device,
            encoder_layers,
            train_mode=False,
        )

        if writer is not None:
            for key, value in train_stats.items():
                writer.add_scalar(f"train/{key}", value, epoch)
            for key, value in valid_stats.items():
                writer.add_scalar(f"valid/{key}", value, epoch)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_stats['loss']:.4f} | valid loss {valid_stats['loss']:.4f} | "
            f"train frame {train_stats['frame_loss']:.4f} note {train_stats['note_loss']:.4f} | "
            f"onset {train_stats['onset_loss']:.4f} global {train_stats['global_onset_loss']:.4f}"
        )

        if epoch % save_every == 0 or epoch == max_epochs:
            checkpoint_path = os.path.join(model_dir, f"epoch{epoch}.model")
            torch.save(model.state_dict(), checkpoint_path)
            print("Saved checkpoint:", checkpoint_path)

    if writer is not None:
        writer.close()


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Tab-estimator-hand-hidonset")

    parser.add_argument("--dataset-dir", default="GuitarSet")
    parser.add_argument("--npz-dir", default=None)
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--config", default="src/config.yaml")
    parser.add_argument("--model-root", default="model")
    parser.add_argument("--tensorboard-root", default="tensorboard")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--run-tag",
        default=None,
        help=(
            "First path component under model/. If omitted and --run-name is set, "
            "the run name is also used as the run tag, matching model/<run>/<run>."
        ),
    )
    parser.add_argument("--n-folds", type=int, default=7)
    parser.add_argument("--test-num", type=int, default=None)
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--n-cores", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--pin-memory", action="store_true")

    parser.add_argument("--pretrained-model", default=None)
    parser.add_argument("--pretrained-allow-partial", action="store_true")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--freeze-frontend", action="store_true")
    parser.add_argument("--no-reinit", action="store_true", help="Do not Xavier-reinitialize before optional checkpoint loading.")

    parser.add_argument("--use-hand-position", action="store_true")
    parser.add_argument("--allow-missing-hand-pos", action="store_true")
    parser.add_argument("--hand-pos-dim", type=int, default=None)
    parser.add_argument(
        "--hand-position-fusion",
        choices=["hidden", "prior", "hidden+prior", "none"],
        default=None,
    )
    parser.add_argument("--hand-hidden-gate-init", type=float, default=None)
    parser.add_argument("--hand-prior-strength", type=float, default=None)
    parser.add_argument("--hand-span", type=int, default=None)
    parser.add_argument("--note-target-length", type=int, default=None)

    parser.add_argument("--use-galoss", action="store_true")
    parser.add_argument("--tab-loss-weight", type=float, default=1.0)
    parser.add_argument("--onset-loss-weight", type=float, default=0.25)
    parser.add_argument("--onset-positive-weight", type=float, default=10.0)
    parser.add_argument("--global-onset-loss-weight", type=float, default=0.25)
    parser.add_argument("--global-onset-positive-weight", type=float, default=10.0)

    parser.add_argument("--onset-hidden-dim", type=int, default=64)
    parser.add_argument("--onset-dropout", type=float, default=0.25)
    parser.add_argument("--onset-kernel-size", type=int, default=3)
    parser.add_argument("--onset-tcn-levels", type=int, default=4)
    parser.add_argument("--onset-target-radius-ms", type=float, default=25.0)
    parser.add_argument("--onset-target-sigma-ms", type=float, default=None)
    parser.add_argument(
        "--onset-target-mode",
        choices=["hard", "triangular", "gaussian"],
        default="hard",
    )
    parser.add_argument("--no-onset-raw-features", action="store_true")
    parser.add_argument("--onset-raw-proj-dim", type=int, default=64)
    parser.add_argument("--onset-raw-dropout", type=float, default=0.10)
    parser.add_argument(
        "--onset-input-mode",
        choices=["full", "encoder", "raw"],
        default="full",
    )
    parser.add_argument("--no-hidden-onset-to-note", action="store_true")
    parser.add_argument("--detach-onset-features-for-note", action="store_true")
    parser.add_argument("--note-hidden-hand-fusion", action="store_true")
    parser.add_argument("--no-note-rest-preserving-prior", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    note_resolution = int(config["note_resolution"])
    train_ratio = float(config["train_ratio"])
    seed_value = int(args.seed if args.seed is not None else config_int(config, "seed_", 0))

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

    opts = resolve_runtime_options(config, args)

    npz_dir = args.npz_dir or default_npz_dir(args.dataset_dir, note_resolution, args.quantized)
    data_list = np.array(sorted(glob.glob(os.path.join(npz_dir, "*.npz"), recursive=True)))
    if len(data_list) == 0:
        raise FileNotFoundError(f"No NPZ files found at: {os.path.join(npz_dir, '*.npz')}")

    inferred_npz_tag = npz_tag_from_path(npz_dir)
    run_name = safe_run_name(args.run_name) if args.run_name else datetime.datetime.now().strftime("%Y%m%d%H%M")
    # This default matches the predict command style requested for hidonset:
    #   python src/predict.py <run_name>/<run_name> ...
    run_tag = safe_run_name(args.run_tag) if args.run_tag else run_name

    base_tensorboard_dir = os.path.join(args.tensorboard_root, run_tag, run_name)
    base_model_dir = os.path.join(args.model_root, run_tag, run_name)
    os.makedirs(base_tensorboard_dir, exist_ok=True)
    os.makedirs(base_model_dir, exist_ok=True)

    config_dst = os.path.join(base_model_dir, "config.yaml")
    if os.path.abspath(args.config) != os.path.abspath(config_dst):
        shutil.copyfile(args.config, config_dst)

    metadata = build_metadata(config, args, opts, npz_dir, run_tag, run_name)
    metadata["source_npz_tag"] = inferred_npz_tag
    with open(os.path.join(base_model_dir, "run_metadata.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

    if args.all_folds:
        test_nums: List[Optional[int]] = list(range(int(args.n_folds)))
    else:
        if args.test_num is not None:
            if args.test_num < 0 or args.test_num >= args.n_folds:
                raise ValueError(f"--test-num must be between 0 and {args.n_folds - 1}")
            test_nums = [int(args.test_num)]
        else:
            test_nums = [None]

    for test_num in test_nums:
        fold_label = f"testNo{test_num:02d}" if test_num is not None else "randomSplit"
        fold_train, fold_valid = make_fold_lists(
            data_list,
            test_num,
            int(args.n_folds),
            train_ratio,
            seed_value,
        )

        fold_model_dir = os.path.join(base_model_dir, fold_label) if test_num is not None else base_model_dir
        fold_tensorboard_dir = os.path.join(base_tensorboard_dir, fold_label) if test_num is not None else base_tensorboard_dir
        os.makedirs(fold_model_dir, exist_ok=True)
        os.makedirs(fold_tensorboard_dir, exist_ok=True)

        fold_metadata = dict(metadata)
        fold_metadata["test_num"] = None if test_num is None else int(test_num)
        fold_metadata["train_files"] = len(fold_train)
        fold_metadata["valid_files"] = len(fold_valid)
        with open(os.path.join(fold_model_dir, "run_metadata.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(fold_metadata, f, sort_keys=False)

        print(f"Training {fold_label}: {len(fold_train)} files, validating on {len(fold_valid)} files")
        train(
            config,
            args,
            opts,
            test_num,
            fold_train,
            fold_valid,
            fold_tensorboard_dir,
            fold_model_dir,
        )


if __name__ == "__main__":
    main()
