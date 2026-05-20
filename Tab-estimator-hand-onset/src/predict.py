#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict.py

BPM-free frame-level tablature + onset evaluation script.

This version evaluates the non-causal TCN-onset BPM-free model and uses
millisecond-based tolerances instead of hardcoded frame tolerances. It also
includes Madmom-style peak-picking helpers for onset activations.

  - per-string onset matching: same string, onset within +/- onset_tolerance_ms
  - global onset matching:     onset within +/- onset_tolerance_ms, string ignored
  - note-event matching:       global onset timing + post-onset fret/string window, then same string/fret within +/- event_tolerance_ms

The conversion from frame indices to time is read from config.yaml:

    frame_seconds = hop_length / down_sampling_rate

This script intentionally does NOT evaluate legacy note-level output:
  - no bpm input
  - no note_len input
  - no note_pred output
  - no beat-grid note-level metrics

It evaluates:
  1. frame-level tablature precision/recall/F1 over dense (frame, string, fret)
  2. onset precision/recall/F1 over note starts on (time, string)
     using a tolerant matching window, default +/-25 ms
  3. onset-decoded note-event precision/recall/F1 over (time, string, fret)
     using global onset peaks for event timing and a short post-onset window
     for fret/string assignment; default note-event tolerance is +/-50 ms
     A configurable label delay can skip unstable attack frames before reading fret/string labels.

The old frame-exact onset scores are also saved as diagnostic columns:
    frame_exact_onset_*

Primary onset columns:
    frame_avg_onset_*
    frame_concat_onset_*

now refer to tolerant onset matching.
"""

import argparse
import glob
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import tqdm
import yaml
from sklearn.metrics import precision_recall_fscore_support

from network import REST_CLASS
from train import (
    build_tab_estimator,
    load_frame_onset_from_npz,
    safe_torch_load,
    unwrap_state_dict,
)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def load_yaml_if_exists(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def metadata_or_config(
    metadata: Dict[str, Any],
    config: Dict[str, Any],
    key: str,
    default: Any,
) -> Any:
    if key in metadata and metadata[key] is not None:
        return metadata[key]
    if key in config and config[key] is not None:
        return config[key]
    return default


def calculate_binary_metrics(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt,
        pred,
        average="binary",
        zero_division=0,
    )
    return float(precision), float(recall), float(f1)


def prf_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def one_hot_argmax_tab(pred: torch.Tensor) -> np.ndarray:
    """
    Convert tab logits/probabilities to one-hot argmax classes.

    Input after squeezing batch:
        (T, 6, 21)

    Output:
        (T, 6, 21), one-hot
    """
    pred_np = pred.detach().cpu().numpy()
    argmax_index = np.argmax(pred_np, axis=2)

    out = np.zeros((pred_np.shape[0], 6, 21), dtype=np.float32)

    for t in range(out.shape[0]):
        for s in range(6):
            out[t, s, int(argmax_index[t, s])] = 1.0

    return out


def tab_classes_from_one_hot(tab: np.ndarray) -> np.ndarray:
    return np.argmax(np.asarray(tab), axis=2).astype(np.int64)


def binary_tab_flat_no_rest(tab_one_hot: np.ndarray) -> np.ndarray:
    """
    Flatten dense tablature one-hot labels excluding the rest class.

    Shape:
        (T, 6, 21) -> (T * 6 * 20,)
    """
    return np.asarray(tab_one_hot[:, :, :REST_CLASS]).flatten()


def frame_seconds_from_config(config: Dict[str, Any]) -> float:
    sr = float(config["down_sampling_rate"])
    hop_length = float(config["hop_length"])

    if sr <= 0:
        raise ValueError(f"Invalid down_sampling_rate: {sr}")

    if hop_length <= 0:
        raise ValueError(f"Invalid hop_length: {hop_length}")

    return hop_length / sr




def ms_to_frames(milliseconds: float, frame_seconds: float) -> int:
    """Convert milliseconds to the nearest number of frames."""
    if milliseconds is None:
        return 0
    if frame_seconds <= 0:
        raise ValueError(f"Invalid frame_seconds: {frame_seconds}")
    return max(0, int(round((float(milliseconds) / 1000.0) / float(frame_seconds))))


def smooth_signal(activations: np.ndarray, smooth: Optional[int] = None) -> np.ndarray:
    """
    Small dependency-free smoothing helper inspired by madmom.

    If smooth is None, 0, or 1, the input is returned unchanged. For an integer
    smooth value > 1, a centered moving average with zero padding is applied.
    Works with 1D activations (T,) or 2D activations (T, C), independently per
    column for 2D input.
    """
    x = np.asarray(activations, dtype=np.float32)

    if smooth is None:
        return x.copy()

    if isinstance(smooth, np.ndarray):
        kernel = np.asarray(smooth, dtype=np.float32).reshape(-1)
        if kernel.size <= 1:
            return x.copy()
        denom = float(np.sum(kernel)) if float(np.sum(kernel)) != 0 else 1.0
        kernel = kernel / denom
        left = kernel.size // 2
        right = kernel.size - left - 1
    else:
        size = int(round(float(smooth)))
        if size <= 1:
            return x.copy()
        kernel = np.ones((size,), dtype=np.float32) / float(size)
        left = size // 2
        right = size - left - 1

    if x.ndim == 1:
        padded = np.pad(x, (left, right), mode="constant")
        return np.convolve(padded, kernel, mode="valid").astype(np.float32)

    if x.ndim == 2:
        out = np.zeros_like(x, dtype=np.float32)
        for c in range(x.shape[1]):
            padded = np.pad(x[:, c], (left, right), mode="constant")
            out[:, c] = np.convolve(padded, kernel, mode="valid").astype(np.float32)
        return out

    raise ValueError("`activations` must be either 1D or 2D")


def _moving_window_reduce_1d(x: np.ndarray, pre: int, post: int, reducer: str) -> np.ndarray:
    """Zero-padded moving mean/max with explicit pre/post frame context."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    pre = max(0, int(pre))
    post = max(0, int(post))
    padded = np.pad(x, (pre, post), mode="constant")
    out = np.zeros_like(x, dtype=np.float32)

    for i in range(x.shape[0]):
        window = padded[i : i + pre + post + 1]
        if reducer == "mean":
            out[i] = float(np.mean(window))
        elif reducer == "max":
            out[i] = float(np.max(window))
        else:
            raise ValueError(f"Unknown reducer: {reducer}")

    return out


def moving_average(activations: np.ndarray, pre_avg: int = 0, post_avg: int = 0) -> np.ndarray:
    """Moving average with zero padding, applied per column for 2D arrays."""
    x = np.asarray(activations, dtype=np.float32)
    pre_avg = max(0, int(pre_avg))
    post_avg = max(0, int(post_avg))

    if pre_avg + post_avg + 1 <= 1:
        return np.zeros_like(x, dtype=np.float32)

    if x.ndim == 1:
        return _moving_window_reduce_1d(x, pre_avg, post_avg, "mean")

    if x.ndim == 2:
        out = np.zeros_like(x, dtype=np.float32)
        for c in range(x.shape[1]):
            out[:, c] = _moving_window_reduce_1d(x[:, c], pre_avg, post_avg, "mean")
        return out

    raise ValueError("`activations` must be either 1D or 2D")


def moving_maximum(activations: np.ndarray, pre_max: int = 1, post_max: int = 1) -> np.ndarray:
    """Moving maximum with zero padding, applied per column for 2D arrays."""
    x = np.asarray(activations, dtype=np.float32)
    pre_max = max(0, int(pre_max))
    post_max = max(0, int(post_max))

    if pre_max + post_max + 1 <= 1:
        return x.copy()

    if x.ndim == 1:
        return _moving_window_reduce_1d(x, pre_max, post_max, "max")

    if x.ndim == 2:
        out = np.zeros_like(x, dtype=np.float32)
        for c in range(x.shape[1]):
            out[:, c] = _moving_window_reduce_1d(x[:, c], pre_max, post_max, "max")
        return out

    raise ValueError("`activations` must be either 1D or 2D")


def peak_picking(
    activations: np.ndarray,
    threshold: float,
    smooth: Optional[int] = None,
    pre_avg: int = 0,
    post_avg: int = 0,
    pre_max: int = 1,
    post_max: int = 1,
):
    """
    Dependency-free Madmom-style peak-picking.

    It keeps activations that are:
      1. above moving average + threshold, and
      2. equal to the local moving maximum.

    Returns np.nonzero indices, matching madmom's behavior:
      - 1D input -> array of peak frame indices
      - 2D input -> tuple(frame_indices, column_indices)
    """
    x = smooth_signal(np.asarray(activations, dtype=np.float32), smooth=smooth)

    if x.ndim not in (1, 2):
        raise ValueError("`activations` must be either 1D or 2D")

    mov_avg = moving_average(x, pre_avg=pre_avg, post_avg=post_avg)
    detections = x * (x >= (mov_avg + float(threshold)))

    if int(pre_max) + int(post_max) + 1 > 1:
        mov_max = moving_maximum(detections, pre_max=pre_max, post_max=post_max)
        detections = detections * (detections == mov_max)

    if x.ndim == 1:
        return np.nonzero(detections)[0]

    return np.nonzero(detections)


def combine_event_frames(frames: Sequence[int], combine_frames: int = 0) -> List[int]:
    """
    Keep the left-most event if multiple events occur within combine_frames.
    This mirrors madmom's combine_events(..., mode='left') behavior.
    """
    frames = sorted(int(f) for f in frames)
    combine_frames = max(0, int(combine_frames))

    if combine_frames <= 0 or len(frames) <= 1:
        return frames

    combined: List[int] = []

    for f in frames:
        if not combined or (f - combined[-1]) > combine_frames:
            combined.append(f)

    return combined


def peak_pick_binary_from_scores(
    scores: np.ndarray,
    threshold: float,
    smooth: Optional[int] = None,
    pre_avg: int = 0,
    post_avg: int = 0,
    pre_max: int = 1,
    post_max: int = 1,
    combine_frames: int = 0,
) -> np.ndarray:
    """
    Convert onset activation scores to a binary onset matrix using peak-picking.

    scores can be:
      - shape (T,) for global onsets
      - shape (T, C) for per-string/per-class onsets
    """
    x = np.asarray(scores, dtype=np.float32)
    binary = np.zeros_like(x, dtype=np.float32)

    peaks = peak_picking(
        x,
        threshold=float(threshold),
        smooth=smooth,
        pre_avg=int(pre_avg),
        post_avg=int(post_avg),
        pre_max=int(pre_max),
        post_max=int(post_max),
    )

    if x.ndim == 1:
        peak_frames = combine_event_frames(peaks.tolist(), combine_frames=combine_frames)
        binary[peak_frames] = 1.0
        return binary

    if x.ndim == 2:
        peak_t, peak_c = peaks
        for c in range(x.shape[1]):
            frames_c = peak_t[peak_c == c].tolist()
            frames_c = combine_event_frames(frames_c, combine_frames=combine_frames)
            if frames_c:
                binary[frames_c, c] = 1.0
        return binary

    raise ValueError("`scores` must be either 1D or 2D")


def threshold_binary_from_scores(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(scores, dtype=np.float32) >= float(threshold)).astype(np.float32)


def onset_scores_from_model_output(logits: torch.Tensor, onset_positive_class: int = 1) -> np.ndarray:
    """
    Convert onset model output to onset probabilities/scores.

    Supported squeezed shapes:
      - (T, 6): sigmoid binary logits per string
      - (T, 6, 2): 2-class per-string logits; use softmax[..., onset_positive_class]
      - (T, 2): global 2-class logits; use softmax[..., onset_positive_class]

    The current BPM-free model uses (T, 6), but this helper makes predict.py
    compatible with a future 2-class Softmax-style onset head.
    """
    z = torch.squeeze(logits, 0).detach().cpu()

    if z.ndim == 2:
        # Usually (T, 6) sigmoid logits. If (T, 2), treat as global 2-class.
        if z.shape[-1] == 2:
            return torch.softmax(z, dim=-1)[..., int(onset_positive_class)].numpy().astype(np.float32)
        return torch.sigmoid(z).numpy().astype(np.float32)

    if z.ndim == 3 and z.shape[-1] == 2:
        return torch.softmax(z, dim=-1)[..., int(onset_positive_class)].numpy().astype(np.float32)

    raise RuntimeError(f"Unexpected onset logits shape after squeeze: {tuple(z.shape)}")



def global_onset_scores_from_model_output(logits: torch.Tensor, onset_positive_class: int = 1) -> np.ndarray:
    """
    Convert global onset model output to a 1D probability vector.

    Supported squeezed shapes:
      - (T,): sigmoid binary logits
      - (T, 1): sigmoid binary logits
      - (T, 2): 2-class logits; use softmax[..., onset_positive_class]
    """
    z = torch.squeeze(logits, 0).detach().cpu()

    if z.ndim == 1:
        return torch.sigmoid(z).numpy().astype(np.float32)

    if z.ndim == 2:
        if z.shape[-1] == 1:
            return torch.sigmoid(z[:, 0]).numpy().astype(np.float32)
        if z.shape[-1] == 2:
            return torch.softmax(z, dim=-1)[..., int(onset_positive_class)].numpy().astype(np.float32)

    raise RuntimeError(f"Unexpected global onset logits shape after squeeze: {tuple(z.shape)}")

def resolve_npz_dir(explicit_npz_dir: Optional[str], trained_model: str) -> str:
    if explicit_npz_dir is not None:
        return explicit_npz_dir

    metadata_path = os.path.join("model", trained_model, "run_metadata.yaml")
    metadata = load_yaml_if_exists(metadata_path)

    if metadata.get("npz_dir"):
        return str(metadata["npz_dir"])

    return os.path.join("data", "npz", "original", "split")


def checkpoint_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    return unwrap_state_dict(safe_torch_load(checkpoint_path, map_location="cpu"))


def checkpoint_has_hand_layers(checkpoint_path: str) -> bool:
    state = checkpoint_state_dict(checkpoint_path)
    return any(
        str(k).startswith("frame_hand_fusion.")
        or str(k).startswith("hand_prior.")
        for k in state.keys()
    )


def checkpoint_is_bpm_free_onset(checkpoint_path: str) -> bool:
    state = checkpoint_state_dict(checkpoint_path)

    has_onset = any(
        str(k).startswith("frame_onset_output_layer.")
        or str(k).startswith("global_onset_output_layer.")
        for k in state.keys()
    )

    # Legacy TabEstimator checkpoints contain note-path parameters. This script is only
    # for the clean BPM-free frame/onset network.
    has_note = any(
        str(k).startswith("note_")
        or "note_tab_output_layer" in str(k)
        or "note_encoder" in str(k)
        for k in state.keys()
    )

    return bool(has_onset and not has_note)


# -----------------------------------------------------------------------------
# Hand-position helper
# -----------------------------------------------------------------------------


def load_frame_hand_tensor_from_npz(
    npz_file: np.lib.npyio.NpzFile,
    npz_filename: str,
    device: torch.device,
    hand_pos_dim: int,
    target_len: int,
    allow_missing_hand_pos: bool = False,
) -> Optional[torch.Tensor]:
    if "frame_hand_pos" in npz_file.files:
        frame_hand = npz_file["frame_hand_pos"].astype(np.float32)[:target_len]
    elif allow_missing_hand_pos:
        frame_hand = np.ones((target_len, hand_pos_dim), dtype=np.float32) / float(hand_pos_dim)
    else:
        raise KeyError(
            f"{npz_filename} is missing frame_hand_pos. "
            "Use the hand-position NPZ split or pass --allow-missing-hand-pos for debugging."
        )

    if frame_hand.ndim != 2:
        raise ValueError(f"{npz_filename}: frame_hand_pos should be 2D, got {frame_hand.shape}")

    if frame_hand.shape[-1] != hand_pos_dim:
        raise ValueError(
            f"{npz_filename}: frame_hand_pos dim is {frame_hand.shape[-1]}, expected {hand_pos_dim}."
        )

    return torch.from_numpy(frame_hand).float().unsqueeze(0).to(device)


# -----------------------------------------------------------------------------
# Model construction
# -----------------------------------------------------------------------------


def build_model_for_prediction(
    trained_model: str,
    epoch: int,
    test_num: int,
    config: Dict[str, Any],
    metadata: Dict[str, Any],
    model_path: str,
    device: torch.device,
    verbose: bool = False,
):
    if not checkpoint_is_bpm_free_onset(model_path):
        raise RuntimeError(
            "This predict.py only supports checkpoints from the BPM-free frame/onset network. "
            "The checkpoint appears to be legacy or incompatible."
        )

    input_feature_type = str(config["input_feature_type"])
    mode = str(config["mode"])
    encoder_type = str(config["encoder_type"])
    use_conv_stack = bool(config["use_conv_stack"])
    hop_length = int(config["hop_length"])
    sr = int(config["down_sampling_rate"])
    cqt_n_bins = int(config["cqt_n_bins"])
    encoder_heads = int(config["encoder_heads"])
    encoder_layers = int(config["encoder_layers"])

    if input_feature_type == "cqt":
        n_bins = cqt_n_bins
    elif input_feature_type == "melspec":
        n_bins = 128
    else:
        raise ValueError(f"Unknown input_feature_type: {input_feature_type}")

    use_hand_position = bool(metadata_or_config(metadata, config, "use_hand_position", False))

    if not use_hand_position and checkpoint_has_hand_layers(model_path):
        use_hand_position = True
        if verbose:
            print("[info] Checkpoint contains frame hand-position layers; forcing use_hand_position=True.")

    hand_pos_dim = int(metadata_or_config(metadata, config, "hand_pos_dim", 20))
    hand_position_fusion = str(metadata_or_config(metadata, config, "hand_position_fusion", "hidden+prior"))
    hand_hidden_gate_init = float(metadata_or_config(metadata, config, "hand_hidden_gate_init", 0.5))
    hand_prior_strength = float(metadata_or_config(metadata, config, "hand_prior_strength", 0.35))
    hand_span = int(metadata_or_config(metadata, config, "hand_span", 4))

    # Prefer run metadata. If metadata is missing, infer whether the checkpoint
    # contains the raw-feature onset projection. This keeps older TCN-onset
    # checkpoints usable while making the new encoder+raw onset branch explicit.
    checkpoint_keys = checkpoint_state_dict(model_path).keys()
    checkpoint_has_raw_onset = any(str(k).startswith("onset_raw_feature_proj.") for k in checkpoint_keys)

    onset_hidden_dim = int(metadata_or_config(metadata, config, "onset_hidden_dim", 64))
    onset_dropout = float(metadata_or_config(metadata, config, "onset_dropout", 0.25))
    onset_kernel_size = int(metadata_or_config(metadata, config, "onset_kernel_size", 3))
    onset_tcn_levels = int(metadata_or_config(metadata, config, "onset_tcn_levels", 4))
    onset_use_raw_features = bool(metadata_or_config(metadata, config, "onset_use_raw_features", checkpoint_has_raw_onset))
    onset_raw_proj_dim = int(metadata_or_config(metadata, config, "onset_raw_proj_dim", 64))
    onset_raw_dropout = float(metadata_or_config(metadata, config, "onset_raw_dropout", 0.10))

    model = build_tab_estimator(
        mode=mode,
        input_feature_type=input_feature_type,
        encoder_type=encoder_type,
        use_custom_decimation_func=False,
        use_conv_stack=use_conv_stack,
        n_bins=n_bins,
        hop_length=hop_length,
        sr=sr,
        encoder_heads=encoder_heads,
        encoder_layers=encoder_layers,
        use_hand_position=use_hand_position,
        hand_pos_dim=hand_pos_dim,
        hand_position_fusion=hand_position_fusion,
        hand_hidden_gate_init=hand_hidden_gate_init,
        hand_prior_strength=hand_prior_strength,
        hand_span=hand_span,
        onset_hidden_dim=onset_hidden_dim,
        onset_dropout=onset_dropout,
        onset_kernel_size=onset_kernel_size,
        onset_tcn_levels=onset_tcn_levels,
        onset_use_raw_features=onset_use_raw_features,
        onset_raw_proj_dim=onset_raw_proj_dim,
        onset_raw_dropout=onset_raw_dropout,
    )

    checkpoint = checkpoint_state_dict(model_path)
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()

    info = {
        "use_hand_position": bool(use_hand_position),
        "hand_pos_dim": int(hand_pos_dim),
        "encoder_layers": int(encoder_layers),
        "input_feature_type": input_feature_type,
        "onset_use_raw_features": bool(onset_use_raw_features),
        "onset_raw_proj_dim": int(onset_raw_proj_dim),
        "onset_raw_dropout": float(onset_raw_dropout),
    }

    if verbose:
        print("model_path:", model_path)
        print("architecture: bpm_free_frame_tab_onset_tcn_encoder_plus_raw" if onset_use_raw_features else "architecture: bpm_free_frame_tab_onset_tcn")
        print("use_hand_position:", use_hand_position)
        print("onset_use_raw_features:", onset_use_raw_features)
        print("onset_raw_proj_dim:", onset_raw_proj_dim)
        print("onset_hidden_dim:", onset_hidden_dim)
        print("onset_kernel_size:", onset_kernel_size)
        print("onset_tcn_levels:", onset_tcn_levels)

    return model, info


# -----------------------------------------------------------------------------
# Event decoding and tolerant evaluation
# -----------------------------------------------------------------------------


def decode_events_from_tab_and_onset(
    tab_one_hot: np.ndarray,
    onset_binary: np.ndarray,
) -> List[Tuple[int, int, int]]:
    """
    Convert dense frame tab + per-string binary onsets to note events.

    Returns events as tuples:
        (frame_index, string_index_low_e_first, fret)
    """
    classes = tab_classes_from_one_hot(tab_one_hot)
    onset_binary = np.asarray(onset_binary)

    events: List[Tuple[int, int, int]] = []
    T = min(classes.shape[0], onset_binary.shape[0])

    for t in range(T):
        for s in range(6):
            if onset_binary[t, s] <= 0:
                continue

            fret = int(classes[t, s])

            if fret >= REST_CLASS:
                continue

            events.append((int(t), int(s), fret))

    return events


def decode_events_from_global_onsets_and_tab_window(
    tab_scores: np.ndarray,
    global_onset_binary: np.ndarray,
    string_onset_scores: Optional[np.ndarray] = None,
    label_window_frames: int = 2,
    label_delay_frames: int = 0,
    string_window_frames: Optional[int] = None,
    tab_threshold: float = 0.50,
    string_threshold: float = 0.30,
    use_string_onset_filter: bool = True,
) -> List[Tuple[int, int, int]]:
    """
    Decode note events from global onset timing and a short post-onset tab window.

    The global onset detector supplies event times. For each global onset frame t,
    the decoder chooses fret/string labels from a short post-onset frame-tab window:

        [t + label_delay_frames, t + label_delay_frames + label_window_frames]

    The delay is useful because frame-tab predictions can be unstable exactly
    at the noisy attack frame. Optionally, the per-string onset head filters
    which strings are allowed to emit notes, using a separate string window.

    Returns events as tuples:
        (global_onset_frame, string_index_low_e_first, fret)
    """
    tab_scores = np.asarray(tab_scores, dtype=np.float32)
    global_onset_binary = np.asarray(global_onset_binary).reshape(-1) > 0

    if tab_scores.ndim != 3 or tab_scores.shape[1] != 6 or tab_scores.shape[2] < REST_CLASS + 1:
        raise ValueError(f"Expected tab_scores shape (T, 6, 21), got {tab_scores.shape}")

    if string_onset_scores is not None:
        string_onset_scores = np.asarray(string_onset_scores, dtype=np.float32)
        if string_onset_scores.ndim != 2 or string_onset_scores.shape[1] != 6:
            raise ValueError(f"Expected string_onset_scores shape (T, 6), got {string_onset_scores.shape}")

    T = min(tab_scores.shape[0], global_onset_binary.shape[0])
    label_window_frames = max(0, int(label_window_frames))
    label_delay_frames = max(0, int(label_delay_frames))

    if string_window_frames is None:
        string_window_frames = label_window_frames
    string_window_frames = max(0, int(string_window_frames))

    events: List[Tuple[int, int, int]] = []

    for t in np.where(global_onset_binary[:T])[0].tolist():
        label_start = min(T, int(t) + label_delay_frames)
        label_end = min(T, label_start + label_window_frames + 1)
        if label_end <= label_start:
            continue

        tab_window = tab_scores[label_start:label_end, :, :REST_CLASS]

        if string_onset_scores is not None:
            string_start = int(t)
            string_end = min(T, string_start + string_window_frames + 1)
            onset_window = string_onset_scores[string_start:string_end, :]
        else:
            onset_window = None

        for s in range(6):
            # Optional per-string onset support. This keeps global timing but
            # avoids emitting every currently sustained string at each global peak.
            if use_string_onset_filter and onset_window is not None:
                if float(np.max(onset_window[:, s])) < float(string_threshold):
                    continue

            string_window = tab_window[:, s, :]
            flat_idx = int(np.argmax(string_window))
            local_frame_idx, fret = np.unravel_index(flat_idx, string_window.shape)
            score = float(string_window[local_frame_idx, fret])

            if score < float(tab_threshold):
                continue

            events.append((int(t), int(s), int(fret)))

    return events


def tolerant_onset_precision_recall_f1(
    pred_onset_binary: np.ndarray,
    gt_onset_binary: np.ndarray,
    frame_seconds: float,
    tolerance_seconds: float,
) -> Tuple[float, float, float, int, int, int]:
    """
    Tolerant onset-only matching.

    A predicted onset is correct if:
        - same string
        - absolute onset-time difference <= tolerance_seconds

    Fret is ignored here. This evaluates onset timing only.
    """
    pred = np.asarray(pred_onset_binary) > 0
    gt = np.asarray(gt_onset_binary) > 0

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, gt={gt.shape}")

    tp = 0
    fp = 0
    fn = 0

    for string_idx in range(pred.shape[1]):
        pred_frames = np.where(pred[:, string_idx])[0].tolist()
        gt_frames = np.where(gt[:, string_idx])[0].tolist()

        used_gt = set()

        for pf in pred_frames:
            best_gt_idx = None
            best_dt = None

            for gt_idx, gf in enumerate(gt_frames):
                if gt_idx in used_gt:
                    continue

                dt = abs(float(pf - gf) * float(frame_seconds))

                if dt <= tolerance_seconds:
                    if best_dt is None or dt < best_dt:
                        best_dt = dt
                        best_gt_idx = gt_idx

            if best_gt_idx is not None:
                used_gt.add(best_gt_idx)
                tp += 1
            else:
                fp += 1

        fn += len(gt_frames) - len(used_gt)

    precision, recall, f1 = prf_from_counts(tp, fp, fn)

    return precision, recall, f1, int(tp), int(fp), int(fn)




def tolerant_global_onset_precision_recall_f1(
    pred_global_onset_binary: np.ndarray,
    gt_global_onset_binary: np.ndarray,
    frame_seconds: float,
    tolerance_seconds: float,
) -> Tuple[float, float, float, int, int, int]:
    """
    Global/no-string onset matching.

    A predicted onset is correct if there is any reference onset within
    +/- tolerance_seconds. String and fret are ignored.
    """
    pred_frames = np.where(np.asarray(pred_global_onset_binary).reshape(-1) > 0)[0].tolist()
    gt_frames = np.where(np.asarray(gt_global_onset_binary).reshape(-1) > 0)[0].tolist()

    if len(pred_frames) == 0 and len(gt_frames) == 0:
        return 1.0, 1.0, 1.0, 0, 0, 0

    if len(pred_frames) == 0:
        return 0.0, 0.0, 0.0, 0, 0, int(len(gt_frames))

    if len(gt_frames) == 0:
        return 0.0, 0.0, 0.0, 0, int(len(pred_frames)), 0

    used_gt = set()
    tp = 0

    for pf in sorted(pred_frames):
        best_gt_idx = None
        best_dt = None

        for gt_idx, gf in enumerate(gt_frames):
            if gt_idx in used_gt:
                continue

            dt = abs(float(pf - gf) * float(frame_seconds))

            if dt <= tolerance_seconds:
                if best_dt is None or dt < best_dt:
                    best_gt_idx = gt_idx
                    best_dt = dt

        if best_gt_idx is not None:
            used_gt.add(best_gt_idx)
            tp += 1

    fp = len(pred_frames) - tp
    fn = len(gt_frames) - tp

    precision, recall, f1 = prf_from_counts(tp, fp, fn)

    return precision, recall, f1, int(tp), int(fp), int(fn)

def event_precision_recall_f1_tolerant(
    pred_events: Sequence[Tuple[int, int, int]],
    gt_events: Sequence[Tuple[int, int, int]],
    frame_seconds: float,
    tolerance_seconds: float,
) -> Tuple[float, float, float, int, int, int]:
    """
    Note-event matching.

    A predicted note event is correct if:
        - same string
        - same fret
        - absolute onset-time difference <= tolerance_seconds

    Events are tuples:
        (frame_index, string_index_low_e_first, fret)
    """
    if len(pred_events) == 0 and len(gt_events) == 0:
        return 1.0, 1.0, 1.0, 0, 0, 0

    if len(pred_events) == 0:
        return 0.0, 0.0, 0.0, 0, 0, int(len(gt_events))

    if len(gt_events) == 0:
        return 0.0, 0.0, 0.0, 0, int(len(pred_events)), 0

    used_gt = set()
    tp = 0

    for pt, ps, pf in sorted(pred_events):
        best_idx = None
        best_dt = None

        for gt_idx, (gt, gs, gf) in enumerate(gt_events):
            if gt_idx in used_gt:
                continue

            if int(ps) != int(gs) or int(pf) != int(gf):
                continue

            dt = abs(float(pt - gt) * float(frame_seconds))

            if dt <= tolerance_seconds:
                if best_dt is None or dt < best_dt:
                    best_idx = gt_idx
                    best_dt = dt

        if best_idx is not None:
            used_gt.add(best_idx)
            tp += 1

    fp = len(pred_events) - tp
    fn = len(gt_events) - tp

    precision, recall, f1 = prf_from_counts(tp, fp, fn)

    return precision, recall, f1, int(tp), int(fp), int(fn)


# -----------------------------------------------------------------------------
# Main scoring
# -----------------------------------------------------------------------------


def calc_score(
    test_num: int,
    trained_model: str,
    use_model_epoch: int,
    config_path: str,
    npz_dir: str,
    device: str = "cpu",
    onset_threshold: float = 0.5,
    onset_tolerance_ms: float = 25.0,
    event_tolerance_ms: float = 50.0,
    use_peak_picking: bool = True,
    peak_smooth_ms: float = 0.0,
    peak_pre_avg_ms: float = 0.0,
    peak_post_avg_ms: float = 0.0,
    peak_pre_max_ms: float = 50.0,
    peak_post_max_ms: float = 50.0,
    peak_combine_ms: float = 30.0,
    event_label_delay_ms: float = 0.0,
    event_label_window_ms: float = 50.0,
    event_string_window_ms: Optional[float] = None,
    event_tab_threshold: float = 0.50,
    event_string_threshold: float = 0.30,
    no_event_string_filter: bool = False,
    onset_positive_class: int = 1,
    allow_missing_hand_pos: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    metadata_path = os.path.join("model", trained_model, "run_metadata.yaml")
    metadata = load_yaml_if_exists(metadata_path)

    mode = str(config["mode"])
    input_feature_type = str(config["input_feature_type"])
    fold_id = f"{test_num:02d}"

    if mode != "tab":
        raise ValueError("This BPM-free predict.py supports config mode: tab only.")

    frame_seconds = frame_seconds_from_config(config)
    frame_ms = frame_seconds * 1000.0
    onset_tolerance_seconds = float(onset_tolerance_ms) / 1000.0
    event_tolerance_seconds = float(event_tolerance_ms) / 1000.0

    peak_smooth_frames = ms_to_frames(peak_smooth_ms, frame_seconds)
    peak_pre_avg_frames = ms_to_frames(peak_pre_avg_ms, frame_seconds)
    peak_post_avg_frames = ms_to_frames(peak_post_avg_ms, frame_seconds)
    peak_pre_max_frames = ms_to_frames(peak_pre_max_ms, frame_seconds)
    peak_post_max_frames = ms_to_frames(peak_post_max_ms, frame_seconds)
    peak_combine_frames = ms_to_frames(peak_combine_ms, frame_seconds)
    event_label_delay_frames = ms_to_frames(event_label_delay_ms, frame_seconds)
    event_label_window_frames = ms_to_frames(event_label_window_ms, frame_seconds)
    if event_string_window_ms is None:
        event_string_window_ms = event_label_window_ms
    event_string_window_frames = ms_to_frames(event_string_window_ms, frame_seconds)
    peak_smooth_arg = peak_smooth_frames if peak_smooth_frames > 1 else None

    model_path = os.path.join("model", trained_model, f"testNo{fold_id}", f"epoch{use_model_epoch}.model")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    device_obj = torch.device(device)

    model, model_info = build_model_for_prediction(
        trained_model=trained_model,
        epoch=use_model_epoch,
        test_num=test_num,
        config=config,
        metadata=metadata,
        model_path=model_path,
        device=device_obj,
        verbose=verbose,
    )

    test_data_path = os.path.join(npz_dir, f"{fold_id}_*.npz")
    test_data_list = np.array(sorted(glob.glob(test_data_path, recursive=True)))

    if len(test_data_list) == 0:
        raise FileNotFoundError(f"No test NPZ files found at: {test_data_path}")

    if verbose:
        print("npz_dir:", npz_dir)
        print(f"frame step: {frame_ms:.3f} ms")
        print(f"onset tolerance: +/- {float(onset_tolerance_ms):.1f} ms")
        print(f"event tolerance: +/- {float(event_tolerance_ms):.1f} ms")
        print(
            f"event label delay/window: +{float(event_label_delay_ms):.1f} ms / "
            f"+{float(event_label_window_ms):.1f} ms "
            f"({event_label_delay_frames} delay frames, {event_label_window_frames} window frames)"
        )
        print(f"event string window: +{float(event_string_window_ms):.1f} ms ({event_string_window_frames} frames)")
        print(f"event tab threshold: {float(event_tab_threshold):.3f}")
        print(f"event string threshold: {float(event_string_threshold):.3f}")
        print(f"event string filter: {not bool(no_event_string_filter)}")
        print(f"onset threshold: {float(onset_threshold):.3f}")
        print(f"peak picking: {bool(use_peak_picking)}")
        if use_peak_picking:
            print(
                "peak picking frames "
                f"smooth={peak_smooth_frames}, "
                f"pre_avg={peak_pre_avg_frames}, post_avg={peak_post_avg_frames}, "
                f"pre_max={peak_pre_max_frames}, post_max={peak_post_max_frames}, "
                f"combine={peak_combine_frames}"
            )

    # Dense frame-tab metrics.
    frame_sum_p = frame_sum_r = frame_sum_f = 0.0
    frame_concat_pred = np.array([], dtype=np.float32)
    frame_concat_gt = np.array([], dtype=np.float32)

    # Frame-exact onset diagnostics.
    exact_onset_sum_p = exact_onset_sum_r = exact_onset_sum_f = 0.0
    exact_onset_concat_pred = np.array([], dtype=np.float32)
    exact_onset_concat_gt = np.array([], dtype=np.float32)

    # Tolerant per-string onset metrics.
    onset_sum_p = onset_sum_r = onset_sum_f = 0.0
    onset_sum_tp = onset_sum_fp = onset_sum_fn = 0

    # Tolerant global/no-string onset metrics.
    global_onset_sum_p = global_onset_sum_r = global_onset_sum_f = 0.0
    global_onset_sum_tp = global_onset_sum_fp = global_onset_sum_fn = 0

    # Tolerant note-event metrics.
    event_sum_p = event_sum_r = event_sum_f = 0.0
    event_sum_tp = event_sum_fp = event_sum_fn = 0

    for npz_filename in tqdm.tqdm(test_data_list):
        npz_file = np.load(npz_filename, allow_pickle=True)

        if input_feature_type == "cqt":
            input_features_np = npz_file["cqt"].astype(np.float32)
        elif input_feature_type == "melspec":
            input_features_np = npz_file["mel_spec"].astype(np.float32)
        else:
            raise ValueError(f"Unknown input_feature_type: {input_feature_type}")

        frame_tab_gt = npz_file["frame_tab"].astype(np.float32)
        frame_onset_gt = load_frame_onset_from_npz(npz_file, frame_tab_gt).astype(np.float32)

        target_len = min(input_features_np.shape[0], frame_tab_gt.shape[0], frame_onset_gt.shape[0])

        input_features_np = input_features_np[:target_len]
        frame_tab_gt = frame_tab_gt[:target_len]
        frame_onset_gt = frame_onset_gt[:target_len]

        input_features = torch.from_numpy(input_features_np).float().unsqueeze(0).to(device_obj)
        frame_len = torch.tensor([target_len], dtype=torch.long, device=device_obj)

        frame_hand_pos = None

        if model_info["use_hand_position"]:
            frame_hand_pos = load_frame_hand_tensor_from_npz(
                npz_file=npz_file,
                npz_filename=npz_filename,
                device=device_obj,
                hand_pos_dim=int(model_info["hand_pos_dim"]),
                target_len=target_len,
                allow_missing_hand_pos=allow_missing_hand_pos,
            )

        with torch.no_grad():
            model_out = model(
                input_features,
                frame_len,
                frame_hand_pos=frame_hand_pos,
            )

        if len(model_out) == 4:
            frame_tab_score, frame_onset_logits, global_onset_logits, olens = model_out
        elif len(model_out) == 3:
            # Backward compatibility with earlier BPM-free checkpoints.
            frame_tab_score, frame_onset_logits, olens = model_out
            global_onset_logits = None
        else:
            raise RuntimeError(f"Unexpected model output length: {len(model_out)}")

        pred_len = int(olens[0].item())

        frame_tab_score_np = torch.squeeze(frame_tab_score, 0).detach().cpu().numpy().astype(np.float32)[:pred_len]
        frame_tab_pred = one_hot_argmax_tab(torch.squeeze(frame_tab_score, 0))[:pred_len]
        frame_onset_score_np = onset_scores_from_model_output(
            frame_onset_logits,
            onset_positive_class=int(onset_positive_class),
        )[:pred_len]

        if frame_onset_score_np.ndim == 1:
            # Global-only onset head fallback. Repeat as a diagnostic per-string
            # score only if needed, but the current model should normally output
            # per-string scores with shape (T, 6).
            frame_onset_score_np = np.repeat(frame_onset_score_np[:, None], 6, axis=1)

        if use_peak_picking:
            frame_onset_pred = peak_pick_binary_from_scores(
                frame_onset_score_np,
                threshold=float(onset_threshold),
                smooth=peak_smooth_arg,
                pre_avg=peak_pre_avg_frames,
                post_avg=peak_post_avg_frames,
                pre_max=peak_pre_max_frames,
                post_max=peak_post_max_frames,
                combine_frames=peak_combine_frames,
            )
        else:
            frame_onset_pred = threshold_binary_from_scores(frame_onset_score_np, float(onset_threshold))

        frame_tab_gt = frame_tab_gt[:pred_len]
        frame_onset_gt = frame_onset_gt[:pred_len]

        if global_onset_logits is not None:
            global_onset_score_np = global_onset_scores_from_model_output(
                global_onset_logits,
                onset_positive_class=int(onset_positive_class),
            )[:pred_len]
        else:
            global_onset_score_np = np.max(frame_onset_score_np, axis=1)

        global_onset_gt = np.any(frame_onset_gt > 0, axis=1).astype(np.float32)

        if use_peak_picking:
            global_onset_pred = peak_pick_binary_from_scores(
                global_onset_score_np,
                threshold=float(onset_threshold),
                smooth=peak_smooth_arg,
                pre_avg=peak_pre_avg_frames,
                post_avg=peak_post_avg_frames,
                pre_max=peak_pre_max_frames,
                post_max=peak_post_max_frames,
                combine_frames=peak_combine_frames,
            )
        else:
            global_onset_pred = threshold_binary_from_scores(global_onset_score_np, float(onset_threshold))

        # ------------------------------------------------------------------
        # Dense frame-tab metrics.
        # ------------------------------------------------------------------
        frame_pred_flat = binary_tab_flat_no_rest(frame_tab_pred)
        frame_gt_flat = binary_tab_flat_no_rest(frame_tab_gt)

        frame_p, frame_r, frame_f = calculate_binary_metrics(frame_pred_flat, frame_gt_flat)

        frame_sum_p += frame_p
        frame_sum_r += frame_r
        frame_sum_f += frame_f

        frame_concat_pred = np.concatenate((frame_concat_pred, frame_pred_flat), axis=None)
        frame_concat_gt = np.concatenate((frame_concat_gt, frame_gt_flat), axis=None)

        # ------------------------------------------------------------------
        # Frame-exact onset diagnostic metrics.
        # ------------------------------------------------------------------
        onset_pred_flat = frame_onset_pred.flatten()
        onset_gt_flat = frame_onset_gt.flatten()

        exact_onset_p, exact_onset_r, exact_onset_f = calculate_binary_metrics(onset_pred_flat, onset_gt_flat)

        exact_onset_sum_p += exact_onset_p
        exact_onset_sum_r += exact_onset_r
        exact_onset_sum_f += exact_onset_f

        exact_onset_concat_pred = np.concatenate((exact_onset_concat_pred, onset_pred_flat), axis=None)
        exact_onset_concat_gt = np.concatenate((exact_onset_concat_gt, onset_gt_flat), axis=None)

        # ------------------------------------------------------------------
        # Tolerant per-string onset-only metrics: same string, +/- onset_tolerance_ms.
        # ------------------------------------------------------------------
        onset_p, onset_r, onset_f, onset_tp, onset_fp, onset_fn = tolerant_onset_precision_recall_f1(
            frame_onset_pred,
            frame_onset_gt,
            frame_seconds=frame_seconds,
            tolerance_seconds=onset_tolerance_seconds,
        )

        onset_sum_p += onset_p
        onset_sum_r += onset_r
        onset_sum_f += onset_f
        onset_sum_tp += onset_tp
        onset_sum_fp += onset_fp
        onset_sum_fn += onset_fn

        # ------------------------------------------------------------------
        # Tolerant global onset-only metrics: any string, +/- onset_tolerance_ms.
        # ------------------------------------------------------------------
        global_onset_p, global_onset_r, global_onset_f, global_onset_tp, global_onset_fp, global_onset_fn = (
            tolerant_global_onset_precision_recall_f1(
                global_onset_pred,
                global_onset_gt,
                frame_seconds=frame_seconds,
                tolerance_seconds=onset_tolerance_seconds,
            )
        )

        global_onset_sum_p += global_onset_p
        global_onset_sum_r += global_onset_r
        global_onset_sum_f += global_onset_f
        global_onset_sum_tp += global_onset_tp
        global_onset_sum_fp += global_onset_fp
        global_onset_sum_fn += global_onset_fn

        # ------------------------------------------------------------------
        # Tolerant decoded note-event metrics: same string/fret, +/- event_tolerance_ms.
        # ------------------------------------------------------------------
        pred_events = decode_events_from_global_onsets_and_tab_window(
            tab_scores=frame_tab_score_np,
            global_onset_binary=global_onset_pred,
            string_onset_scores=frame_onset_score_np,
            label_window_frames=event_label_window_frames,
            label_delay_frames=event_label_delay_frames,
            string_window_frames=event_string_window_frames,
            tab_threshold=float(event_tab_threshold),
            string_threshold=float(event_string_threshold),
            use_string_onset_filter=not bool(no_event_string_filter),
        )
        gt_events = decode_events_from_tab_and_onset(frame_tab_gt, frame_onset_gt)

        event_p, event_r, event_f, event_tp, event_fp, event_fn = event_precision_recall_f1_tolerant(
            pred_events,
            gt_events,
            frame_seconds=frame_seconds,
            tolerance_seconds=event_tolerance_seconds,
        )

        event_sum_p += event_p
        event_sum_r += event_r
        event_sum_f += event_f
        event_sum_tp += event_tp
        event_sum_fp += event_fp
        event_sum_fn += event_fn

        # ------------------------------------------------------------------
        # Save per-file predictions.
        # ------------------------------------------------------------------
        npz_save_dir = os.path.join(
            "result",
            "bpm_free_frame_onset",
            f"{trained_model}_epoch{use_model_epoch}",
            "npz",
            f"test_{fold_id}",
        )
        os.makedirs(npz_save_dir, exist_ok=True)

        npz_save_filename = os.path.join(npz_save_dir, os.path.split(npz_filename)[1])

        np.savez_compressed(
            npz_save_filename,
            input_features=input_features_np,
            frame_tab_pred_score=frame_tab_score_np,
            frame_tab_pred=frame_tab_pred,
            frame_tab_gt=frame_tab_gt,
            frame_onset_pred_score=frame_onset_score_np,
            frame_onset_pred=frame_onset_pred,
            frame_onset_gt=frame_onset_gt,
            global_onset_pred_score=global_onset_score_np,
            global_onset_pred=global_onset_pred,
            global_onset_gt=global_onset_gt,
            pred_events=np.asarray(pred_events, dtype=np.int64) if pred_events else np.zeros((0, 3), dtype=np.int64),
            gt_events=np.asarray(gt_events, dtype=np.int64) if gt_events else np.zeros((0, 3), dtype=np.int64),
            frame_seconds=np.asarray([frame_seconds], dtype=np.float32),
            onset_tolerance_ms=np.asarray([float(onset_tolerance_ms)], dtype=np.float32),
            event_tolerance_ms=np.asarray([float(event_tolerance_ms)], dtype=np.float32),
            event_label_delay_ms=np.asarray([float(event_label_delay_ms)], dtype=np.float32),
            event_label_window_ms=np.asarray([float(event_label_window_ms)], dtype=np.float32),
            event_string_window_ms=np.asarray([float(event_string_window_ms)], dtype=np.float32),
            event_tab_threshold=np.asarray([float(event_tab_threshold)], dtype=np.float32),
            event_string_threshold=np.asarray([float(event_string_threshold)], dtype=np.float32),
            no_event_string_filter=np.asarray([bool(no_event_string_filter)]),
            onset_threshold=np.asarray([float(onset_threshold)], dtype=np.float32),
            use_peak_picking=np.asarray([bool(use_peak_picking)]),
            peak_smooth_ms=np.asarray([float(peak_smooth_ms)], dtype=np.float32),
            peak_pre_avg_ms=np.asarray([float(peak_pre_avg_ms)], dtype=np.float32),
            peak_post_avg_ms=np.asarray([float(peak_post_avg_ms)], dtype=np.float32),
            peak_pre_max_ms=np.asarray([float(peak_pre_max_ms)], dtype=np.float32),
            peak_post_max_ms=np.asarray([float(peak_post_max_ms)], dtype=np.float32),
            peak_combine_ms=np.asarray([float(peak_combine_ms)], dtype=np.float32),
        )

    n_files = float(len(test_data_list))

    frame_avg_p = frame_sum_p / n_files
    frame_avg_r = frame_sum_r / n_files
    frame_avg_f = frame_sum_f / n_files
    frame_concat_p, frame_concat_r, frame_concat_f = calculate_binary_metrics(frame_concat_pred, frame_concat_gt)

    exact_onset_avg_p = exact_onset_sum_p / n_files
    exact_onset_avg_r = exact_onset_sum_r / n_files
    exact_onset_avg_f = exact_onset_sum_f / n_files
    exact_onset_concat_p, exact_onset_concat_r, exact_onset_concat_f = calculate_binary_metrics(
        exact_onset_concat_pred,
        exact_onset_concat_gt,
    )

    onset_avg_p = onset_sum_p / n_files
    onset_avg_r = onset_sum_r / n_files
    onset_avg_f = onset_sum_f / n_files
    onset_micro_p, onset_micro_r, onset_micro_f = prf_from_counts(onset_sum_tp, onset_sum_fp, onset_sum_fn)

    global_onset_avg_p = global_onset_sum_p / n_files
    global_onset_avg_r = global_onset_sum_r / n_files
    global_onset_avg_f = global_onset_sum_f / n_files
    global_onset_micro_p, global_onset_micro_r, global_onset_micro_f = prf_from_counts(
        global_onset_sum_tp,
        global_onset_sum_fp,
        global_onset_sum_fn,
    )

    event_avg_p = event_sum_p / n_files
    event_avg_r = event_sum_r / n_files
    event_avg_f = event_sum_f / n_files
    event_micro_p, event_micro_r, event_micro_f = prf_from_counts(event_sum_tp, event_sum_fp, event_sum_fn)

    if verbose:
        print(f"frame_avg_tab_p/r/f       = {frame_avg_p:.4f}, {frame_avg_r:.4f}, {frame_avg_f:.4f}")
        print(f"exact_onset_avg_p/r/f     = {exact_onset_avg_p:.4f}, {exact_onset_avg_r:.4f}, {exact_onset_avg_f:.4f}")
        print(f"tolerant_onset_avg_p/r/f        = {onset_avg_p:.4f}, {onset_avg_r:.4f}, {onset_avg_f:.4f}")
        print(f"tolerant_onset_micro_p/r/f      = {onset_micro_p:.4f}, {onset_micro_r:.4f}, {onset_micro_f:.4f}")
        print(f"global_onset_avg_p/r/f          = {global_onset_avg_p:.4f}, {global_onset_avg_r:.4f}, {global_onset_avg_f:.4f}")
        print(f"global_onset_micro_p/r/f        = {global_onset_micro_p:.4f}, {global_onset_micro_r:.4f}, {global_onset_micro_f:.4f}")
        print(f"event_avg_p/r/f                 = {event_avg_p:.4f}, {event_avg_r:.4f}, {event_avg_f:.4f}")
        print(f"event_micro_p/r/f         = {event_micro_p:.4f}, {event_micro_r:.4f}, {event_micro_f:.4f}")
        print(f"event TP/FP/FN            = {event_sum_tp}, {event_sum_fp}, {event_sum_fn}")

    # Always print the four headline metrics requested for quick terminal checks.
    # Note: this BPM-free model has no legacy beat-grid note_pred output.
    # Here note_avg_tab_f is an alias for onset-decoded note-event F1.
    print()
    print("Headline metrics")
    print(f"frame_frame_avg_tab_f = {frame_concat_f:.4f}")
    # print(f"note_avg_tab_f        = {event_avg_f:.4f}  # alias for onset-decoded event_avg_f")
    print(f"frame_avg_onset_f     = {onset_avg_f:.4f}")
    print(f"event_avg_f           = {event_avg_f:.4f}")
    print()

    result = pd.DataFrame(
        [[
            float(frame_ms),
            float(onset_tolerance_ms),
            float(event_tolerance_ms),
            float(event_label_delay_ms),
            float(event_label_window_ms),
            float(event_string_window_ms),
            float(event_tab_threshold),
            float(event_string_threshold),
            float(no_event_string_filter),
            float(onset_threshold),
            float(use_peak_picking),
            float(peak_smooth_ms),
            float(peak_pre_avg_ms),
            float(peak_post_avg_ms),
            float(peak_pre_max_ms),
            float(peak_post_max_ms),
            float(peak_combine_ms),

            frame_avg_p,
            frame_avg_r,
            frame_avg_f,
            frame_concat_p,
            frame_concat_r,
            frame_concat_f,

            # Legacy-compatible aliases for terminal/table convenience.
            frame_concat_f,  # frame_frame_avg_tab_f
            event_avg_f,    # note_avg_tab_f: onset-decoded note-event F1, not legacy note_pred

            # Primary onset columns now use tolerant matching.
            onset_avg_p,
            onset_avg_r,
            onset_avg_f,
            onset_micro_p,
            onset_micro_r,
            onset_micro_f,
            onset_sum_tp,
            onset_sum_fp,
            onset_sum_fn,

            global_onset_avg_p,
            global_onset_avg_r,
            global_onset_avg_f,
            global_onset_micro_p,
            global_onset_micro_r,
            global_onset_micro_f,
            global_onset_sum_tp,
            global_onset_sum_fp,
            global_onset_sum_fn,

            # Exact diagnostic onset columns.
            exact_onset_avg_p,
            exact_onset_avg_r,
            exact_onset_avg_f,
            exact_onset_concat_p,
            exact_onset_concat_r,
            exact_onset_concat_f,

            event_avg_p,
            event_avg_r,
            event_avg_f,
            event_micro_p,
            event_micro_r,
            event_micro_f,
            event_sum_tp,
            event_sum_fp,
            event_sum_fn,
        ]],
        columns=[
            "frame_step_ms",
            "onset_tolerance_ms",
            "event_tolerance_ms",
            "event_label_delay_ms",
            "event_label_window_ms",
            "event_string_window_ms",
            "event_tab_threshold",
            "event_string_threshold",
            "no_event_string_filter",
            "onset_threshold",
            "use_peak_picking",
            "peak_smooth_ms",
            "peak_pre_avg_ms",
            "peak_post_avg_ms",
            "peak_pre_max_ms",
            "peak_post_max_ms",
            "peak_combine_ms",

            "frame_avg_tab_p",
            "frame_avg_tab_r",
            "frame_avg_tab_f",
            "frame_concat_tab_p",
            "frame_concat_tab_r",
            "frame_concat_tab_f",

            "frame_frame_avg_tab_f",
            "note_avg_tab_f",

            "frame_avg_onset_p",
            "frame_avg_onset_r",
            "frame_avg_onset_f",
            "frame_concat_onset_p",
            "frame_concat_onset_r",
            "frame_concat_onset_f",
            "onset_tp",
            "onset_fp",
            "onset_fn",

            "global_avg_onset_p",
            "global_avg_onset_r",
            "global_avg_onset_f",
            "global_concat_onset_p",
            "global_concat_onset_r",
            "global_concat_onset_f",
            "global_onset_tp",
            "global_onset_fp",
            "global_onset_fn",

            "frame_exact_onset_avg_p",
            "frame_exact_onset_avg_r",
            "frame_exact_onset_avg_f",
            "frame_exact_onset_concat_p",
            "frame_exact_onset_concat_r",
            "frame_exact_onset_concat_f",

            "event_avg_p",
            "event_avg_r",
            "event_avg_f",
            "event_micro_p",
            "event_micro_r",
            "event_micro_f",
            "event_tp",
            "event_fp",
            "event_fn",
        ],
        index=[f"No{fold_id}"],
    )

    return result


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a BPM-free frame-tab + TCN-onset checkpoint with "
            "millisecond-based tolerances, Madmom-style peak-picking, "
            "and both per-string and global onset metrics."
        )
    )

    parser.add_argument(
        "model",
        type=str,
        help="Model run under model/. Example: dataset/run",
    )

    parser.add_argument(
        "epoch",
        type=int,
        help="Checkpoint epoch. Example: 192",
    )

    parser.add_argument(
        "--test-num",
        type=int,
        default=0,
        help="Run one test fold. Example: --test-num 0 uses 00_*.npz.",
    )

    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="Evaluate all folds from 0 to --n-folds-1 instead of only --test-num.",
    )

    parser.add_argument(
        "--n-folds",
        type=int,
        default=6,
        help="Number of folds to evaluate when --all-folds is used.",
    )

    parser.add_argument(
        "--npz-dir",
        default=None,
        help=(
            "NPZ split directory to evaluate. If omitted, reads "
            "model/<run>/run_metadata.yaml when available."
        ),
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use. Example: cpu, cuda, cuda:0. Default: cpu.",
    )

    parser.add_argument(
        "--allow-missing-hand-pos",
        action="store_true",
        help=(
            "For hand-conditioned checkpoints, use uniform frame hand priors if an NPZ "
            "is missing frame_hand_pos. Debug only; do not use for final scores."
        ),
    )

    parser.add_argument(
        "--onset-threshold",
        type=float,
        default=0.5,
        help="Threshold for sigmoid(onset_logits). Default: 0.5.",
    )

    parser.add_argument(
        "--no-peak-picking",
        action="store_true",
        help="Disable Madmom-style peak-picking and use simple thresholding instead.",
    )

    parser.add_argument(
        "--peak-smooth-ms",
        type=float,
        default=0.0,
        help="Optional activation smoothing window in ms before peak-picking. Default: 0.",
    )

    parser.add_argument(
        "--peak-pre-avg-ms",
        type=float,
        default=0.0,
        help="Moving-average past context in ms. Usually 0 for neural activations.",
    )

    parser.add_argument(
        "--peak-post-avg-ms",
        type=float,
        default=0.0,
        help="Moving-average future context in ms. Usually 0 for neural activations.",
    )

    parser.add_argument(
        "--peak-pre-max-ms",
        type=float,
        default=50.0,
        help="Past context in ms for local-maximum peak-picking. Default: 50 ms.",
    )

    parser.add_argument(
        "--peak-post-max-ms",
        type=float,
        default=50.0,
        help="Future context in ms for local-maximum peak-picking. Default: 50 ms.",
    )

    parser.add_argument(
        "--peak-combine-ms",
        type=float,
        default=30.0,
        help="Keep only the left-most onset inside this ms window. Default: 30 ms.",
    )

    parser.add_argument(
        "--onset-positive-class",
        type=int,
        default=1,
        help="For 2-class onset logits, class index interpreted as onset. Default: 1.",
    )

    parser.add_argument(
        "--onset-tolerance-ms",
        type=float,
        default=25.0,
        help="Tolerant onset matching window in milliseconds. Default: +/-25 ms.",
    )

    parser.add_argument(
        "--event-tolerance-ms",
        type=float,
        default=50.0,
        help="Tolerant note-event matching window in milliseconds. Default: +/-50 ms.",
    )

    parser.add_argument(
        "--event-label-delay-ms",
        type=float,
        default=0.0,
        help=(
            "Delay after each global onset before reading frame_tab labels, in ms. "
            "Use ~20-25 ms to skip unstable attack frames. Default: 0 ms."
        ),
    )

    parser.add_argument(
        "--event-label-window-ms",
        type=float,
        default=50.0,
        help="Post-delay window used to choose fret/string from frame_tab, in ms. Default: +50 ms.",
    )

    parser.add_argument(
        "--event-string-window-ms",
        type=float,
        default=None,
        help=(
            "Window after each global onset used for per-string onset support, in ms. "
            "Default: same as --event-label-window-ms."
        ),
    )

    parser.add_argument(
        "--event-tab-threshold",
        type=float,
        default=0.50,
        help="Minimum non-rest frame_tab probability needed to emit a note event. Default: 0.50.",
    )

    parser.add_argument(
        "--event-string-threshold",
        type=float,
        default=0.30,
        help="Minimum per-string onset probability near a global onset to emit that string. Default: 0.30.",
    )

    parser.add_argument(
        "--no-event-string-filter",
        action="store_true",
        help="Decode event strings from frame_tab only; do not require per-string onset support.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print detailed information.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    trained_model = args.model
    use_model_epoch = int(args.epoch)

    config_path = os.path.join("model", trained_model, "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    npz_dir = resolve_npz_dir(args.npz_dir, trained_model)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise EnvironmentError(f"CUDA requested but unavailable: {args.device}")

    if args.all_folds:
        test_nums = list(range(int(args.n_folds)))
    else:
        if args.test_num < 0 or args.test_num >= args.n_folds:
            raise ValueError(f"--test-num must be between 0 and {args.n_folds - 1}")
        test_nums = [int(args.test_num)]

    csv_path = os.path.join(
        "result",
        "bpm_free_frame_onset",
        trained_model + f"_epoch{use_model_epoch}",
        "metrics.csv",
    )

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    result_rows = []

    for test_num in test_nums:
        print(f"Test No. {test_num:02d}")

        fold_result = calc_score(
            test_num=test_num,
            trained_model=trained_model,
            use_model_epoch=use_model_epoch,
            config_path=config_path,
            npz_dir=npz_dir,
            device=str(args.device),
            onset_threshold=float(args.onset_threshold),
            onset_tolerance_ms=float(args.onset_tolerance_ms),
            event_tolerance_ms=float(args.event_tolerance_ms),
            use_peak_picking=not bool(args.no_peak_picking),
            peak_smooth_ms=float(args.peak_smooth_ms),
            peak_pre_avg_ms=float(args.peak_pre_avg_ms),
            peak_post_avg_ms=float(args.peak_post_avg_ms),
            peak_pre_max_ms=float(args.peak_pre_max_ms),
            peak_post_max_ms=float(args.peak_post_max_ms),
            peak_combine_ms=float(args.peak_combine_ms),
            event_label_delay_ms=float(args.event_label_delay_ms),
            event_label_window_ms=float(args.event_label_window_ms),
            event_string_window_ms=None if args.event_string_window_ms is None else float(args.event_string_window_ms),
            event_tab_threshold=float(args.event_tab_threshold),
            event_string_threshold=float(args.event_string_threshold),
            no_event_string_filter=bool(args.no_event_string_filter),
            onset_positive_class=int(args.onset_positive_class),
            allow_missing_hand_pos=bool(args.allow_missing_hand_pos),
            verbose=bool(args.verbose),
        )

        result_rows.append(fold_result)

    result = pd.concat(result_rows)

    if len(result_rows) > 1:
        result = pd.concat([result, result.describe().iloc[1:3]])

    result.to_csv(csv_path, float_format="%.3f")

    print("Saved metrics to:", csv_path)
    print(f"Onset tolerance: +/- {float(args.onset_tolerance_ms):.1f} ms")
    print(f"Event tolerance: +/- {float(args.event_tolerance_ms):.1f} ms")
    print(f"Event label delay: +{float(args.event_label_delay_ms):.1f} ms")
    print(f"Event label window: +{float(args.event_label_window_ms):.1f} ms")
    event_string_window_print = args.event_string_window_ms if args.event_string_window_ms is not None else args.event_label_window_ms
    print(f"Event string window: +{float(event_string_window_print):.1f} ms")
    print(f"Event tab threshold: {float(args.event_tab_threshold):.3f}")
    print(f"Event string threshold: {float(args.event_string_threshold):.3f}")
    print(f"Peak picking: {not bool(args.no_peak_picking)}")


if __name__ == "__main__":
    main()
