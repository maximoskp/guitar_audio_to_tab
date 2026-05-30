#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict.py

Evaluation script for the chord-aware event assembly version.

It evaluates:
  1. dense frame-tab metrics;
  2. auxiliary per-string/global onset metrics;
  3. learned event assembly metrics using event head outputs.

The final symbolic events are decoded from:
  - event_logits        -> candidate event times
  - event_string_logits -> selected strings for each event
  - event_fret_logits   -> fret labels per selected string

The event head is intended to reduce hand-written chord/arpeggio heuristics.
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
    derive_event_type_targets,
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


def metadata_or_config(metadata: Dict[str, Any], config: Dict[str, Any], key: str, default: Any) -> Any:
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
    return np.asarray(tab_one_hot[:, :, :REST_CLASS]).flatten()


def frame_seconds_from_config(config: Dict[str, Any]) -> float:
    sr = float(config["down_sampling_rate"])
    hop_length = float(config["hop_length"])
    if sr <= 0 or hop_length <= 0:
        raise ValueError("Invalid sample rate or hop length.")
    return hop_length / sr


def ms_to_frames(milliseconds: float, frame_seconds: float) -> int:
    if milliseconds is None:
        return 0
    if frame_seconds <= 0:
        raise ValueError(f"Invalid frame_seconds: {frame_seconds}")
    return max(0, int(round((float(milliseconds) / 1000.0) / float(frame_seconds))))


def smooth_signal(activations: np.ndarray, smooth: Optional[int] = None) -> np.ndarray:
    x = np.asarray(activations, dtype=np.float32)
    if smooth is None:
        return x.copy()
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

    raise ValueError("activations must be 1D or 2D")


def _moving_window_reduce_1d(x: np.ndarray, pre: int, post: int, reducer: str) -> np.ndarray:
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
    x = np.asarray(activations, dtype=np.float32)
    if pre_avg + post_avg + 1 <= 1:
        return np.zeros_like(x, dtype=np.float32)
    if x.ndim == 1:
        return _moving_window_reduce_1d(x, pre_avg, post_avg, "mean")
    if x.ndim == 2:
        out = np.zeros_like(x, dtype=np.float32)
        for c in range(x.shape[1]):
            out[:, c] = _moving_window_reduce_1d(x[:, c], pre_avg, post_avg, "mean")
        return out
    raise ValueError("activations must be 1D or 2D")


def moving_maximum(activations: np.ndarray, pre_max: int = 1, post_max: int = 1) -> np.ndarray:
    x = np.asarray(activations, dtype=np.float32)
    if pre_max + post_max + 1 <= 1:
        return x.copy()
    if x.ndim == 1:
        return _moving_window_reduce_1d(x, pre_max, post_max, "max")
    if x.ndim == 2:
        out = np.zeros_like(x, dtype=np.float32)
        for c in range(x.shape[1]):
            out[:, c] = _moving_window_reduce_1d(x[:, c], pre_max, post_max, "max")
        return out
    raise ValueError("activations must be 1D or 2D")


def peak_picking(
    activations: np.ndarray,
    threshold: float,
    smooth: Optional[int] = None,
    pre_avg: int = 0,
    post_avg: int = 0,
    pre_max: int = 1,
    post_max: int = 1,
):
    x = smooth_signal(np.asarray(activations, dtype=np.float32), smooth=smooth)
    if x.ndim not in (1, 2):
        raise ValueError("activations must be 1D or 2D")

    mov_avg = moving_average(x, pre_avg=pre_avg, post_avg=post_avg)
    detections = x * (x >= (mov_avg + float(threshold)))

    if int(pre_max) + int(post_max) + 1 > 1:
        mov_max = moving_maximum(detections, pre_max=pre_max, post_max=post_max)
        detections = detections * (detections == mov_max)

    return np.nonzero(detections)


def combine_event_frames(frames: Sequence[int], combine_frames: int = 0) -> List[int]:
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
        peak_frames = combine_event_frames(peaks[0].tolist(), combine_frames=combine_frames)
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

    raise ValueError("scores must be 1D or 2D")


def threshold_binary_from_scores(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(scores, dtype=np.float32) >= float(threshold)).astype(np.float32)


def onset_scores_from_model_output(logits: torch.Tensor) -> np.ndarray:
    z = torch.squeeze(logits, 0).detach().cpu()
    if z.ndim == 2:
        return torch.sigmoid(z).numpy().astype(np.float32)
    raise RuntimeError(f"Unexpected onset logits shape: {tuple(z.shape)}")


def global_onset_scores_from_model_output(logits: torch.Tensor) -> np.ndarray:
    z = torch.squeeze(logits, 0).detach().cpu()
    if z.ndim == 1:
        return torch.sigmoid(z).numpy().astype(np.float32)
    if z.ndim == 2 and z.shape[-1] == 1:
        return torch.sigmoid(z[:, 0]).numpy().astype(np.float32)
    raise RuntimeError(f"Unexpected global onset logits shape: {tuple(z.shape)}")


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


def checkpoint_is_event_assembly(checkpoint_path: str) -> bool:
    state = checkpoint_state_dict(checkpoint_path)
    return any(str(k).startswith("event_assembly_head.") for k in state.keys())


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
    if not checkpoint_is_event_assembly(model_path):
        raise RuntimeError(
            "This predict.py expects a checkpoint from the chord-aware event assembly network."
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
            print("[info] Checkpoint contains hand-position layers; forcing use_hand_position=True.")

    hand_pos_dim = int(metadata_or_config(metadata, config, "hand_pos_dim", 20))
    hand_position_fusion = str(metadata_or_config(metadata, config, "hand_position_fusion", "hidden+prior"))
    hand_hidden_gate_init = float(metadata_or_config(metadata, config, "hand_hidden_gate_init", 0.5))
    hand_prior_strength = float(metadata_or_config(metadata, config, "hand_prior_strength", 0.35))
    hand_span = int(metadata_or_config(metadata, config, "hand_span", 4))

    onset_hidden_dim = int(metadata_or_config(metadata, config, "onset_hidden_dim", 64))
    onset_dropout = float(metadata_or_config(metadata, config, "onset_dropout", 0.25))
    onset_kernel_size = int(metadata_or_config(metadata, config, "onset_kernel_size", 3))
    onset_tcn_levels = int(metadata_or_config(metadata, config, "onset_tcn_levels", 4))
    onset_use_raw_features = bool(metadata_or_config(metadata, config, "onset_use_raw_features", True))
    onset_raw_proj_dim = int(metadata_or_config(metadata, config, "onset_raw_proj_dim", 64))
    onset_raw_dropout = float(metadata_or_config(metadata, config, "onset_raw_dropout", 0.10))
    onset_input_mode = str(metadata_or_config(metadata, config, "onset_input_mode", "full"))

    event_head_hidden_dim = int(metadata_or_config(metadata, config, "event_head_hidden_dim", 128))
    event_head_tcn_levels = int(metadata_or_config(metadata, config, "event_head_tcn_levels", 6))
    event_head_kernel_size = int(metadata_or_config(metadata, config, "event_head_kernel_size", 5))
    event_head_dropout = float(metadata_or_config(metadata, config, "event_head_dropout", 0.25))
    event_input_uses_tab_probs = bool(metadata_or_config(metadata, config, "event_input_uses_tab_probs", True))
    detach_tab_probs_for_event = bool(metadata_or_config(metadata, config, "detach_tab_probs_for_event", False))

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
        onset_input_mode=onset_input_mode,
        event_head_hidden_dim=event_head_hidden_dim,
        event_head_tcn_levels=event_head_tcn_levels,
        event_head_kernel_size=event_head_kernel_size,
        event_head_dropout=event_head_dropout,
        event_input_uses_tab_probs=event_input_uses_tab_probs,
        detach_tab_probs_for_event=detach_tab_probs_for_event,
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
        "event_head_tcn_levels": int(event_head_tcn_levels),
        "onset_input_mode": onset_input_mode,
    }

    if verbose:
        print("model_path:", model_path)
        print("architecture: bpm_free_frame_tab_onset_chord_aware_event_assembly")
        print("use_hand_position:", use_hand_position)
        print("onset_input_mode:", onset_input_mode)
        print("event_head_tcn_levels:", event_head_tcn_levels)

    return model, info


# -----------------------------------------------------------------------------
# Decoding and evaluation
# -----------------------------------------------------------------------------


def decode_events_from_tab_and_onset(tab_one_hot: np.ndarray, onset_binary: np.ndarray) -> List[Tuple[int, int, int]]:
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


def decode_events_from_event_head(
    event_scores: np.ndarray,
    event_string_scores: np.ndarray,
    event_fret_probs: np.ndarray,
    event_threshold: float = 0.50,
    string_threshold: float = 0.50,
    fret_threshold: float = 0.35,
    label_window_frames: int = 1,
    label_delay_frames: int = 0,
    combine_frames: int = 1,
    peak_picking_enabled: bool = True,
) -> List[Tuple[int, int, int]]:
    """
    Decode learned event head outputs.

    Unlike the old decoder, this does not require per-string onset peaks. The
    event head predicts event times and string masks directly. This is intended
    to handle chords and arpeggios in one learned mechanism.
    """
    event_scores = np.asarray(event_scores, dtype=np.float32).reshape(-1)
    event_string_scores = np.asarray(event_string_scores, dtype=np.float32)
    event_fret_probs = np.asarray(event_fret_probs, dtype=np.float32)

    T = min(event_scores.shape[0], event_string_scores.shape[0], event_fret_probs.shape[0])
    if T <= 0:
        return []

    if peak_picking_enabled:
        event_binary = peak_pick_binary_from_scores(
            event_scores[:T],
            threshold=float(event_threshold),
            pre_avg=0,
            post_avg=0,
            pre_max=1,
            post_max=1,
            combine_frames=combine_frames,
        )
    else:
        event_binary = threshold_binary_from_scores(event_scores[:T], float(event_threshold))

    events: List[Tuple[int, int, int]] = []

    label_window_frames = max(0, int(label_window_frames))
    label_delay_frames = max(0, int(label_delay_frames))

    for t in np.where(event_binary[:T] > 0)[0].tolist():
        label_start = min(T, int(t) + label_delay_frames)
        label_end = min(T, label_start + label_window_frames + 1)
        if label_end <= label_start:
            continue

        string_window = event_string_scores[label_start:label_end]
        fret_window = event_fret_probs[label_start:label_end, :, :REST_CLASS]

        for s in range(6):
            s_score = float(np.max(string_window[:, s]))
            if s_score < float(string_threshold):
                continue

            flat_idx = int(np.argmax(fret_window[:, s, :]))
            local_t, fret = np.unravel_index(flat_idx, fret_window[:, s, :].shape)
            fret_score = float(fret_window[local_t, s, fret])

            if fret_score < float(fret_threshold):
                continue

            events.append((int(t), int(s), int(fret)))

    return events


def tolerant_onset_precision_recall_f1(
    pred_onset_binary: np.ndarray,
    gt_onset_binary: np.ndarray,
    frame_seconds: float,
    tolerance_seconds: float,
) -> Tuple[float, float, float, int, int, int]:
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
    pred_frames = np.where(np.asarray(pred_global_onset_binary).reshape(-1) > 0)[0].tolist()
    gt_frames = np.where(np.asarray(gt_global_onset_binary).reshape(-1) > 0)[0].tolist()

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
    p, r, f = prf_from_counts(tp, fp, fn)
    return p, r, f, int(tp), int(fp), int(fn)


def event_precision_recall_f1_tolerant(
    pred_events: Sequence[Tuple[int, int, int]],
    gt_events: Sequence[Tuple[int, int, int]],
    frame_seconds: float,
    tolerance_seconds: float,
) -> Tuple[float, float, float, int, int, int]:
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
    p, r, f = prf_from_counts(tp, fp, fn)
    return p, r, f, int(tp), int(fp), int(fn)


# -----------------------------------------------------------------------------
# Main scoring
# -----------------------------------------------------------------------------


def calc_score(
    test_num: int,
    trained_model: str,
    use_model_epoch: int,
    config_path: str,
    npz_dir: str,
    verbose: bool = True,
    device: str = "cpu",
    allow_missing_hand_pos: bool = False,
    onset_threshold: float = 0.80,
    global_onset_threshold: float = 0.80,
    event_threshold: float = 0.50,
    event_string_threshold: float = 0.50,
    event_fret_threshold: float = 0.35,
    onset_tolerance_ms: float = 25.0,
    event_tolerance_ms: float = 50.0,
    event_label_delay_ms: float = 0.0,
    event_label_window_ms: float = 50.0,
    peak_picking_enabled: bool = True,
) -> pd.DataFrame:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    metadata_path = os.path.join("model", trained_model, "run_metadata.yaml")
    metadata = load_yaml_if_exists(metadata_path)

    frame_seconds = frame_seconds_from_config(config)
    event_label_delay_frames = ms_to_frames(event_label_delay_ms, frame_seconds)
    event_label_window_frames = ms_to_frames(event_label_window_ms, frame_seconds)
    onset_tolerance_seconds = float(onset_tolerance_ms) / 1000.0
    event_tolerance_seconds = float(event_tolerance_ms) / 1000.0

    input_feature_type = str(config["input_feature_type"])
    mode = str(config["mode"])
    if mode != "tab":
        raise ValueError("This predict.py supports tab mode only.")

    fold_id = f"{test_num:02d}"

    model_path = os.path.join(
        "model",
        trained_model,
        f"testNo{fold_id}",
        f"epoch{use_model_epoch}.model",
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    device_obj = torch.device(device)

    model, info = build_model_for_prediction(
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
        print(f"test_num={test_num}, mode={mode}")
        print("npz_dir:", npz_dir)
        print("test files:", len(test_data_list))

    rows = []
    frame_concat_pred = np.array([], dtype=np.float32)
    frame_concat_gt = np.array([], dtype=np.float32)

    onset_tp_total = onset_fp_total = onset_fn_total = 0
    global_tp_total = global_fp_total = global_fn_total = 0
    event_tp_total = event_fp_total = event_fn_total = 0

    for npz_filename in tqdm.tqdm(test_data_list):
        npz_file = np.load(npz_filename, allow_pickle=True)

        if input_feature_type == "cqt":
            input_features_np = npz_file["cqt"].astype(np.float32)
        elif input_feature_type == "melspec":
            input_features_np = npz_file["mel_spec"].astype(np.float32)
        else:
            raise ValueError(f"Unknown input_feature_type: {input_feature_type}")

        if "frame_tab" not in npz_file.files:
            raise KeyError(f"{npz_filename} does not contain frame_tab")

        frame_tab_gt_np = npz_file["frame_tab"].astype(np.float32)
        frame_onset_gt_np = load_frame_onset_from_npz(npz_file, frame_tab_gt_np).astype(np.float32)

        target_len = min(input_features_np.shape[0], frame_tab_gt_np.shape[0], frame_onset_gt_np.shape[0])
        input_features_np = input_features_np[:target_len]
        frame_tab_gt_np = frame_tab_gt_np[:target_len]
        frame_onset_gt_np = frame_onset_gt_np[:target_len]

        input_features = torch.from_numpy(input_features_np).float().unsqueeze(0).to(device_obj)
        frame_len = torch.tensor([target_len], dtype=torch.long, device=device_obj)

        frame_hand_pos = None
        if info["use_hand_position"]:
            frame_hand_pos = load_frame_hand_tensor_from_npz(
                npz_file=npz_file,
                npz_filename=npz_filename,
                device=device_obj,
                hand_pos_dim=int(info["hand_pos_dim"]),
                target_len=target_len,
                allow_missing_hand_pos=allow_missing_hand_pos,
            )

        with torch.no_grad():
            (
                frame_tab_pred,
                frame_onset_logits,
                global_onset_logits,
                event_logits,
                event_string_logits,
                event_fret_logits,
                event_type_logits,
                olens,
            ) = model(
                input_features,
                frame_len,
                frame_hand_pos=frame_hand_pos,
            )

        pred_len = int(olens[0].detach().cpu().item())
        frame_tab_pred = frame_tab_pred[:, :pred_len]
        frame_onset_logits = frame_onset_logits[:, :pred_len]
        global_onset_logits = global_onset_logits[:, :pred_len]
        event_logits = event_logits[:, :pred_len]
        event_string_logits = event_string_logits[:, :pred_len]
        event_fret_logits = event_fret_logits[:, :pred_len]

        frame_gt_np = frame_tab_gt_np[:pred_len]
        onset_gt_np = frame_onset_gt_np[:pred_len]
        global_onset_gt_np = np.max(onset_gt_np, axis=1).astype(np.float32)

        frame_pred_onehot = one_hot_argmax_tab(torch.squeeze(frame_tab_pred, 0))
        frame_p, frame_r, frame_f = calculate_binary_metrics(
            binary_tab_flat_no_rest(frame_pred_onehot),
            binary_tab_flat_no_rest(frame_gt_np),
        )

        frame_concat_pred = np.concatenate([frame_concat_pred, binary_tab_flat_no_rest(frame_pred_onehot)])
        frame_concat_gt = np.concatenate([frame_concat_gt, binary_tab_flat_no_rest(frame_gt_np)])

        onset_scores = onset_scores_from_model_output(frame_onset_logits)
        global_onset_scores = global_onset_scores_from_model_output(global_onset_logits)
        event_scores = torch.sigmoid(torch.squeeze(event_logits, 0)).detach().cpu().numpy().astype(np.float32)
        event_string_scores = torch.sigmoid(torch.squeeze(event_string_logits, 0)).detach().cpu().numpy().astype(np.float32)
        event_fret_probs = torch.softmax(torch.squeeze(event_fret_logits, 0), dim=-1).detach().cpu().numpy().astype(np.float32)

        if peak_picking_enabled:
            onset_pred_binary = peak_pick_binary_from_scores(
                onset_scores,
                threshold=float(onset_threshold),
                pre_avg=0,
                post_avg=0,
                pre_max=1,
                post_max=1,
                combine_frames=0,
            )
            global_onset_pred_binary = peak_pick_binary_from_scores(
                global_onset_scores,
                threshold=float(global_onset_threshold),
                pre_avg=0,
                post_avg=0,
                pre_max=1,
                post_max=1,
                combine_frames=0,
            )
        else:
            onset_pred_binary = threshold_binary_from_scores(onset_scores, onset_threshold)
            global_onset_pred_binary = threshold_binary_from_scores(global_onset_scores, global_onset_threshold)

        onset_p, onset_r, onset_f, onset_tp, onset_fp, onset_fn = tolerant_onset_precision_recall_f1(
            onset_pred_binary,
            onset_gt_np,
            frame_seconds=frame_seconds,
            tolerance_seconds=onset_tolerance_seconds,
        )
        global_p, global_r, global_f, global_tp, global_fp, global_fn = tolerant_global_onset_precision_recall_f1(
            global_onset_pred_binary,
            global_onset_gt_np,
            frame_seconds=frame_seconds,
            tolerance_seconds=onset_tolerance_seconds,
        )

        event_pred = decode_events_from_event_head(
            event_scores=event_scores,
            event_string_scores=event_string_scores,
            event_fret_probs=event_fret_probs,
            event_threshold=event_threshold,
            string_threshold=event_string_threshold,
            fret_threshold=event_fret_threshold,
            label_window_frames=event_label_window_frames,
            label_delay_frames=event_label_delay_frames,
            combine_frames=1,
            peak_picking_enabled=peak_picking_enabled,
        )
        event_gt = decode_events_from_tab_and_onset(frame_gt_np, onset_gt_np)

        event_p, event_r, event_f, event_tp, event_fp, event_fn = event_precision_recall_f1_tolerant(
            event_pred,
            event_gt,
            frame_seconds=frame_seconds,
            tolerance_seconds=event_tolerance_seconds,
        )

        onset_tp_total += onset_tp
        onset_fp_total += onset_fp
        onset_fn_total += onset_fn
        global_tp_total += global_tp
        global_fp_total += global_fp
        global_fn_total += global_fn
        event_tp_total += event_tp
        event_fp_total += event_fp
        event_fn_total += event_fn

        rows.append({
            "file": os.path.basename(npz_filename),
            "frame_tab_p": frame_p,
            "frame_tab_r": frame_r,
            "frame_tab_f": frame_f,
            "onset_p": onset_p,
            "onset_r": onset_r,
            "onset_f": onset_f,
            "global_onset_p": global_p,
            "global_onset_r": global_r,
            "global_onset_f": global_f,
            "event_head_p": event_p,
            "event_head_r": event_r,
            "event_head_f": event_f,
            "event_head_tp": event_tp,
            "event_head_fp": event_fp,
            "event_head_fn": event_fn,
            "pred_events": len(event_pred),
            "gt_events": len(event_gt),
        })

    df = pd.DataFrame(rows)

    frame_concat_p, frame_concat_r, frame_concat_f = calculate_binary_metrics(frame_concat_pred, frame_concat_gt)
    onset_micro_p, onset_micro_r, onset_micro_f = prf_from_counts(onset_tp_total, onset_fp_total, onset_fn_total)
    global_micro_p, global_micro_r, global_micro_f = prf_from_counts(global_tp_total, global_fp_total, global_fn_total)
    event_micro_p, event_micro_r, event_micro_f = prf_from_counts(event_tp_total, event_fp_total, event_fn_total)

    summary = {
        "file": "__SUMMARY__",
        "frame_tab_p": float(df["frame_tab_p"].mean()) if len(df) else 0.0,
        "frame_tab_r": float(df["frame_tab_r"].mean()) if len(df) else 0.0,
        "frame_tab_f": float(df["frame_tab_f"].mean()) if len(df) else 0.0,
        "frame_concat_tab_p": frame_concat_p,
        "frame_concat_tab_r": frame_concat_r,
        "frame_concat_tab_f": frame_concat_f,
        "onset_p": onset_micro_p,
        "onset_r": onset_micro_r,
        "onset_f": onset_micro_f,
        "global_onset_p": global_micro_p,
        "global_onset_r": global_micro_r,
        "global_onset_f": global_micro_f,
        "event_head_p": event_micro_p,
        "event_head_r": event_micro_r,
        "event_head_f": event_micro_f,
        "event_head_tp": event_tp_total,
        "event_head_fp": event_fp_total,
        "event_head_fn": event_fn_total,
        "pred_events": int(df["pred_events"].sum()) if len(df) else 0,
        "gt_events": int(df["gt_events"].sum()) if len(df) else 0,
    }
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

    result_dir = os.path.join(
        "result",
        "event_assembly",
        f"{trained_model}_epoch{use_model_epoch}",
    )
    os.makedirs(result_dir, exist_ok=True)
    out_path = os.path.join(result_dir, "metrics.csv")
    df.to_csv(out_path, index=False)

    print("\nHeadline metrics")
    print(f"frame_tab_f      = {summary['frame_tab_f']:.4f}")
    print(f"onset_f         = {summary['onset_f']:.4f}")
    print(f"global_onset_f  = {summary['global_onset_f']:.4f}")
    print(f"event_head_f    = {summary['event_head_f']:.4f}")
    print("Saved metrics to:", out_path)

    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate BPM-free frame-tab + onset + chord-aware event assembly checkpoint."
    )
    parser.add_argument("trained_model", help="Model run path under model/, e.g. dataset/run_name")
    parser.add_argument("epoch", type=int)
    parser.add_argument("--test-num", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=6)
    parser.add_argument("--npz-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-missing-hand-pos", action="store_true")

    parser.add_argument("--onset-threshold", type=float, default=0.80)
    parser.add_argument("--global-onset-threshold", type=float, default=0.80)
    parser.add_argument("--event-threshold", type=float, default=0.50)
    parser.add_argument("--event-string-threshold", type=float, default=0.50)
    parser.add_argument("--event-fret-threshold", type=float, default=0.35)
    parser.add_argument("--onset-tolerance-ms", type=float, default=25.0)
    parser.add_argument("--event-tolerance-ms", type=float, default=50.0)
    parser.add_argument("--event-label-delay-ms", type=float, default=0.0)
    parser.add_argument("--event-label-window-ms", type=float, default=50.0)
    parser.add_argument("--no-peak-picking", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.config is not None:
        config_path = args.config
    else:
        config_path = os.path.join("model", args.trained_model, "config.yaml")
        if not os.path.exists(config_path):
            config_path = "src/config.yaml"

    npz_dir = resolve_npz_dir(args.npz_dir, args.trained_model)

    calc_score(
        test_num=int(args.test_num),
        trained_model=args.trained_model,
        use_model_epoch=int(args.epoch),
        config_path=config_path,
        npz_dir=npz_dir,
        verbose=bool(args.verbose),
        device=args.device,
        allow_missing_hand_pos=bool(args.allow_missing_hand_pos),
        onset_threshold=float(args.onset_threshold),
        global_onset_threshold=float(args.global_onset_threshold),
        event_threshold=float(args.event_threshold),
        event_string_threshold=float(args.event_string_threshold),
        event_fret_threshold=float(args.event_fret_threshold),
        onset_tolerance_ms=float(args.onset_tolerance_ms),
        event_tolerance_ms=float(args.event_tolerance_ms),
        event_label_delay_ms=float(args.event_label_delay_ms),
        event_label_window_ms=float(args.event_label_window_ms),
        peak_picking_enabled=not bool(args.no_peak_picking),
    )


if __name__ == "__main__":
    main()
