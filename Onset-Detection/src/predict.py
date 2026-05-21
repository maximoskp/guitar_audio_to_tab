#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict.py

Audio-only onset detection evaluation script.

This script evaluates an audio-only onset detector with per-string and global
onset metrics. No hand-position input or tablature prediction is required.

It evaluates:
  1. Per-string onset precision/recall/F1 using tolerant matching
     (same string, within +/- onset_tolerance_ms)
  2. Global onset precision/recall/F1 using tolerant matching
     (any string, within +/- onset_tolerance_ms)
  3. Frame-exact onset metrics as diagnostic columns
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

from network import OnsetDetector
from train import (
    build_onset_detector,
    load_frame_onset_from_npz,
    safe_torch_load,
    unwrap_state_dict,
)


# REST_CLASS for fallback onset derivation from frame_tab
REST_CLASS = 20


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


def onset_scores_from_model_output(logits: torch.Tensor) -> np.ndarray:
    """
    Convert onset model output to onset probabilities/scores.

    Shape: (T, 6), sigmoid binary logits per string.
    Returns: (T, 6) sigmoid scores.
    """
    z = torch.squeeze(logits, 0).detach().cpu()
    return torch.sigmoid(z).numpy().astype(np.float32)


def global_onset_scores_from_model_output(logits: torch.Tensor) -> np.ndarray:
    """
    Convert global onset model output to a 1D probability vector.

    Shape: (T,) sigmoid binary logits.
    Returns: (T,) sigmoid scores.
    """
    z = torch.squeeze(logits, 0).detach().cpu()
    if z.ndim == 2 and z.shape[-1] == 1:
        z = z.squeeze(-1)
    return torch.sigmoid(z).numpy().astype(np.float32)


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


def checkpoint_is_onset_only(checkpoint_path: str) -> bool:
    """Check if checkpoint contains only onset layers (no tablature)."""
    state = checkpoint_state_dict(checkpoint_path)

    has_onset = any(
        str(k).startswith("frame_onset_output_layer.")
        or str(k).startswith("global_onset_output_layer.")
        for k in state.keys()
    )

    has_tab = any(
        str(k).startswith("frame_tab_output_layer.")
        for k in state.keys()
    )

    return bool(has_onset and not has_tab)


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
    if not checkpoint_is_onset_only(model_path):
        raise RuntimeError(
            "This predict.py only supports onset-only checkpoints. "
            "The checkpoint appears to contain tablature or other incompatible layers."
        )

    input_feature_type = str(config["input_feature_type"])
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

    # Infer onset configuration from metadata or checkpoint
    checkpoint_keys = checkpoint_state_dict(model_path).keys()
    checkpoint_has_raw_onset = any(str(k).startswith("onset_raw_feature_proj.") for k in checkpoint_keys)

    onset_hidden_dim = int(metadata_or_config(metadata, config, "onset_hidden_dim", 64))
    onset_dropout = float(metadata_or_config(metadata, config, "onset_dropout", 0.25))
    onset_kernel_size = int(metadata_or_config(metadata, config, "onset_kernel_size", 3))
    onset_tcn_levels = int(metadata_or_config(metadata, config, "onset_tcn_levels", 4))
    onset_use_raw_features = bool(metadata_or_config(metadata, config, "onset_use_raw_features", checkpoint_has_raw_onset))
    onset_raw_proj_dim = int(metadata_or_config(metadata, config, "onset_raw_proj_dim", 64))
    onset_raw_dropout = float(metadata_or_config(metadata, config, "onset_raw_dropout", 0.10))
    onset_input_mode = str(metadata_or_config(metadata, config, "onset_input_mode", "full"))

    model = build_onset_detector(
        input_feature_type=input_feature_type,
        encoder_type=encoder_type,
        use_conv_stack=use_conv_stack,
        n_bins=n_bins,
        hop_length=hop_length,
        sr=sr,
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
    )

    checkpoint = checkpoint_state_dict(model_path)
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()

    info = {
        "encoder_layers": int(encoder_layers),
        "input_feature_type": input_feature_type,
        "onset_use_raw_features": bool(onset_use_raw_features),
        "onset_raw_proj_dim": int(onset_raw_proj_dim),
        "onset_raw_dropout": float(onset_raw_dropout),
        "onset_input_mode": onset_input_mode,
    }

    if verbose:
        print("model_path:", model_path)
        print("architecture: audio_onset_detection")
        print("onset_use_raw_features:", onset_use_raw_features)
        print("onset_input_mode:", onset_input_mode)
        print("onset_raw_proj_dim:", onset_raw_proj_dim)
        print("onset_hidden_dim:", onset_hidden_dim)
        print("onset_kernel_size:", onset_kernel_size)
        print("onset_tcn_levels:", onset_tcn_levels)

    return model, info


# -----------------------------------------------------------------------------
# Tolerant evaluation
# -----------------------------------------------------------------------------


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
    use_peak_picking: bool = True,
    peak_smooth_ms: float = 0.0,
    peak_pre_avg_ms: float = 0.0,
    peak_post_avg_ms: float = 0.0,
    peak_pre_max_ms: float = 50.0,
    peak_post_max_ms: float = 50.0,
    peak_combine_ms: float = 30.0,
    verbose: bool = False,
) -> pd.DataFrame:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    metadata_path = os.path.join("model", trained_model, "run_metadata.yaml")
    metadata = load_yaml_if_exists(metadata_path)

    input_feature_type = str(config["input_feature_type"])
    fold_id = f"{test_num:02d}"

    frame_seconds = frame_seconds_from_config(config)
    frame_ms = frame_seconds * 1000.0
    onset_tolerance_seconds = float(onset_tolerance_ms) / 1000.0

    peak_smooth_frames = ms_to_frames(peak_smooth_ms, frame_seconds)
    peak_pre_avg_frames = ms_to_frames(peak_pre_avg_ms, frame_seconds)
    peak_post_avg_frames = ms_to_frames(peak_post_avg_ms, frame_seconds)
    peak_pre_max_frames = ms_to_frames(peak_pre_max_ms, frame_seconds)
    peak_post_max_frames = ms_to_frames(peak_post_max_ms, frame_seconds)
    peak_combine_frames = ms_to_frames(peak_combine_ms, frame_seconds)
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

    for npz_filename in tqdm.tqdm(test_data_list):
        npz_file = np.load(npz_filename, allow_pickle=True)

        if input_feature_type == "cqt":
            input_features_np = npz_file["cqt"].astype(np.float32)
        elif input_feature_type == "melspec":
            input_features_np = npz_file["mel_spec"].astype(np.float32)
        else:
            raise ValueError(f"Unknown input_feature_type: {input_feature_type}")

        # Load onset targets, with fallback to frame_tab if needed
        frame_tab = npz_file.get("frame_tab", None)
        if frame_tab is not None:
            frame_tab = frame_tab.astype(np.float32)

        try:
            frame_onset_gt = load_frame_onset_from_npz(npz_file, frame_tab).astype(np.float32)
        except KeyError as e:
            print(f"[warn] {npz_filename}: {e}, skipping")
            continue

        target_len = min(input_features_np.shape[0], frame_onset_gt.shape[0])

        input_features_np = input_features_np[:target_len]
        frame_onset_gt = frame_onset_gt[:target_len]

        input_features = torch.from_numpy(input_features_np).float().unsqueeze(0).to(device_obj)
        frame_len = torch.tensor([target_len], dtype=torch.long, device=device_obj)

        with torch.no_grad():
            frame_onset_logits, global_onset_logits, olens = model(
                input_features,
                frame_len,
            )

        pred_len = int(olens[0].item())

        frame_onset_score_np = onset_scores_from_model_output(
            frame_onset_logits,
        )[:pred_len]

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

        frame_onset_gt = frame_onset_gt[:pred_len]

        global_onset_score_np = global_onset_scores_from_model_output(
            global_onset_logits,
        )[:pred_len]

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
        # Save per-file predictions.
        # ------------------------------------------------------------------
        npz_save_dir = os.path.join(
            "result",
            "audio_onset_detection",
            f"{trained_model}_epoch{use_model_epoch}",
            "npz",
            f"test_{fold_id}",
        )
        os.makedirs(npz_save_dir, exist_ok=True)

        npz_save_filename = os.path.join(npz_save_dir, os.path.split(npz_filename)[1])

        np.savez_compressed(
            npz_save_filename,
            input_features=input_features_np,
            frame_onset_pred_score=frame_onset_score_np,
            frame_onset_pred=frame_onset_pred,
            frame_onset_gt=frame_onset_gt,
            global_onset_pred_score=global_onset_score_np,
            global_onset_pred=global_onset_pred,
            global_onset_gt=global_onset_gt,
            frame_seconds=np.asarray([frame_seconds], dtype=np.float32),
            onset_tolerance_ms=np.asarray([float(onset_tolerance_ms)], dtype=np.float32),
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

    if verbose:
        print(f"exact_onset_avg_p/r/f     = {exact_onset_avg_p:.4f}, {exact_onset_avg_r:.4f}, {exact_onset_avg_f:.4f}")
        print(f"tolerant_onset_avg_p/r/f  = {onset_avg_p:.4f}, {onset_avg_r:.4f}, {onset_avg_f:.4f}")
        print(f"tolerant_onset_micro_p/r/f = {onset_micro_p:.4f}, {onset_micro_r:.4f}, {onset_micro_f:.4f}")
        print(f"global_onset_avg_p/r/f    = {global_onset_avg_p:.4f}, {global_onset_avg_r:.4f}, {global_onset_avg_f:.4f}")
        print(f"global_onset_micro_p/r/f  = {global_onset_micro_p:.4f}, {global_onset_micro_r:.4f}, {global_onset_micro_f:.4f}")

    # Print headline metrics for quick terminal checks
    print()
    print("Headline metrics")
    print(f"frame_avg_onset_f     = {onset_avg_f:.4f}")
    print(f"global_onset_f        = {global_onset_avg_f:.4f}")
    print()

    result = pd.DataFrame(
        [[
            float(frame_ms),
            float(onset_tolerance_ms),
            float(onset_threshold),
            float(use_peak_picking),
            float(peak_smooth_ms),
            float(peak_pre_avg_ms),
            float(peak_post_avg_ms),
            float(peak_pre_max_ms),
            float(peak_post_max_ms),
            float(peak_combine_ms),

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

            exact_onset_avg_p,
            exact_onset_avg_r,
            exact_onset_avg_f,
            exact_onset_concat_p,
            exact_onset_concat_r,
            exact_onset_concat_f,
        ]],
        columns=[
            "frame_step_ms",
            "onset_tolerance_ms",
            "onset_threshold",
            "use_peak_picking",
            "peak_smooth_ms",
            "peak_pre_avg_ms",
            "peak_post_avg_ms",
            "peak_pre_max_ms",
            "peak_post_max_ms",
            "peak_combine_ms",

            "frame_avg_onset_p",
            "frame_avg_onset_r",
            "frame_avg_onset_f",
            "frame_concat_onset_p",
            "frame_concat_onset_r",
            "frame_concat_onset_f",
            "onset_tp",
            "onset_fp",
            "onset_fn",

            "global_onset_p",
            "global_onset_r",
            "global_onset_f",
            "global_onset_micro_p",
            "global_onset_micro_r",
            "global_onset_micro_f",
            "global_onset_tp",
            "global_onset_fp",
            "global_onset_fn",

            "frame_exact_onset_avg_p",
            "frame_exact_onset_avg_r",
            "frame_exact_onset_avg_f",
            "frame_exact_onset_concat_p",
            "frame_exact_onset_concat_r",
            "frame_exact_onset_concat_f",
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
            "Evaluate an audio-only onset detection checkpoint with "
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
        help="Moving-maximum past context in ms. Default: 50 ms.",
    )

    parser.add_argument(
        "--peak-post-max-ms",
        type=float,
        default=50.0,
        help="Moving-maximum future context in ms. Default: 50 ms.",
    )

    parser.add_argument(
        "--peak-combine-ms",
        type=float,
        default=30.0,
        help="Combine multiple peaks within this many ms into one. Default: 30 ms.",
    )

    parser.add_argument(
        "--onset-tolerance-ms",
        type=float,
        default=25.0,
        help="Tolerant matching window for onsets. Default: +/- 25 ms.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config_path = "src/config.yaml"
    npz_dir = resolve_npz_dir(args.npz_dir, args.model)

    test_nums = range(args.n_folds) if args.all_folds else [args.test_num]

    results = []

    for test_num in test_nums:
        try:
            result = calc_score(
                test_num=test_num,
                trained_model=args.model,
                use_model_epoch=args.epoch,
                config_path=config_path,
                npz_dir=npz_dir,
                device=args.device,
                onset_threshold=args.onset_threshold,
                onset_tolerance_ms=args.onset_tolerance_ms,
                use_peak_picking=not args.no_peak_picking,
                peak_smooth_ms=args.peak_smooth_ms,
                peak_pre_avg_ms=args.peak_pre_avg_ms,
                peak_post_avg_ms=args.peak_post_avg_ms,
                peak_pre_max_ms=args.peak_pre_max_ms,
                peak_post_max_ms=args.peak_post_max_ms,
                peak_combine_ms=args.peak_combine_ms,
                verbose=True,
            )
            results.append(result)
        except FileNotFoundError as e:
            print(f"[skip fold {test_num:02d}] {e}")
            continue

    if not results:
        print("No results to save.")
        return

    combined = pd.concat(results)

    result_dir = os.path.join("result", "audio_onset_detection", f"{args.model}_epoch{args.epoch}")
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, "metrics.csv")
    combined.to_csv(result_path)

    print(f"\nResults saved to: {result_path}")
    print(combined)


if __name__ == "__main__":
    main()
