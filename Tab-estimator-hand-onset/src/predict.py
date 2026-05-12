#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict.py

BPM-free frame-level tablature + onset evaluation script.

This version evaluates the non-causal TCN-onset BPM-free model and uses
millisecond-based tolerances instead of hardcoded frame tolerances:

  - onset matching:     same string, onset within +/- onset_tolerance_ms
  - note-event matching: same string, same fret, onset within +/- event_tolerance_ms

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
     using a tolerant matching window, default +/-50 ms

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

    has_onset = any(str(k).startswith("frame_onset_output_layer.") for k in state.keys())

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

    onset_hidden_dim = int(metadata_or_config(metadata, config, "onset_hidden_dim", 64))
    onset_dropout = float(metadata_or_config(metadata, config, "onset_dropout", 0.25))
    onset_kernel_size = int(metadata_or_config(metadata, config, "onset_kernel_size", 5))
    onset_tcn_levels = int(metadata_or_config(metadata, config, "onset_tcn_levels", 3))

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
    }

    if verbose:
        print("model_path:", model_path)
        print("architecture: bpm_free_frame_tab_onset_tcn")
        print("use_hand_position:", use_hand_position)
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
    Convert dense frame tab + binary onsets to note events.

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
    onset_tolerance_ms: float = 50.0,
    event_tolerance_ms: float = 50.0,
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
        print(f"onset threshold: {float(onset_threshold):.3f}")

    # Dense frame-tab metrics.
    frame_sum_p = frame_sum_r = frame_sum_f = 0.0
    frame_concat_pred = np.array([], dtype=np.float32)
    frame_concat_gt = np.array([], dtype=np.float32)

    # Frame-exact onset diagnostics.
    exact_onset_sum_p = exact_onset_sum_r = exact_onset_sum_f = 0.0
    exact_onset_concat_pred = np.array([], dtype=np.float32)
    exact_onset_concat_gt = np.array([], dtype=np.float32)

    # Tolerant onset metrics.
    onset_sum_p = onset_sum_r = onset_sum_f = 0.0
    onset_sum_tp = onset_sum_fp = onset_sum_fn = 0

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
            frame_tab_score, frame_onset_logits, olens = model(
                input_features,
                frame_len,
                frame_hand_pos=frame_hand_pos,
            )

        pred_len = int(olens[0].item())

        frame_tab_pred = one_hot_argmax_tab(torch.squeeze(frame_tab_score, 0))[:pred_len]
        frame_onset_score_np = torch.sigmoid(torch.squeeze(frame_onset_logits, 0)).detach().cpu().numpy()[:pred_len]
        frame_onset_pred = (frame_onset_score_np >= float(onset_threshold)).astype(np.float32)

        frame_tab_gt = frame_tab_gt[:pred_len]
        frame_onset_gt = frame_onset_gt[:pred_len]

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
        # Tolerant onset-only metrics: same string, +/- onset_tolerance_ms.
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
        # Tolerant decoded note-event metrics: same string/fret, +/- event_tolerance_ms.
        # ------------------------------------------------------------------
        pred_events = decode_events_from_tab_and_onset(frame_tab_pred, frame_onset_pred)
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
            frame_tab_pred=frame_tab_pred,
            frame_tab_gt=frame_tab_gt,
            frame_onset_pred_score=frame_onset_score_np,
            frame_onset_pred=frame_onset_pred,
            frame_onset_gt=frame_onset_gt,
            pred_events=np.asarray(pred_events, dtype=np.int64) if pred_events else np.zeros((0, 3), dtype=np.int64),
            gt_events=np.asarray(gt_events, dtype=np.int64) if gt_events else np.zeros((0, 3), dtype=np.int64),
            frame_seconds=np.asarray([frame_seconds], dtype=np.float32),
            onset_tolerance_ms=np.asarray([float(onset_tolerance_ms)], dtype=np.float32),
            event_tolerance_ms=np.asarray([float(event_tolerance_ms)], dtype=np.float32),
            onset_threshold=np.asarray([float(onset_threshold)], dtype=np.float32),
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

    event_avg_p = event_sum_p / n_files
    event_avg_r = event_sum_r / n_files
    event_avg_f = event_sum_f / n_files
    event_micro_p, event_micro_r, event_micro_f = prf_from_counts(event_sum_tp, event_sum_fp, event_sum_fn)

    if verbose:
        print(f"frame_avg_tab_p/r/f       = {frame_avg_p:.4f}, {frame_avg_r:.4f}, {frame_avg_f:.4f}")
        print(f"exact_onset_avg_p/r/f     = {exact_onset_avg_p:.4f}, {exact_onset_avg_r:.4f}, {exact_onset_avg_f:.4f}")
        print(f"tolerant_onset_avg_p/r/f  = {onset_avg_p:.4f}, {onset_avg_r:.4f}, {onset_avg_f:.4f}")
        print(f"tolerant_onset_micro_p/r/f= {onset_micro_p:.4f}, {onset_micro_r:.4f}, {onset_micro_f:.4f}")
        print(f"event_avg_p/r/f           = {event_avg_p:.4f}, {event_avg_r:.4f}, {event_avg_f:.4f}")
        print(f"event_micro_p/r/f         = {event_micro_p:.4f}, {event_micro_r:.4f}, {event_micro_f:.4f}")
        print(f"event TP/FP/FN            = {event_sum_tp}, {event_sum_fp}, {event_sum_fn}")

    result = pd.DataFrame(
        [[
            float(frame_ms),
            float(onset_tolerance_ms),
            float(event_tolerance_ms),
            float(onset_threshold),

            frame_avg_p,
            frame_avg_r,
            frame_avg_f,
            frame_concat_p,
            frame_concat_r,
            frame_concat_f,

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
            "onset_threshold",

            "frame_avg_tab_p",
            "frame_avg_tab_r",
            "frame_avg_tab_f",
            "frame_concat_tab_p",
            "frame_concat_tab_r",
            "frame_concat_tab_f",

            "frame_avg_onset_p",
            "frame_avg_onset_r",
            "frame_avg_onset_f",
            "frame_concat_onset_p",
            "frame_concat_onset_r",
            "frame_concat_onset_f",
            "onset_tp",
            "onset_fp",
            "onset_fn",

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
            "millisecond-based onset and note-event tolerances."
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
        "--onset-tolerance-ms",
        type=float,
        default=50.0,
        help="Tolerant onset matching window in milliseconds. Default: +/-25 ms.",
    )

    parser.add_argument(
        "--event-tolerance-ms",
        type=float,
        default=50.0,
        help="Tolerant note-event matching window in milliseconds. Default: +/-50 ms.",
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


if __name__ == "__main__":
    main()
