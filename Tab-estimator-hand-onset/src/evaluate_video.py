#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_video.py

Offline, app-compatible evaluator for VideoTestSet folders.

This script deliberately reuses the same backend utility path as the app:
  - tabest_utils.load_tabestimator_hand_onset_model
  - tabest_utils.create_audio_only_tabestimator_npz
  - tabest_utils.add_actual_hand_positions_to_npz
  - tabest_utils.split_npz_for_tabestimator_inference
  - tabest_utils.run_tabestimator_hand_onset_npz_inference
  - tabest_utils.events_to_vextab_text
  - gp5_utils.parse_gp_file_to_vextab

It also ports the frontend VexTab evaluation logic from DeepVideoPanel.jsx, then
adds a second pitch-based metric where notes are compared by sounding MIDI pitch
instead of string/fret identity.

Expected layout:

VideoTestSet/
  Sample-Name/
    extracted_audio.wav
    offline_fretboard.json          # not used for inference; kept for completeness
    offline_hand_bboxes.json        # must contain handbboxes with hand_soft_vector/finger_frets
    Sample-Name.gp5

Example:

python src/evaluate_video.py \
  --video-test-dir VideoTestSet \
  --sample The-Last-Of-Us-Jessica-Mazin-Never-Let-Me-Down-Again \
  --test-num 6

For your single-repo layout, keep the script at src/evaluate_video.py and run it from the repository root. It will use src/tabest_utils.py, src/gp5_utils.py, src/network.py, and model/.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_ONSET_MODEL_RUN = (
    "guitarset_guitartechs_egdb_goat_idmt_reverb_aug_handpos_clean/"
    "guitarset_guitartechs_egdb_goat_idmt_reverb_aug_handpos_clean"
)
DEFAULT_ONSET_MODEL_EPOCH = 192
DEFAULT_TEST_NUM = 6

REST_CLASS = 20

# VexTab / Guitar Pro string numbering: 1 = high e, 6 = low E.
STANDARD_TUNING_BASE_MIDI = {
    1: 64,  # high e
    2: 59,  # B
    3: 55,  # G
    4: 50,  # D
    5: 45,  # A
    6: 40,  # low E
}


# =============================================================================
# Import app modules
# =============================================================================


def _candidate_backend_dirs(backend_dir: Optional[str]) -> List[Path]:
    here = Path(__file__).resolve()
    cwd = Path.cwd().resolve()
    candidates: List[Path] = []

    if backend_dir:
        candidates.append(Path(backend_dir).expanduser().resolve())

    candidates.extend(
        [
            cwd,
            cwd / "backend_cv",
            cwd / "backend",
            cwd.parent,
            cwd.parent / "backend_cv",
            cwd.parent / "backend",
            here.parent,
            here.parent.parent,
            here.parent.parent / "backend_cv",
            here.parent.parent / "backend",
        ]
    )

    out: List[Path] = []
    seen = set()
    for p in candidates:
        try:
            rp = p.resolve()
        except Exception:
            continue
        if str(rp) not in seen:
            seen.add(str(rp))
            out.append(rp)
    return out


def import_app_modules(backend_dir: Optional[str] = None):
    searched = []
    for d in _candidate_backend_dirs(backend_dir):
        searched.append(str(d))
        if (d / "tabest_utils.py").is_file() and (d / "gp5_utils.py").is_file():
            # Support the single-repository layout:
            #   Tab-estimator-hand-onset/
            #     src/tabest_utils.py
            #     src/gp5_utils.py
            #     src/network.py
            #     model/...
            #
            # The uploaded tabest_utils.py was originally written for a backend
            # layout with sibling light repos, and it reads TABEST_ROOT /
            # TABEST_ONSET_ROOT at import time.  Set safe defaults before import
            # so tabest_utils finds this repository's src/network.py.
            if d.name == "src" and (d / "network.py").is_file():
                project_root = d.parent.resolve()
                os.environ.setdefault("TABEST_ROOT", str(project_root))
                os.environ.setdefault("TABEST_ONSET_ROOT", str(project_root))
                os.environ.setdefault("TABEST_HIDONSET_ROOT", str(project_root))

            if str(d) not in sys.path:
                sys.path.insert(0, str(d))

            tabest_utils = importlib.import_module("tabest_utils")
            gp5_utils = importlib.import_module("gp5_utils")
            return tabest_utils, gp5_utils, d

    raise ImportError(
        "Could not find both tabest_utils.py and gp5_utils.py.\n"
        "Pass --backend-dir pointing to the folder that contains these scripts.\n"
        "For your repo layout, this is usually: --backend-dir src\n"
        "Searched:\n  " + "\n  ".join(searched)
    )


# =============================================================================
# Sample discovery
# =============================================================================


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_samples(video_test_dir: Path, sample_name: Optional[str] = None) -> List[Dict[str, Path]]:
    if not video_test_dir.is_dir():
        raise FileNotFoundError(f"VideoTestSet directory not found: {video_test_dir}")

    dirs = [p for p in sorted(video_test_dir.iterdir()) if p.is_dir()]

    if sample_name:
        exact = [p for p in dirs if p.name == sample_name]
        if exact:
            dirs = exact
        else:
            partial = [p for p in dirs if sample_name.lower() in p.name.lower()]
            if not partial:
                raise FileNotFoundError(f"No sample matching {sample_name!r} under {video_test_dir}")
            dirs = partial

    samples: List[Dict[str, Path]] = []
    for d in dirs:
        audio = d / "extracted_audio.wav"
        fretboard = d / "offline_fretboard.json"
        hand = d / "offline_hand_bboxes.json"
        preferred_gp5 = d / f"{d.name}.gp5"
        gp5s = sorted(d.glob("*.gp5"))
        gp5 = preferred_gp5 if preferred_gp5.is_file() else (gp5s[0] if gp5s else None)

        missing = []
        if not audio.is_file():
            missing.append("extracted_audio.wav")
        if not fretboard.is_file():
            missing.append("offline_fretboard.json")
        if not hand.is_file():
            missing.append("offline_hand_bboxes.json")
        if gp5 is None or not gp5.is_file():
            missing.append("*.gp5")

        if missing:
            print(f"[skip] {d.name}: missing {', '.join(missing)}")
            continue

        samples.append(
            {
                "name": d.name,
                "root": d,
                "audio": audio,
                "fretboard": fretboard,
                "hand": hand,
                "gp5": gp5,
            }
        )

    if not samples:
        raise FileNotFoundError(f"No complete samples found under {video_test_dir}")

    return samples


# =============================================================================
# Frontend-equivalent VexTab parsing and scoring
# =============================================================================


def unique_notes(notes: Sequence[Dict[str, Any]]) -> List[Dict[str, int]]:
    seen = set()
    out: List[Dict[str, int]] = []
    for note in notes or []:
        try:
            fret = int(round(float(note.get("fret"))))
            string = int(round(float(note.get("string"))))
        except Exception:
            continue
        if fret < 0 or string < 1 or string > 6:
            continue
        key = (fret, string)
        if key in seen:
            continue
        seen.add(key)
        out.append({"fret": fret, "string": string})
    return sorted(out, key=lambda n: (n["string"], n["fret"]))


def parse_vextab_steps(tab_text: str) -> List[Dict[str, Any]]:
    if not tab_text:
        return []

    note_sections: List[str] = []
    for line in str(tab_text).splitlines():
        trimmed = line.strip()
        if not trimmed or trimmed.startswith("//"):
            continue
        idx = trimmed.find("notes")
        if idx >= 0:
            note_sections.append(trimmed[idx + 5 :].strip())

    body = " ".join(note_sections)
    steps: List[Dict[str, Any]] = []

    i = 0
    while i < len(body):
        ch = body[i]

        if ch.isspace() or ch == "|":
            i += 1
            continue

        if ch == "(":
            end = body.find(")", i + 1)
            if end < 0:
                break
            chord_text = body[i + 1 : end]
            notes = []
            for fret, string in __import__("re").findall(r"(\d+)\/(\d+)", chord_text):
                notes.append({"fret": int(fret), "string": int(string)})
            notes = unique_notes(notes)
            if notes:
                steps.append({"type": "chord", "notes": notes})
            i = end + 1
            continue

        m = __import__("re").match(r"^(\d+)\/(\d+)", body[i:])
        if m:
            steps.append(
                {
                    "type": "note",
                    "notes": unique_notes([{"fret": int(m.group(1)), "string": int(m.group(2))}]),
                }
            )
            i += len(m.group(0))
            continue

        # Skip duration/rest/control tokens like :q, :8, =:|, etc.
        m = __import__("re").match(r"^\S+", body[i:])
        i += len(m.group(0)) if m else 1

    return steps


def note_key(note: Dict[str, Any], mode: str) -> str:
    fret = int(note["fret"])
    string = int(note["string"])
    if mode == "string_fret":
        return f"{fret}/{string}"
    if mode == "pitch":
        return str(STANDARD_TUNING_BASE_MIDI[string] + fret)
    raise ValueError(f"Unknown metric mode: {mode}")


def step_keys(step: Dict[str, Any], mode: str) -> List[str]:
    return sorted(set(note_key(n, mode) for n in unique_notes(step.get("notes", []))))


def same_step(a: Dict[str, Any], b: Dict[str, Any], mode: str) -> bool:
    return step_keys(a, mode) == step_keys(b, mode)


def intersection_size(a: Dict[str, Any], b: Dict[str, Any], mode: str) -> int:
    aa = set(step_keys(a, mode))
    bb = set(step_keys(b, mode))
    return len(aa & bb)


def evaluate_step_sequences(reference_steps: List[Dict[str, Any]], predicted_steps: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    # Direct Python port of DeepVideoPanel.jsx/evaluateTabStepSequences, parameterized by note key.
    n = len(reference_steps)
    m = len(predicted_steps)

    dp = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
    back = [[None for _ in range(m + 1)] for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
        if i > 0:
            back[i][0] = "up"

    for j in range(m + 1):
        dp[0][j] = j
        if j > 0:
            back[0][j] = "left"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost_diag = dp[i - 1][j - 1] + (0 if same_step(reference_steps[i - 1], predicted_steps[j - 1], mode) else 1)
            cost_up = dp[i - 1][j] + 1
            cost_left = dp[i][j - 1] + 1
            best = min(cost_diag, cost_up, cost_left)
            dp[i][j] = best
            if best == cost_diag:
                back[i][j] = "diag"
            elif best == cost_up:
                back[i][j] = "up"
            else:
                back[i][j] = "left"

    i, j = n, m
    exact_matches = 0
    chord_exact_matches = 0
    substitutions = 0
    missed = 0
    extra = 0

    while i > 0 or j > 0:
        move = back[i][j]

        if move == "diag":
            ref = reference_steps[i - 1]
            pred = predicted_steps[j - 1]
            ref_keys = step_keys(ref, mode)
            pred_keys = step_keys(pred, mode)
            common = intersection_size(ref, pred, mode)

            exact_matches += common
            if ref_keys == pred_keys:
                chord_exact_matches += 1

            unmatched_ref = max(0, len(ref_keys) - common)
            unmatched_pred = max(0, len(pred_keys) - common)
            substitutions += min(unmatched_ref, unmatched_pred)
            missed += max(0, unmatched_ref - unmatched_pred)
            extra += max(0, unmatched_pred - unmatched_ref)

            i -= 1
            j -= 1
        elif move == "up":
            missed += len(step_keys(reference_steps[i - 1], mode))
            i -= 1
        elif move == "left":
            extra += len(step_keys(predicted_steps[j - 1], mode))
            j -= 1
        else:
            break

    ref_count = sum(len(step_keys(step, mode)) for step in reference_steps)
    pred_count = sum(len(step_keys(step, mode)) for step in predicted_steps)
    precision = exact_matches / pred_count if pred_count > 0 else 0.0
    recall = exact_matches / ref_count if ref_count > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0

    return {
        f"{mode}_ref_count": int(ref_count),
        f"{mode}_pred_count": int(pred_count),
        f"{mode}_ref_steps": int(n),
        f"{mode}_pred_steps": int(m),
        f"{mode}_exact_matches": int(exact_matches),
        f"{mode}_chord_exact_matches": int(chord_exact_matches),
        f"{mode}_substitutions": int(substitutions),
        f"{mode}_missed": int(missed),
        f"{mode}_extra": int(extra),
        f"{mode}_accuracy_over_reference": float(recall),
        f"{mode}_precision": float(precision),
        f"{mode}_recall": float(recall),
        f"{mode}_f1": float(f1),
        f"{mode}_edit_distance": int(dp[n][m]),
    }


def vextab_notes_line(tab_text: str) -> str:
    lines = []
    for line in str(tab_text or "").splitlines():
        t = line.strip()
        if t.startswith("notes"):
            lines.append(t)
    return "\n".join(lines)


# =============================================================================
# App-compatible inference
# =============================================================================


def load_reference_vextab(gp5_utils, gp5_path: Path, args) -> Tuple[str, Dict[str, Any]]:
    kwargs = {
        "path": str(gp5_path),
        "track_index": args.gp5_track_index,
        "track_name_contains": args.gp5_track_name_contains,
        "all_guitar_tracks": bool(args.gp5_all_guitar_tracks),
        "require_standard_tuning": bool(args.gp5_require_standard_tuning),
        "max_fret": int(args.gp5_max_fret),
        "include_tied": bool(args.gp5_include_tied),
    }
    result = gp5_utils.parse_gp_file_to_vextab(**kwargs)
    vextab = result.get("vextab") or result.get("vextab_text") or result.get("tab_text")
    if not vextab:
        raise RuntimeError(f"gp5_utils.parse_gp_file_to_vextab did not return VexTab text for {gp5_path}")
    return str(vextab), result


def estimate_or_choose_bpm(tabest_utils, audio_path: Path, config: Dict[str, Any], bpm_override: Optional[float]) -> float:
    if bpm_override is not None:
        bpm = float(bpm_override)
    else:
        import librosa

        sr = int(config["down_sampling_rate"])
        y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
        bpm = float(tabest_utils.estimate_bpm_for_tabestimator(y, sr))

    if not np.isfinite(bpm) or bpm <= 0:
        bpm = 120.0
    return float(bpm)


def hand_bboxes_from_json(hand_json_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    hand_data = read_json(hand_json_path)
    if isinstance(hand_data, dict):
        hand_bboxes = hand_data.get("handbboxes") or hand_data.get("bboxes") or hand_data.get("hand_bboxes") or []
    elif isinstance(hand_data, list):
        hand_bboxes = hand_data
        hand_data = {"handbboxes": hand_bboxes}
    else:
        hand_bboxes = []
        hand_data = {}
    if not isinstance(hand_bboxes, list):
        hand_bboxes = []
    return hand_bboxes, hand_data


def warn_hand_metadata(sample_name: str, hand_data: Dict[str, Any], args) -> List[str]:
    warnings = []

    cached_n_frets = hand_data.get("n_frets", hand_data.get("max_fret")) if isinstance(hand_data, dict) else None
    if cached_n_frets is not None:
        try:
            if int(cached_n_frets) != int(args.n_frets):
                warnings.append(
                    f"cached hand n_frets={cached_n_frets}, CLI n_frets={args.n_frets}; "
                    "evaluate_video does not recompute CV, so cached values are used."
                )
        except Exception:
            pass

    cached_fps = hand_data.get("hand_tracking_fps", hand_data.get("sample_fps")) if isinstance(hand_data, dict) else None
    if cached_fps is not None:
        try:
            if abs(float(cached_fps) - float(args.hand_tracking_fps)) > 1e-6:
                warnings.append(
                    f"cached hand_tracking_fps={cached_fps}, CLI hand_tracking_fps={args.hand_tracking_fps}; "
                    "evaluate_video does not recompute CV, so cached values are used."
                )
        except Exception:
            pass

    for w in warnings:
        print(f"[warn] {sample_name}: {w}")
    return warnings


def run_app_like_onset_inference(tabest_utils, sample: Dict[str, Path], bundle: Dict[str, Any], args, sample_out_dir: Path) -> Dict[str, Any]:
    model = bundle["model"]
    config = bundle["config"]
    device = bundle["device"]
    hand_pos_dim = int(bundle["hand_pos_dim"])
    hand_span = int(bundle.get("hand_span", 4))

    sr = int(config["down_sampling_rate"])
    hop_length = int(config["hop_length"])
    frame_dur = float(hop_length) / float(sr)
    note_resolution = int(config.get("note_resolution", 16))

    used_bpm = estimate_or_choose_bpm(tabest_utils, sample["audio"], config, args.bpm)

    hand_bboxes, hand_data = hand_bboxes_from_json(sample["hand"])
    if not hand_bboxes:
        raise RuntimeError(f"No hand bboxes found in {sample['hand']}")
    hand_warnings = warn_hand_metadata(str(sample["name"]), hand_data, args)

    runtime_npz_dir = sample_out_dir / "runtime_npz"
    runtime_split_dir = runtime_npz_dir / "split"
    if args.clean_runtime_npz and runtime_npz_dir.exists():
        shutil.rmtree(runtime_npz_dir)
    runtime_npz_dir.mkdir(parents=True, exist_ok=True)
    runtime_split_dir.mkdir(parents=True, exist_ok=True)

    audio_only_npz = tabest_utils.create_audio_only_tabestimator_npz(
        audio_filename=str(sample["audio"]),
        npz_path_no_ext=str(runtime_npz_dir / "audio_only"),
        config=config,
        tempo=used_bpm,
    )

    with_hand_npz = tabest_utils.add_actual_hand_positions_to_npz(
        input_npz_path=audio_only_npz,
        output_npz_path=str(runtime_npz_dir / "with_actual_handpos.npz"),
        hand_bboxes=hand_bboxes,
        config=config,
        tempo=used_bpm,
        n_positions=hand_pos_dim,
        hand_span=hand_span,
        max_dt=float(args.hand_max_dt),
        sigma=float(args.hand_pos_sigma),
        overwrite=True,
        debug=bool(args.debug_hand_positions),
    )

    split_npz_paths = tabest_utils.split_npz_for_tabestimator_inference(
        input_npz_path=with_hand_npz,
        output_split_dir=str(runtime_split_dir),
        note_resolution=note_resolution,
        overwrite=True,
    )
    if not split_npz_paths:
        raise RuntimeError("No runtime split NPZ files were created.")

    all_events: List[Dict[str, Any]] = []
    all_frame_events: List[Dict[str, Any]] = []
    chunks: List[Dict[str, Any]] = []
    frame_ascii_blocks: List[str] = []
    global_frame_start = 0

    for split_idx, split_npz_path in enumerate(split_npz_paths):
        split_start_time = float(global_frame_start) * frame_dur

        result = tabest_utils.run_tabestimator_hand_onset_npz_inference(
            model=model,
            npz_path=split_npz_path,
            config=config,
            device=device,
            hand_pos_dim=hand_pos_dim,
            feature_key=None,
            normalize_hand_rows=False,
            start_time=split_start_time,
            start_frame=global_frame_start,
            frame_step_width=int(args.frame_step_width),
            onset_threshold=float(args.onset_threshold),
            global_onset_threshold=float(args.global_onset_threshold),
            peak_pre_max_ms=float(args.peak_pre_max_ms),
            peak_post_max_ms=float(args.peak_post_max_ms),
            peak_combine_ms=float(args.peak_combine_ms),
            event_label_delay_ms=float(args.event_label_delay_ms),
            event_label_window_ms=float(args.event_label_window_ms),
            event_string_window_ms=float(args.event_string_window_ms),
            event_tab_threshold=float(args.event_tab_threshold),
            event_string_threshold=float(args.event_string_threshold),
            no_event_string_filter=bool(args.no_event_string_filter),
            event_decode_mode=str(args.event_decode_mode),
            event_chord_group_ms=float(args.event_chord_group_ms),
            use_global_onset_confirmation=bool(args.use_global_onset_confirmation),
            global_confirm_window_ms=float(args.global_confirm_window_ms),
            use_global_onset_fallback=bool(args.use_global_onset_fallback),
            global_fallback_max_notes=int(args.global_fallback_max_notes),
            repeat_same_fret_policy=str(args.repeat_same_fret_policy),
            min_repeat_ms=float(args.min_repeat_ms),
            repeat_onset_threshold=float(args.repeat_onset_threshold),
            repeat_global_threshold=float(args.repeat_global_threshold),
            require_global_for_repeats=bool(args.require_global_for_repeats),
            same_string_any_fret_min_ms=float(args.same_string_any_fret_min_ms),
        )

        chunk_events = result.get("events", [])
        chunk_frame_events = result.get("frame_events", [])
        all_events.extend(chunk_events)
        all_frame_events.extend(chunk_frame_events)

        frame_ascii_block = (
            f"# Split {split_idx:03d} | onset-frame-level | "
            f"start={split_start_time:.2f}s | npz={Path(split_npz_path).name}\n"
            f"{result.get('frame_ascii_tab', '')}"
        )
        frame_ascii_blocks.append(frame_ascii_block)

        frame_pred_len = int(result.get("frame_pred_len", result.get("frame_count", 0)))
        chunks.append(
            {
                "chunk_index": int(split_idx),
                "start_time": float(split_start_time),
                "start_frame": int(global_frame_start),
                "npz_path": str(split_npz_path),
                "feature_shape": result.get("feature_shape"),
                "frame_hand_pos_shape": result.get("frame_hand_pos_shape"),
                "frame_hand_pos_index": result.get("frame_hand_pos_index"),
                "events": chunk_events,
                "note_events": chunk_events,
                "onset_events": chunk_events,
                "raw_onset_events": result.get("raw_onset_events", []),
                "ungrouped_onset_events": result.get("ungrouped_onset_events", []),
                "frame_events": chunk_frame_events,
                "frame_ascii_tab": result.get("frame_ascii_tab", ""),
                "frame_tab_classes_low_e_first": result.get("frame_pred_classes_low_e_first", []),
                "global_onset_peak_frames": result.get("global_onset_peak_frames", []),
                "per_string_onset_peak_frames": result.get("per_string_onset_peak_frames", []),
                "frame_pred_len": int(frame_pred_len),
                "frame_dur": float(frame_dur),
                "event_decoder": result.get("event_decoder", {}),
                "repeat_filter_diagnostics": result.get("repeat_filter_diagnostics"),
            }
        )
        global_frame_start += int(frame_pred_len)

    frame_ascii_tab = "\n\n".join(frame_ascii_blocks)
    vextab = tabest_utils.events_to_vextab_text(all_events, width=int(args.vextab_width))

    return {
        "sample": str(sample["name"]),
        "audio_path": str(sample["audio"]),
        "gp5_path": str(sample["gp5"]),
        "bpm_for_npz_only": float(used_bpm),
        "sample_rate": int(sr),
        "sr": int(sr),
        "down_sampling_rate": int(sr),
        "hop_length": int(hop_length),
        "frame_dur": float(frame_dur),
        "frame_ms": float(frame_dur * 1000.0),
        "hand_pos_dim": int(hand_pos_dim),
        "hand_span": int(hand_span),
        "hand_bbox_count": int(len(hand_bboxes)),
        "hand_max_dt": float(args.hand_max_dt),
        "n_frets": int(args.n_frets),
        "hand_tracking_fps": float(args.hand_tracking_fps),
        "hand_warnings": hand_warnings,
        "runtime_npz_dir": str(runtime_npz_dir),
        "audio_only_npz": str(audio_only_npz),
        "with_actual_handpos_npz": str(with_hand_npz),
        "split_npz_dir": str(runtime_split_dir),
        "split_npz_paths": [str(p) for p in split_npz_paths],
        "decoder": decoder_dict_from_args(args),
        "events": all_events,
        "note_events": all_events,
        "onset_events": all_events,
        "vextab": vextab,
        "note_vextab": vextab,
        "ascii_tab": frame_ascii_tab,
        "frame_ascii_tab": frame_ascii_tab,
        "frame_events": all_frame_events,
        "chunks": chunks,
        "count_chunks": int(len(chunks)),
        "count_events": int(len(all_events)),
        "count_note_events": int(len(all_events)),
        "count_onset_events": int(len(all_events)),
        "count_frame_events": int(len(all_frame_events)),
    }


def decoder_dict_from_args(args) -> Dict[str, Any]:
    return {
        "mode": str(args.event_decode_mode),
        "primary_timing": "per_string_onset" if str(args.event_decode_mode).lower() in {"string_onset", "hybrid"} else "global_onset",
        "onset_threshold": float(args.onset_threshold),
        "global_onset_threshold": float(args.global_onset_threshold),
        "peak_pre_max_ms": float(args.peak_pre_max_ms),
        "peak_post_max_ms": float(args.peak_post_max_ms),
        "peak_combine_ms": float(args.peak_combine_ms),
        "event_label_delay_ms": float(args.event_label_delay_ms),
        "event_label_window_ms": float(args.event_label_window_ms),
        "event_string_window_ms": float(args.event_string_window_ms),
        "event_tab_threshold": float(args.event_tab_threshold),
        "event_string_threshold": float(args.event_string_threshold),
        "no_event_string_filter": bool(args.no_event_string_filter),
        "use_string_onset_filter": not bool(args.no_event_string_filter),
        "event_chord_group_ms": float(args.event_chord_group_ms),
        "use_global_onset_confirmation": bool(args.use_global_onset_confirmation),
        "global_confirm_window_ms": float(args.global_confirm_window_ms),
        "use_global_onset_fallback": bool(args.use_global_onset_fallback),
        "global_fallback_max_notes": int(args.global_fallback_max_notes),
        "repeat_same_fret_policy": str(args.repeat_same_fret_policy),
        "min_repeat_ms": float(args.min_repeat_ms),
        "repeat_onset_threshold": float(args.repeat_onset_threshold),
        "repeat_global_threshold": float(args.repeat_global_threshold),
        "require_global_for_repeats": bool(args.require_global_for_repeats),
        "same_string_any_fret_min_ms": float(args.same_string_any_fret_min_ms),
    }


# =============================================================================
# Evaluation per sample / summaries
# =============================================================================


def evaluate_sample(tabest_utils, gp5_utils, sample: Dict[str, Path], bundle: Dict[str, Any], args) -> Dict[str, Any]:
    sample_name = str(sample["name"])
    out_dir = Path(args.output_dir) / sample_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {sample_name} ===")

    prediction = run_app_like_onset_inference(tabest_utils, sample, bundle, args, out_dir)
    reference_vextab, reference_meta = load_reference_vextab(gp5_utils, sample["gp5"], args)
    predicted_vextab = prediction["vextab"]

    ref_steps = parse_vextab_steps(reference_vextab)
    pred_steps = parse_vextab_steps(predicted_vextab)

    sf_metrics = evaluate_step_sequences(ref_steps, pred_steps, mode="string_fret")
    pitch_metrics = evaluate_step_sequences(ref_steps, pred_steps, mode="pitch")

    # Save detailed artifacts.
    (out_dir / "reference.vextab").write_text(reference_vextab, encoding="utf-8")
    (out_dir / "prediction.vextab").write_text(predicted_vextab, encoding="utf-8")
    (out_dir / "reference_notes.txt").write_text(vextab_notes_line(reference_vextab) + "\n", encoding="utf-8")
    (out_dir / "prediction_notes.txt").write_text(vextab_notes_line(predicted_vextab) + "\n", encoding="utf-8")
    (out_dir / "events.json").write_text(json.dumps(prediction.get("events", []), indent=2), encoding="utf-8")
    (out_dir / "chunks.json").write_text(json.dumps(prediction.get("chunks", []), indent=2), encoding="utf-8")
    (out_dir / "reference_meta.json").write_text(json.dumps(reference_meta, indent=2, default=str), encoding="utf-8")
    (out_dir / "prediction_full.json").write_text(json.dumps(prediction, indent=2, default=str), encoding="utf-8")

    result = {
        "sample": sample_name,
        "audio_path": str(sample["audio"]),
        "gp5_path": str(sample["gp5"]),
        "bpm_for_npz_only": float(prediction.get("bpm_for_npz_only", 0.0)),
        "count_chunks": int(prediction.get("count_chunks", 0)),
        "count_events": int(prediction.get("count_events", 0)),
        "count_frame_events": int(prediction.get("count_frame_events", 0)),
        "reference_steps_parsed": int(len(ref_steps)),
        "predicted_steps_parsed": int(len(pred_steps)),
        **sf_metrics,
        **pitch_metrics,
        "model_run": str(args.model_run),
        "epoch": int(args.epoch),
        "test_num": int(args.test_num),
        **{f"decoder_{k}": v for k, v in decoder_dict_from_args(args).items()},
    }

    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print(
        "string-fret: "
        f"P={result['string_fret_precision']:.3f} "
        f"R={result['string_fret_recall']:.3f} "
        f"F1={result['string_fret_f1']:.3f} "
        f"({result['string_fret_exact_matches']}/{result['string_fret_pred_count']}/{result['string_fret_ref_count']})"
    )
    print(
        "pitch:       "
        f"P={result['pitch_precision']:.3f} "
        f"R={result['pitch_recall']:.3f} "
        f"F1={result['pitch_f1']:.3f} "
        f"({result['pitch_exact_matches']}/{result['pitch_pred_count']}/{result['pitch_ref_count']})"
    )
    print(f"wrote: {out_dir}")

    return result


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    out: Dict[str, Any] = {"sample": "MEAN"}
    numeric_keys = []
    for k in sorted({kk for r in rows for kk in r.keys()}):
        vals = [r.get(k) for r in rows]
        if vals and all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals if v is not None):
            numeric_keys.append(k)
    for k in numeric_keys:
        vals = [float(r[k]) for r in rows if r.get(k) is not None]
        if vals:
            out[k] = float(np.mean(vals))
    return out


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    p = argparse.ArgumentParser(description="App-compatible VideoTestSet evaluator for TabEstimator hand+onset.")

    p.add_argument("--video-test-dir", default=str(Path(__file__).resolve().parents[1] / "VideoTestSet"))
    p.add_argument("--sample", default=None, help="Evaluate only one sample folder by exact or partial name.")
    p.add_argument("--output-dir", default="video_eval_results")
    p.add_argument("--backend-dir", default=str(Path(__file__).resolve().parent), help="Folder containing tabest_utils.py and gp5_utils.py. In this repo layout, default is src/.")

    p.add_argument("--model-run", default=DEFAULT_ONSET_MODEL_RUN)
    p.add_argument("--epoch", type=int, default=DEFAULT_ONSET_MODEL_EPOCH)
    p.add_argument("--test-num", type=int, default=DEFAULT_TEST_NUM)
    p.add_argument("--model-root", default=str(Path(__file__).resolve().parents[1] / "model"), help="Model root passed to tabest_utils loader. In this repo layout, default is ../model relative to src/evaluate_video.py.")
    p.add_argument("--checkpoint-path", default=None)
    p.add_argument("--config-path", default=None)
    p.add_argument("--device", default=None)

    p.add_argument("--bpm", type=float, default=None, help="Optional BPM used only to create/split the runtime NPZ. If omitted, estimated from audio.")
    p.add_argument("--hand-tracking-fps", type=float, default=8.0, help="Metadata/check only; this evaluator uses cached offline_hand_bboxes.json.")
    p.add_argument("--n-frets", type=int, default=24, help="Metadata/check only; this evaluator uses cached offline_hand_bboxes.json.")
    p.add_argument("--hand-max-dt", type=float, default=0.15)
    p.add_argument("--hand-pos-sigma", type=float, default=1.0)
    p.add_argument("--frame-step-width", type=int, default=2)
    p.add_argument("--debug-hand-positions", action="store_true")
    p.add_argument("--clean-runtime-npz", action="store_true", default=True)
    p.add_argument("--keep-runtime-npz", dest="clean_runtime_npz", action="store_false")

    # App/UI defaults from DeepVideoPanel.jsx state.
    p.add_argument("--onset-threshold", type=float, default=0.7)
    p.add_argument("--global-onset-threshold", type=float, default=0.7)
    p.add_argument("--peak-pre-max-ms", type=float, default=90.0)
    p.add_argument("--peak-post-max-ms", type=float, default=90.0)
    p.add_argument("--peak-combine-ms", type=float, default=50.0)

    p.add_argument("--event-label-delay-ms", type=float, default=50.0)
    p.add_argument("--event-label-window-ms", type=float, default=50.0)
    p.add_argument("--event-string-window-ms", type=float, default=50.0)
    p.add_argument("--event-tab-threshold", type=float, default=0.5)
    p.add_argument("--event-string-threshold", type=float, default=0.3)
    p.add_argument("--no-event-string-filter", action="store_true", default=False)

    p.add_argument("--event-decode-mode", choices=["string_onset", "global_onset", "hybrid"], default="string_onset")
    p.add_argument("--event-chord-group-ms", type=float, default=50.0)
    p.add_argument("--use-global-onset-confirmation", action="store_true", default=False)
    p.add_argument("--global-confirm-window-ms", type=float, default=40.0)
    p.add_argument("--use-global-onset-fallback", action="store_true", default=True)
    p.add_argument("--no-global-onset-fallback", dest="use_global_onset_fallback", action="store_false")
    p.add_argument("--global-fallback-max-notes", type=int, default=1)

    p.add_argument(
        "--repeat-same-fret-policy",
        choices=["off", "refractory", "strong_onset", "tab_change_or_strong_onset"],
        default="tab_change_or_strong_onset",
    )
    p.add_argument("--min-repeat-ms", type=float, default=90.0)
    p.add_argument("--repeat-onset-threshold", type=float, default=0.90)
    p.add_argument("--repeat-global-threshold", type=float, default=0.65)
    p.add_argument("--require-global-for-repeats", action="store_true", default=True)
    p.add_argument("--no-require-global-for-repeats", dest="require_global_for_repeats", action="store_false")
    p.add_argument("--same-string-any-fret-min-ms", type=float, default=35.0)

    # GP5 conversion defaults mirror FastAPI /tab/gp5-to-vextab.
    p.add_argument("--gp5-track-index", type=int, default=None)
    p.add_argument("--gp5-track-name-contains", default=None)
    p.add_argument("--gp5-all-guitar-tracks", action="store_true", default=False)
    p.add_argument("--gp5-require-standard-tuning", action="store_true", default=True)
    p.add_argument("--gp5-no-require-standard-tuning", dest="gp5_require_standard_tuning", action="store_false")
    p.add_argument("--gp5-max-fret", type=int, default=19)
    p.add_argument("--gp5-include-tied", action="store_true", default=False)
    p.add_argument("--vextab-width", type=int, default=1240)

    return p.parse_args()


def main() -> int:
    args = parse_args()

    tabest_utils, gp5_utils, backend_dir = import_app_modules(args.backend_dir)
    print("backend_dir:", backend_dir)

    model_root = args.model_root if args.model_root not in {"", "None", "none", None} else None
    bundle = tabest_utils.load_tabestimator_hand_onset_model(
        model_run=args.model_run,
        epoch=args.epoch,
        test_num=args.test_num,
        model_root=model_root,
        checkpoint_path=args.checkpoint_path,
        config_path=args.config_path,
        device=args.device,
    )

    print("checkpoint:", bundle.get("checkpoint_path"))
    print("config:", bundle.get("config_path"))
    print("device:", bundle.get("device"))

    samples = discover_samples(Path(args.video_test_dir), sample_name=args.sample)
    print("samples:", [s["name"] for s in samples])

    rows: List[Dict[str, Any]] = []
    for sample in samples:
        try:
            rows.append(evaluate_sample(tabest_utils, gp5_utils, sample, bundle, args))
        except Exception as exc:
            print(f"[error] {sample['name']}: {exc}")
            traceback.print_exc()
            if args.sample:
                raise
            rows.append({"sample": sample["name"], "error": str(exc)})

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    mean_row = aggregate_rows([r for r in rows if "error" not in r])
    all_rows = rows + ([mean_row] if mean_row else [])

    write_csv(out_root / "summary.csv", all_rows)
    (out_root / "summary.json").write_text(json.dumps(all_rows, indent=2, default=str), encoding="utf-8")

    if mean_row:
        print("\n=== MEAN ===")
        print(
            "string-fret: "
            f"P={mean_row.get('string_fret_precision', 0):.3f} "
            f"R={mean_row.get('string_fret_recall', 0):.3f} "
            f"F1={mean_row.get('string_fret_f1', 0):.3f}"
        )
        print(
            "pitch:       "
            f"P={mean_row.get('pitch_precision', 0):.3f} "
            f"R={mean_row.get('pitch_recall', 0):.3f} "
            f"F1={mean_row.get('pitch_f1', 0):.3f}"
        )
        print("summary:", out_root / "summary.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
