#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a Tab-Estimator-style dataset from 5% of DadaGP, using Firebrand
single-note audio samples and optional synthetic/phantom hand-position heatmaps.

Output .npz files go to:
    data/npz/firebrand_phantom/split/

Each .npz contains the usual Tab-Estimator keys:
    cqt, log_cqt, mel_spec,
    tab, tab_onset, frame_tab, frame_tab_onset,
    F0, F0_onset, frame_F0, frame_F0_onset,
    tempo, len_in_notes

Optional extra keys for a hand-conditioned modified Tab-Estimator model
when --hand-mode phantom is used:
    phantom_hand, frame_phantom_hand,
    phantom_hand_conf, frame_phantom_hand_conf

When --hand-mode none is used, the .npz files are audio-only/standard
Tab-Estimator-style files and do not contain phantom hand arrays.

Assumptions:
- PyGuitarPro string 1 is high e and string 6 is low E.
- Tab-Estimator string axis is low E -> high e.
- Firebrand folders use string1=high e ... string6=low E.
- Old Firebrand samples often exist only for frets 0..12, so the default
  --max-render-fret is 12. If you have 0..19 samples, set it to 19.

Install:
    pip install numpy librosa scipy pyguitarpro tqdm

Optional MP3 debug output also requires ffmpeg:
    sudo apt-get install ffmpeg
    # or: brew install ffmpeg

Example:
    python create_dadagp_firebrand_phantom_npz.py \
      --dadagp-root data/DadaGP-v1.1/DadaGP-v1.1 \
      --firebrand-audio-root data/guitar_samples \
      --firebrand-onset-root data/onsets \
      --out-dir data/npz/firebrand_phantom/split \
      --data-percentage 0.05 \
      --max-render-fret 12 \
      --hand-mode phantom \
      --write-mp3 \
      --mp3-bitrate 32k
"""
import argparse
import json
import math
import random
import re
import shutil
import subprocess
import tempfile
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
from tqdm import tqdm

try:
    import guitarpro as gp
except ImportError as exc:
    raise SystemExit("Install PyGuitarPro with: pip install pyguitarpro") from exc

# Tab-Estimator-like constants.
N_STRINGS = 6
NO_PLAY = 20                 # fret classes 0..19, plus 20 = not played
N_TAB_CLASSES = 21
N_F0_CLASSES = 44            # MIDI 40..83
LOW_E_MIDI = 40
OPEN_MIDI_LOW_TO_HIGH = np.array([40, 45, 50, 55, 59, 64], dtype=np.int32)
DEFAULT_TICKS_PER_QUARTER = 960


@dataclass
class NoteEvent:
    start_sec: float
    end_sec: float
    tab_string: int     # 0=low E, 5=high e
    gp_string: int      # 1=high e, 6=low E; used for Firebrand folders
    fret: int
    midi: int
    velocity: float
    source_file: str
    track_name: str


@dataclass
class FirebrandSample:
    audio: np.ndarray
    sr: int
    gp_string: int
    fret: int
    source_path: str


def safe_name(path: Path, max_len: int = 90) -> str:
    s = re.sub(r"[^a-zA-Z0-9_\-]+", "_", path.stem)
    return s[:max_len]


def gp_string_to_tab_index(gp_string: int) -> int:
    """GP: 1=high e..6=low E. Tab-Estimator: 0=low E..5=high e."""
    return 6 - int(gp_string)


def midi_from_string_fret(tab_string: int, fret: int) -> int:
    return int(OPEN_MIDI_LOW_TO_HIGH[tab_string] + fret)


def get_song_tempo(song) -> float:
    tempo = getattr(song, "tempo", 120)
    try:
        return float(getattr(tempo, "value"))
    except Exception:
        try:
            return float(tempo)
        except Exception:
            return 120.0


def ticks_per_quarter() -> int:
    try:
        return int(gp.models.Duration.quarterTime)
    except Exception:
        return DEFAULT_TICKS_PER_QUARTER


def beat_duration_ticks(beat) -> int:
    dur = getattr(beat, "duration", None)
    if dur is None:
        return ticks_per_quarter()
    t = getattr(dur, "time", None)
    if t is not None:
        try:
            return max(1, int(t))
        except Exception:
            pass
    value = int(getattr(dur, "value", 4))
    is_dotted = bool(getattr(dur, "isDotted", False))
    base = int(ticks_per_quarter() * 4 / value)
    if is_dotted:
        base = int(round(base * 1.5))
    tuplet = getattr(dur, "tuplet", None)
    if tuplet is not None:
        enters = getattr(tuplet, "enters", 1) or 1
        times = getattr(tuplet, "times", 1) or 1
        base = int(round(base * times / enters))
    return max(1, base)


def load_onset_seconds(onset_path: Path, sr: int) -> float:
    if not onset_path.exists():
        return 0.0
    try:
        val = float(onset_path.read_text().strip().split()[0])
        return val / sr if val > 5.0 else max(0.0, val)
    except Exception:
        return 0.0


def normalize_audio(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return (x / m * peak).astype(np.float32) if m > 1e-8 else x


def ensure_len(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size >= n:
        return x[:n].copy()
    y = np.zeros(n, dtype=np.float32)
    y[:x.size] = x
    return y


def fade_edges(x: np.ndarray, fade_len: int = 128) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    fade_len = min(fade_len, x.size // 2)
    if fade_len <= 1:
        return x
    ramp = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    x[:fade_len] *= ramp
    x[-fade_len:] *= ramp[::-1]
    return x


# ---------------------------------------------------------------------
# Firebrand sample bank
# ---------------------------------------------------------------------
def load_firebrand_samples(
    audio_root: Path,
    onset_root: Optional[Path],
    sr: int,
    max_render_fret: int,
    guitar_prefix: str = "fender",
    n_instances: int = 1,
) -> Dict[Tuple[int, int], List[FirebrandSample]]:
    """
    Load Firebrand note samples.

    Supports two layouts.

    Layout A, old multi-instance layout:

        audio_root/
            firebrand1/
                string1/
                    0.wav
                    1.wav
                string2/
                ...
            firebrand2/
            ...

        onset_root/
            firebrand1/
                string1/
                    0.txt
                    1.txt
                ...

    Layout B, new flat layout:

        audio_root/
            string1/
                0.wav
                0.txt
                1.wav
                1.txt
                ...
            string2/
            ...

    In Layout B, onset_root can be None, because the .txt files are beside
    the .wav files.
    """

    bank: Dict[Tuple[int, int], List[FirebrandSample]] = defaultdict(list)

    audio_root = Path(audio_root)
    onset_root = Path(onset_root) if onset_root is not None else audio_root

    # ------------------------------------------------------------
    # Detect new flat layout:
    #   audio_root/string1/0.wav
    #   audio_root/string1/0.txt
    # ------------------------------------------------------------
    flat_layout = all((audio_root / f"string{s}").exists() for s in range(1, 7))

    string_sources = []

    if flat_layout:
        print("[info] Detected flat Firebrand layout: string1..string6 directly under audio root.")

        for gp_string in range(1, 7):
            string_dir = audio_root / f"string{gp_string}"

            # Prefer onset_root/stringX if it exists, otherwise use same folder as audio.
            candidate_onset_dir = onset_root / f"string{gp_string}"
            if candidate_onset_dir.exists():
                onset_string_dir = candidate_onset_dir
            else:
                onset_string_dir = string_dir

            string_sources.append(
                {
                    "label": "flat",
                    "gp_string": gp_string,
                    "string_dir": string_dir,
                    "onset_string_dir": onset_string_dir,
                }
            )

    else:
        print("[info] Detected old multi-instance Firebrand layout.")

        for gidx in range(1, n_instances + 1):
            guitar_dir = audio_root / f"{guitar_prefix}{gidx}"
            onset_guitar_dir = onset_root / f"{guitar_prefix}{gidx}"

            if not guitar_dir.exists():
                continue

            for gp_string in range(1, 7):
                string_dir = guitar_dir / f"string{gp_string}"

                # Prefer separate onset tree if present.
                candidate_onset_dir = onset_guitar_dir / f"string{gp_string}"
                if candidate_onset_dir.exists():
                    onset_string_dir = candidate_onset_dir
                else:
                    onset_string_dir = string_dir

                if not string_dir.exists():
                    continue

                string_sources.append(
                    {
                        "label": f"{guitar_prefix}{gidx}",
                        "gp_string": gp_string,
                        "string_dir": string_dir,
                        "onset_string_dir": onset_string_dir,
                    }
                )

    # ------------------------------------------------------------
    # Load samples from whichever layout was detected.
    # ------------------------------------------------------------
    for src in string_sources:
        gp_string = src["gp_string"]
        string_dir = src["string_dir"]
        onset_string_dir = src["onset_string_dir"]

        for fret in range(0, max_render_fret + 1):
            wav_path = string_dir / f"{fret}.wav"
            onset_path = onset_string_dir / f"{fret}.txt"

            if not wav_path.exists():
                continue

            try:
                y, src_sr = librosa.load(wav_path, sr=None, mono=True)

                onset_sec = load_onset_seconds(onset_path, src_sr)
                onset_sample = int(round(onset_sec * src_sr))
                onset_sample = max(0, min(onset_sample, len(y) - 1))

                y = y[onset_sample:]

                if src_sr != sr:
                    y = librosa.resample(
                        y.astype(np.float32),
                        orig_sr=src_sr,
                        target_sr=sr,
                    )

                y = normalize_audio(y, peak=1.0)

                bank[(gp_string, fret)].append(
                    FirebrandSample(
                        audio=y.astype(np.float32),
                        sr=sr,
                        gp_string=gp_string,
                        fret=fret,
                        source_path=str(wav_path),
                    )
                )

            except Exception as exc:
                print(f"[warn] failed to load {wav_path}: {exc}")

    return dict(bank)

def load_firebrand_samples_old(
    audio_root: Path,
    onset_root: Path,
    sr: int,
    max_render_fret: int,
    guitar_prefix: str = "firebrand",
    n_instances: int = 10,
) -> Dict[Tuple[int, int], List[FirebrandSample]]:
    bank: Dict[Tuple[int, int], List[FirebrandSample]] = defaultdict(list)
    for gidx in range(1, n_instances + 1):
        guitar_dir = audio_root / f"{guitar_prefix}{gidx}"
        onset_guitar_dir = onset_root / f"{guitar_prefix}{gidx}"
        if not guitar_dir.exists():
            continue
        for gp_string in range(1, 7):
            string_dir = guitar_dir / f"string{gp_string}"
            onset_string_dir = onset_guitar_dir / f"string{gp_string}"
            if not string_dir.exists():
                continue
            for fret in range(0, max_render_fret + 1):
                wav_path = string_dir / f"{fret}.wav"
                onset_path = onset_string_dir / f"{fret}.txt"
                if not wav_path.exists():
                    continue
                try:
                    y, src_sr = librosa.load(wav_path, sr=None, mono=True)
                    onset_sec = load_onset_seconds(onset_path, src_sr)
                    onset_sample = int(round(onset_sec * src_sr))
                    onset_sample = max(0, min(onset_sample, len(y) - 1))
                    y = y[onset_sample:]
                    if src_sr != sr:
                        y = librosa.resample(y.astype(np.float32), orig_sr=src_sr, target_sr=sr)
                    y = normalize_audio(y, peak=1.0)
                    bank[(gp_string, fret)].append(
                        FirebrandSample(y.astype(np.float32), sr, gp_string, fret, str(wav_path))
                    )
                except Exception as exc:
                    print(f"[warn] failed to load {wav_path}: {exc}")
    return dict(bank)


def choose_sample(bank: Dict[Tuple[int, int], List[FirebrandSample]], gp_string: int, fret: int, rng) -> Optional[np.ndarray]:
    candidates = bank.get((gp_string, fret), [])
    if not candidates:
        return None
    return candidates[int(rng.integers(0, len(candidates)))].audio


def render_segment_audio(events: Sequence[NoteEvent], seg_start: float, seg_dur: float, bank, sr: int, rng, tail_sec: float) -> np.ndarray:
    n = int(round(seg_dur * sr))
    audio = np.zeros(n, dtype=np.float32)
    for ev in events:
        local_start = ev.start_sec - seg_start
        local_end = ev.end_sec - seg_start
        if local_end <= 0 or local_start >= seg_dur:
            continue
        insert_at = int(round(max(0.0, local_start) * sr))
        desired_len = int(round((min(seg_dur, local_end) - max(0.0, local_start) + tail_sec) * sr))
        desired_len = max(1, desired_len)
        sample = choose_sample(bank, ev.gp_string, ev.fret, rng)
        if sample is None:
            continue
        sample = ensure_len(sample, desired_len)
        sample = sample * float(ev.velocity) * float(rng.uniform(0.75, 1.05))
        jitter = int(rng.normal(0, 0.004 * sr))  # 4 ms std
        insert_at = max(0, insert_at + jitter)
        if insert_at >= n:
            continue
        sample = fade_edges(sample[:n - insert_at], min(128, sample.size // 8))
        audio[insert_at:insert_at + sample.size] += sample
    return normalize_audio(audio, peak=0.95)


# ---------------------------------------------------------------------
# DadaGP / Guitar Pro parsing
# ---------------------------------------------------------------------

def collect_gp_files(root: Path, percentage: float, seed: int) -> List[Path]:
    exts = {".gp", ".gp3", ".gp4", ".gp5"}
    files = sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    rnd = random.Random(seed)
    selected = [p for p in files if rnd.random() <= percentage]
    if not selected and files:
        selected = [rnd.choice(files)]
    return selected


def is_probably_guitar_track(track) -> bool:
    if getattr(track, "isPercussionTrack", False):
        return False
    strings = getattr(track, "strings", [])
    return len(strings) >= 6


def parse_guitarpro_file(gp_path: Path, max_render_fret: int, max_tab_fret: int, min_notes: int) -> Tuple[float, Dict[str, List[NoteEvent]]]:
    song = gp.parse(str(gp_path))
    tempo = get_song_tempo(song)
    sec_per_tick = (60.0 / tempo) / float(ticks_per_quarter())
    tracks_events: Dict[str, List[NoteEvent]] = {}

    for track_idx, track in enumerate(getattr(song, "tracks", [])):
        if not is_probably_guitar_track(track):
            continue
        track_name = getattr(track, "name", f"track{track_idx}")
        safe_track_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", track_name)[:40]
        track_key = f"{track_idx:02d}_{safe_track_name}"
        raw = []
        min_tick = None

        for measure in getattr(track, "measures", []):
            for voice in getattr(measure, "voices", []):
                for beat in getattr(voice, "beats", []):
                    beat_start = getattr(beat, "start", None)
                    if beat_start is None:
                        continue
                    dur_ticks = beat_duration_ticks(beat)
                    for note in getattr(beat, "notes", []):
                        gp_string = int(getattr(note, "string", 0))
                        fret = int(getattr(note, "value", -1))
                        if gp_string < 1 or gp_string > 6 or fret < 0:
                            continue
                        if fret > max_tab_fret or fret > max_render_fret:
                            continue
                        tab_string = gp_string_to_tab_index(gp_string)
                        midi = midi_from_string_fret(tab_string, fret)
                        if midi < LOW_E_MIDI or midi >= LOW_E_MIDI + N_F0_CLASSES:
                            continue
                        velocity = max(0.1, min(1.2, float(getattr(note, "velocity", 95)) / 95.0))
                        start_tick = int(beat_start)
                        end_tick = int(beat_start + dur_ticks * float(getattr(note, "durationPercent", 1.0)))
                        min_tick = start_tick if min_tick is None else min(min_tick, start_tick)
                        raw.append((start_tick, end_tick, tab_string, gp_string, fret, midi, velocity))

        if min_tick is None or len(raw) < min_notes:
            continue

        events: List[NoteEvent] = []
        for st, en, tab_s, gp_s, fret, midi, vel in raw:
            start_sec = (st - min_tick) * sec_per_tick
            end_sec = max(start_sec + 0.03, (en - min_tick) * sec_per_tick)
            events.append(NoteEvent(start_sec, end_sec, tab_s, gp_s, fret, midi, vel, str(gp_path), track_name))
        events.sort(key=lambda e: (e.start_sec, e.tab_string, e.fret))
        tracks_events[track_key] = events
    return tempo, tracks_events


# ---------------------------------------------------------------------
# Labels and phantom hand
# ---------------------------------------------------------------------

def empty_tab(length: int) -> np.ndarray:
    y = np.zeros((length, N_STRINGS, N_TAB_CLASSES), dtype=np.float32)
    y[:, :, NO_PLAY] = 1.0
    return y


def add_tab_sustain(y: np.ndarray, t0: int, t1: int, s: int, fret: int) -> None:
    if fret < 0 or fret >= NO_PLAY:
        return
    t0, t1 = int(t0), int(t1)
    t0 = max(0, min(y.shape[0], t0))
    t1 = max(0, min(y.shape[0], t1))
    if t1 <= t0:
        t1 = min(y.shape[0], t0 + 1)
    if t0 >= y.shape[0]:
        return
    y[t0:t1, s, NO_PLAY] = 0.0
    y[t0:t1, s, fret] = 1.0


def add_tab_onset(y: np.ndarray, t: int, s: int, fret: int) -> None:
    if fret < 0 or fret >= NO_PLAY:
        return
    if 0 <= t < y.shape[0]:
        y[t, s, NO_PLAY] = 0.0
        y[t, s, fret] = 1.0


def add_f0_sustain(y: np.ndarray, t0: int, t1: int, midi: int) -> None:
    idx = int(midi - LOW_E_MIDI)
    if idx < 0 or idx >= y.shape[1]:
        return
    t0 = max(0, min(y.shape[0], int(t0)))
    t1 = max(0, min(y.shape[0], int(t1)))
    if t1 <= t0:
        t1 = min(y.shape[0], t0 + 1)
    if t0 < y.shape[0]:
        y[t0:t1, idx] = 1.0


def add_f0_onset(y: np.ndarray, t: int, midi: int) -> None:
    idx = int(midi - LOW_E_MIDI)
    if 0 <= idx < y.shape[1] and 0 <= t < y.shape[0]:
        y[t, idx] = 1.0


def make_labels(events: Sequence[NoteEvent], seg_start: float, seg_dur: float, tempo: float, sr: int, hop: int, note_resolution: int) -> Dict[str, np.ndarray]:
    note_dur = 60.0 / tempo / note_resolution * 4.0
    len_notes = note_resolution * 4  # 64 if note_resolution=16
    frame_len = int(round(seg_dur * sr / hop))
    norm_len = sr / float(hop)

    tab = empty_tab(len_notes)
    tab_onset = empty_tab(len_notes)
    frame_tab = empty_tab(frame_len)
    frame_tab_onset = empty_tab(frame_len)
    F0 = np.zeros((len_notes, N_F0_CLASSES), dtype=np.float32)
    F0_onset = np.zeros((len_notes, N_F0_CLASSES), dtype=np.float32)
    frame_F0 = np.zeros((frame_len, N_F0_CLASSES), dtype=np.float32)
    frame_F0_onset = np.zeros((frame_len, N_F0_CLASSES), dtype=np.float32)

    for ev in events:
        local_st = ev.start_sec - seg_start
        local_en = ev.end_sec - seg_start
        if local_en <= 0 or local_st >= seg_dur:
            continue
        clipped_st = max(0.0, local_st)
        clipped_en = min(seg_dur, local_en)
        n0 = int(round(clipped_st / note_dur))
        n1 = int(math.ceil(clipped_en / note_dur))
        f0 = int(round(clipped_st * norm_len))
        f1 = int(math.ceil(clipped_en * norm_len))
        add_tab_sustain(tab, n0, n1, ev.tab_string, ev.fret)
        add_tab_sustain(frame_tab, f0, f1, ev.tab_string, ev.fret)
        add_f0_sustain(F0, n0, n1, ev.midi)
        add_f0_sustain(frame_F0, f0, f1, ev.midi)
        if 0 <= local_st < seg_dur:
            onset_n = int(round(local_st / note_dur))
            onset_f = int(round(local_st * norm_len))
            add_tab_onset(tab_onset, onset_n, ev.tab_string, ev.fret)
            add_tab_onset(frame_tab_onset, onset_f, ev.tab_string, ev.fret)
            add_f0_onset(F0_onset, onset_n, ev.midi)
            add_f0_onset(frame_F0_onset, onset_f, ev.midi)

    return dict(
        tab=tab, tab_onset=tab_onset,
        frame_tab=frame_tab, frame_tab_onset=frame_tab_onset,
        F0=F0, F0_onset=F0_onset,
        frame_F0=frame_F0, frame_F0_onset=frame_F0_onset,
    )


def tab_step_to_pattern_key(step: np.ndarray) -> str:
    parts = []
    for s in range(N_STRINGS):
        cls = int(np.argmax(step[s]))
        if cls != NO_PLAY:
            parts.append(f"s{s}f{cls}")
    return "+".join(parts) if parts else "REST"


def fretted_span(step: np.ndarray) -> int:
    frets = []
    for s in range(N_STRINGS):
        cls = int(np.argmax(step[s]))
        if cls != NO_PLAY and cls > 0:
            frets.append(cls)
    return max(frets) - min(frets) if frets else 0


def hand_heatmap(active_frets: Sequence[int], n_frets: int, rng, sigma: float = 1.35) -> Tuple[np.ndarray, float]:
    fretted = [int(f) for f in active_frets if 0 < int(f) < n_frets]
    if not fretted:
        return np.zeros(n_frets, dtype=np.float32), 0.0
    min_f, max_f = min(fretted), max(fretted)
    center = int(round((min_f + max_f) / 2.0))
    # camera-like shift error; zero is most likely
    shifts = np.array([-2, -1, 0, 1, 2])
    probs = np.array([0.08, 0.18, 0.48, 0.18, 0.08])
    center += int(rng.choice(shifts, p=probs))
    center = int(np.clip(center, 0, n_frets - 1))
    if rng.random() < 0.15:
        sigma *= rng.uniform(1.4, 2.2)
    x = np.arange(n_frets, dtype=np.float32)
    h = np.exp(-0.5 * ((x - center) / sigma) ** 2).astype(np.float32)
    h[max(0, min_f - 1):min(n_frets, max_f + 2)] = np.maximum(h[max(0, min_f - 1):min(n_frets, max_f + 2)], 0.65)
    if rng.random() < 0.10:  # missing camera frame
        return np.zeros(n_frets, dtype=np.float32), 0.0
    conf = float(rng.uniform(0.65, 1.0))
    return (h * conf).astype(np.float32), conf


def hand_from_tab(tab: np.ndarray, n_frets: int, rng) -> Tuple[np.ndarray, np.ndarray]:
    T = tab.shape[0]
    hand = np.zeros((T, n_frets), dtype=np.float32)
    conf = np.zeros(T, dtype=np.float32)
    for t in range(T):
        active = []
        for s in range(N_STRINGS):
            cls = int(np.argmax(tab[t, s]))
            if cls != NO_PLAY:
                active.append(cls)
        hand[t], conf[t] = hand_heatmap(active, n_frets, rng)
    # tiny temporal smoothing, like video tracking
    if T >= 3:
        k = np.array([0.2, 0.6, 0.2], dtype=np.float32)
        for f in range(n_frets):
            hand[:, f] = np.convolve(hand[:, f], k, mode="same")
        conf = np.convolve(conf, k, mode="same").astype(np.float32)
    return hand, conf


# ---------------------------------------------------------------------
# Features and segmentation
# ---------------------------------------------------------------------

def fix_frames(x: np.ndarray, frames: int) -> np.ndarray:
    if x.shape[0] >= frames:
        return x[:frames].astype(np.float32)
    pad = np.zeros((frames - x.shape[0], x.shape[1]), dtype=np.float32)
    return np.vstack([x, pad]).astype(np.float32)


def compute_features(audio: np.ndarray, sr: int, hop: int, cqt_bins: int, bpo: int, frames: int) -> Dict[str, np.ndarray]:
    audio = librosa.util.normalize(audio.astype(np.float32))
    cqt = np.abs(librosa.cqt(audio, sr=sr, hop_length=hop, n_bins=cqt_bins, bins_per_octave=bpo)).T.astype(np.float32)
    log_cqt = librosa.amplitude_to_db(np.abs(cqt)).astype(np.float32)
    mel = np.abs(librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=hop)).T.astype(np.float32)
    return dict(cqt=fix_frames(cqt, frames), log_cqt=fix_frames(log_cqt, frames), mel_spec=fix_frames(mel, frames))


def segment_starts(events: Sequence[NoteEvent], tempo: float, note_resolution: int) -> List[Tuple[float, float]]:
    if not events:
        return []
    note_dur = 60.0 / tempo / note_resolution * 4.0
    seg_dur = note_resolution * 4 * note_dur  # 4 bars
    end = max(e.end_sec for e in events)
    return [(i * seg_dur, seg_dur) for i in range(int(math.ceil(end / seg_dur)))]


def events_in_segment(events: Sequence[NoteEvent], st: float, dur: float) -> List[NoteEvent]:
    en = st + dur
    return [e for e in events if e.end_sec > st and e.start_sec < en]


def active_steps(tab: np.ndarray) -> int:
    return sum(1 for t in range(tab.shape[0]) if tab_step_to_pattern_key(tab[t]) != "REST")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def build(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading Firebrand samples...")
    bank = load_firebrand_samples(Path(args.firebrand_audio_root), Path(args.firebrand_onset_root), args.sr, args.max_render_fret, args.guitar_prefix, args.n_firebrand_instances)
    if not bank:
        raise SystemExit("No Firebrand samples loaded. Check paths and folder names.")
    print(f"Loaded {sum(len(v) for v in bank.values())} note instances across {len(bank)} string/fret keys.")

    print("[2/5] Selecting DadaGP files...")
    gp_files = collect_gp_files(Path(args.dadagp_root), args.data_percentage, args.seed)
    print(f"Selected {len(gp_files)} files.")

    pattern_counts = Counter()
    failed = []
    skipped = Counter()
    written = 0

    print("[3/5] Creating .npz segments...")
    for file_idx, gp_path in enumerate(tqdm(gp_files)):
        fold = f"{file_idx % args.n_folds:02d}"
        try:
            tempo, tracks = parse_guitarpro_file(gp_path, args.max_render_fret, args.max_tab_fret, args.min_notes_per_track)
        except Exception as exc:
            failed.append({"file": str(gp_path), "error": repr(exc)})
            skipped["parse_error"] += 1
            continue
        if tempo < args.min_tempo or tempo > args.max_tempo:
            skipped["tempo_out_of_range"] += 1
            continue
        for track_key, events in tracks.items():
            for seg_idx, (seg_st, seg_dur) in enumerate(segment_starts(events, tempo, args.note_resolution)):
                seg_events = events_in_segment(events, seg_st, seg_dur)
                if not seg_events:
                    skipped["empty_segment"] += 1
                    continue
                labels = make_labels(seg_events, seg_st, seg_dur, tempo, args.sr, args.hop_length, args.note_resolution)
                tab = labels["tab"]
                if active_steps(tab) < args.min_active_steps:
                    skipped["not_enough_active_steps"] += 1
                    continue
                too_wide = False
                pattern_keys = []
                for t in range(tab.shape[0]):
                    key = tab_step_to_pattern_key(tab[t])
                    pattern_keys.append(key)
                    if key != "REST":
                        pattern_counts[key] += 1
                        if fretted_span(tab[t]) > args.max_fret_span:
                            too_wide = True
                if too_wide and args.skip_wide_fingering_patterns:
                    skipped["wide_fingering_pattern"] += 1
                    continue

                audio = render_segment_audio(seg_events, seg_st, seg_dur, bank, args.sr, rng, args.sample_tail_sec)
                frames = labels["frame_tab"].shape[0]
                feats = compute_features(audio, args.sr, args.hop_length, args.cqt_n_bins, args.bins_per_octave, frames)

                out_name = f"{fold}_{safe_name(gp_path)}_{track_key}_seg{seg_idx:04d}.npz"
                out_path = out_dir / out_name

                save_payload = dict(
                    cqt=feats["cqt"], log_cqt=feats["log_cqt"], mel_spec=feats["mel_spec"],
                    tab=labels["tab"], tab_onset=labels["tab_onset"],
                    frame_tab=labels["frame_tab"], frame_tab_onset=labels["frame_tab_onset"],
                    F0=labels["F0"], F0_onset=labels["F0_onset"],
                    frame_F0=labels["frame_F0"], frame_F0_onset=labels["frame_F0_onset"],
                    tempo=np.array(float(tempo), dtype=np.float32),
                    len_in_notes=np.array(args.note_resolution * 4, dtype=np.int32),
                    source_gp=str(gp_path), source_track=track_key,
                    segment_start_sec=np.array(seg_st, dtype=np.float32),
                    segment_duration_sec=np.array(seg_dur, dtype=np.float32),
                    pattern_keys=np.array(pattern_keys),
                )

                if args.hand_mode == "phantom":
                    phantom_hand, phantom_conf = hand_from_tab(labels["tab"], args.hand_frets, rng)
                    frame_phantom_hand, frame_phantom_conf = hand_from_tab(labels["frame_tab"], args.hand_frets, rng)
                    save_payload.update(
                        phantom_hand=phantom_hand,
                        frame_phantom_hand=frame_phantom_hand,
                        phantom_hand_conf=phantom_conf,
                        frame_phantom_hand_conf=frame_phantom_conf,
                    )
                elif args.hand_mode != "none":
                    raise ValueError(f"Unknown --hand-mode: {args.hand_mode}")

                np.savez_compressed(out_path, **save_payload)

                if args.write_mp3:
                    mp3_dir = Path(args.mp3_dir) if args.mp3_dir else (out_dir.parent / "mp3_debug")
                    mp3_name = out_path.with_suffix(".mp3").name
                    mp3_path = mp3_dir / mp3_name
                    try:
                        write_low_quality_mp3(
                            audio=audio,
                            sr=args.sr,
                            mp3_path=mp3_path,
                            bitrate=args.mp3_bitrate,
                        )
                    except Exception as exc:
                        skipped["mp3_write_failed"] += 1
                        if args.fail_on_mp3_error:
                            raise
                        print(f"[warn] MP3 write failed for {mp3_path}: {exc}")

                written += 1
                if args.max_segments is not None and written >= args.max_segments:
                    break
            if args.max_segments is not None and written >= args.max_segments:
                break
        if args.max_segments is not None and written >= args.max_segments:
            break

    print("[4/5] Saving logs...")
    parent = out_dir.parent
    with (parent / "fingering_pattern_counts.json").open("w", encoding="utf-8") as f:
        json.dump(pattern_counts.most_common(), f, indent=2)
    with (parent / "failed_files.json").open("w", encoding="utf-8") as f:
        json.dump(failed, f, indent=2)
    stats = dict(
        written_npz_segments=written,
        selected_gp_files=len(gp_files),
        failed_files=len(failed),
        skipped=dict(skipped),
        output_dir=str(out_dir),
        config=vars(args),
    )
    with (parent / "dataset_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print("[5/5] Done.")
    print(json.dumps(stats, indent=2))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create Firebrand/DadaGP .npz files for Tab-Estimator, with optional phantom-hand conditioning.")
    p.add_argument("--dadagp-root", default="data/DadaGP-v1.1")
    p.add_argument("--firebrand-audio-root", default="data/guitar_samples")
    p.add_argument("--firebrand-onset-root", default="data/onsets")
    p.add_argument("--out-dir", default="data/npz/firebrand_phantom/split")
    p.add_argument("--data-percentage", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-folds", type=int, default=6)
    p.add_argument("--guitar-prefix", default="fender")
    p.add_argument("--n-firebrand-instances", type=int, default=10)
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--hop-length", type=int, default=512)
    p.add_argument("--cqt-n-bins", type=int, default=192)
    p.add_argument("--bins-per-octave", type=int, default=24)
    p.add_argument("--note-resolution", type=int, default=16)
    p.add_argument("--max-tab-fret", type=int, default=19)
    p.add_argument("--max-render-fret", type=int, default=12)
    p.add_argument("--hand-frets", type=int, default=20)
    p.add_argument(
        "--hand-mode",
        choices=["none", "phantom"],
        default="none",
        help=(
            "Choose whether to include hand-conditioning arrays in the .npz files. "
            "Use 'none' for standard audio-only Tab-Estimator-style data. "
            "Use 'phantom' to add phantom_hand/frame_phantom_hand arrays for a modified hand-conditioned model."
        ),
    )
    p.add_argument("--min-notes-per-track", type=int, default=8)
    p.add_argument("--min-active-steps", type=int, default=2)
    p.add_argument("--max-fret-span", type=int, default=5)
    p.add_argument("--skip-wide-fingering-patterns", action="store_true")
    p.add_argument("--min-tempo", type=float, default=40.0)
    p.add_argument("--max-tempo", type=float, default=240.0)
    p.add_argument("--sample-tail-sec", type=float, default=0.25)

    # Optional listening/debug output. The .npz remains the training artifact.
    p.add_argument("--write-mp3", action="store_true",
                   help="Also render a low-quality MP3 for every saved .npz segment.")
    p.add_argument("--mp3-dir", default=None,
                   help="Directory for MP3 debug files. Default: <out-dir parent>/mp3_debug")
    p.add_argument("--mp3-bitrate", default="32k",
                   help="Low-quality MP3 bitrate, for example 24k, 32k, or 48k.")
    p.add_argument("--fail-on-mp3-error", action="store_true",
                   help="Stop the dataset build if an MP3 cannot be written.")

    p.add_argument("--max-segments", type=int, default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
