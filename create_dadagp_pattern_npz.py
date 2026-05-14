#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_dadagp_pattern_npz.py

Pattern-first dataset creator for Tab-Estimator.

This script mines unique fingering/chord patterns from DadaGP Guitar Pro files,
renders each pattern using your recorded note samples, and saves one
Tab-Estimator-style .npz file per rendered pattern.

It is different from the segment-based generator:
    segment generator:
        DadaGP track -> 4-bar musical sequence -> .npz

    this script:
        DadaGP -> unique tab/chord/fingering patterns -> isolated rendered pattern -> .npz

Goal
----
Create at least --target-patterns isolated fingering/chord pattern .npz files.

Default output:
    data/npz/dadagp_patterns/split/

Each .npz contains the usual Tab-Estimator keys:
    cqt, log_cqt, mel_spec,
    tab, tab_onset, frame_tab, frame_tab_onset,
    F0, F0_onset, frame_F0, frame_F0_onset,
    tempo, len_in_notes

Optional extra hand-conditioning keys when --hand-mode phantom:
    phantom_hand, frame_phantom_hand,
    phantom_hand_conf, frame_phantom_hand_conf

Supported sample layouts
------------------------
Layout A, old multi-instance:
    audio_root/
        firebrand1/string1/0.wav
        firebrand1/string1/0.txt
        firebrand2/string1/0.wav
        ...

Layout B, your new flat layout:
    audio_root/
        string1/
            0.wav
            0.txt
            1.wav
            1.txt
            ...
        string2/
        ...

Important assumptions
---------------------
- PyGuitarPro string 1 is high e and string 6 is low E.
- Tab-Estimator string axis is low E -> high e.
- Sample folders use string1=high e ... string6=low E.
- Tab-Estimator default tab classes are:
      frets 0..19
      class 20 = not played
- If your sample folder contains frets 0..21, this script still defaults to
  max fret 19 because the original Tab-Estimator output supports only 0..19.

Example
-------
python create_dadagp_pattern_npz.py \
  --dadagp-root data/DadaGP-v1.1/DadaGP-v1.1 \
  --sample-root data/fender_samples \
  --out-dir data/npz/dadagp_patterns/split \
  --data-percentage 0.05 \
  --target-patterns 1000 \
  --max-render-fret 19 \
  --max-tab-fret 19 \
  --hand-mode none \
  --write-mp3

For phantom hand:
python create_dadagp_pattern_npz.py \
  --dadagp-root data/DadaGP-v1.1/DadaGP-v1.1 \
  --sample-root data/fender_samples \
  --out-dir data/npz/dadagp_patterns_phantom/split \
  --data-percentage 0.05 \
  --target-patterns 1000 \
  --hand-mode phantom
"""

import argparse
import json
import math
import os
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


N_STRINGS = 6
NO_PLAY = 20
N_TAB_CLASSES = 21
N_F0_CLASSES = 44
LOW_E_MIDI = 40
OPEN_MIDI_LOW_TO_HIGH = np.array([40, 45, 50, 55, 59, 64], dtype=np.int32)


@dataclass
class FirebrandSample:
    audio: np.ndarray
    sr: int
    gp_string: int
    fret: int
    source_path: str


@dataclass(frozen=True)
class Pattern:
    """
    notes tuple entries:
        (tab_string, gp_string, fret, midi)
    """
    notes: Tuple[Tuple[int, int, int, int], ...]


def safe_name_text(s: str, max_len: int = 90) -> str:
    s = re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(s)).strip("_")
    return (s or "pattern")[:max_len]


def gp_string_to_tab_index(gp_string: int) -> int:
    return 6 - int(gp_string)


def tab_index_to_gp_string(tab_string: int) -> int:
    return 6 - int(tab_string)


def midi_from_string_fret(tab_string: int, fret: int) -> int:
    return int(OPEN_MIDI_LOW_TO_HIGH[tab_string] + fret)


def pattern_key(pattern: Pattern) -> str:
    if not pattern.notes:
        return "REST"
    return "+".join([f"s{tab_s}f{fret}" for tab_s, gp_s, fret, midi in pattern.notes])


def fretted_span(pattern: Pattern) -> int:
    frets = [fret for tab_s, gp_s, fret, midi in pattern.notes if fret > 0]
    return max(frets) - min(frets) if frets else 0


def normalize_audio(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    m = float(np.max(np.abs(x)))
    if m > 1e-8:
        x = x / m * peak
    return x.astype(np.float32)


def ensure_len(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size >= n:
        return x[:n].copy()
    y = np.zeros(n, dtype=np.float32)
    y[:x.size] = x
    return y


def fade_edges(x: np.ndarray, fade_len: int = 128) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    if x.size == 0:
        return x
    fade_len = min(fade_len, x.size // 2)
    if fade_len <= 1:
        return x
    ramp = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    x[:fade_len] *= ramp
    x[-fade_len:] *= ramp[::-1]
    return x


def load_onset_seconds(onset_path: Path, sr: int) -> float:
    if not onset_path.exists():
        return 0.0
    try:
        val = float(onset_path.read_text().strip().split()[0])
        return val / sr if val > 5.0 else max(0.0, val)
    except Exception:
        return 0.0


def write_low_quality_mp3(audio: np.ndarray, sr: int, mp3_path: Path, bitrate: str = "32k") -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on PATH. Install ffmpeg or run without --write-mp3.")
    mp3_path = Path(mp3_path)
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    audio = normalize_audio(audio, peak=0.95)
    audio_i16 = np.clip(audio, -1.0, 1.0)
    audio_i16 = (audio_i16 * 32767.0).astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = Path(tmp.name)

    try:
        with wave.open(str(tmp_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sr))
            wf.writeframes(audio_i16.tobytes())
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_wav),
             "-codec:a", "libmp3lame", "-b:a", str(bitrate), str(mp3_path)],
            check=True,
        )
    finally:
        try:
            tmp_wav.unlink()
        except FileNotFoundError:
            pass


def load_note_samples(
    sample_root: Path,
    onset_root: Optional[Path],
    sr: int,
    max_render_fret: int,
    guitar_prefix: str = "fender",
    n_instances: int = 1,
) -> Dict[Tuple[int, int], List[FirebrandSample]]:
    """
    Load note samples from either:
      new flat layout:
          sample_root/string1/0.wav and sample_root/string1/0.txt
      old multi-instance layout:
          sample_root/fender1/string1/0.wav
    """
    bank: Dict[Tuple[int, int], List[FirebrandSample]] = defaultdict(list)
    sample_root = Path(sample_root)
    onset_root = Path(onset_root) if onset_root is not None else sample_root

    flat_layout = all((sample_root / f"string{s}").exists() for s in range(1, 7))
    string_sources = []

    if flat_layout:
        print("[info] Detected flat sample layout: string1..string6 directly under sample root.")
        for gp_string in range(1, 7):
            string_dir = sample_root / f"string{gp_string}"
            candidate_onset_dir = onset_root / f"string{gp_string}"
            onset_string_dir = candidate_onset_dir if candidate_onset_dir.exists() else string_dir
            string_sources.append((gp_string, string_dir, onset_string_dir, "flat"))
    else:
        print("[info] Detected old multi-instance sample layout.")
        for gidx in range(1, n_instances + 1):
            guitar_dir = sample_root / f"{guitar_prefix}{gidx}"
            onset_guitar_dir = onset_root / f"{guitar_prefix}{gidx}"
            if not guitar_dir.exists():
                continue
            for gp_string in range(1, 7):
                string_dir = guitar_dir / f"string{gp_string}"
                candidate_onset_dir = onset_guitar_dir / f"string{gp_string}"
                onset_string_dir = candidate_onset_dir if candidate_onset_dir.exists() else string_dir
                if string_dir.exists():
                    string_sources.append((gp_string, string_dir, onset_string_dir, f"{guitar_prefix}{gidx}"))

    for gp_string, string_dir, onset_string_dir, label in string_sources:
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


def sample_exists(bank: Dict[Tuple[int, int], List[FirebrandSample]], gp_string: int, fret: int) -> bool:
    return len(bank.get((gp_string, fret), [])) > 0


def choose_sample(bank: Dict[Tuple[int, int], List[FirebrandSample]], gp_string: int, fret: int, rng) -> Optional[np.ndarray]:
    candidates = bank.get((gp_string, fret), [])
    if not candidates:
        return None
    return candidates[int(rng.integers(0, len(candidates)))].audio


def collect_gp_files(root: Path, percentage: float, seed: int) -> List[Path]:
    exts = {".gp", ".gp3", ".gp4", ".gp5"}
    files = sorted([p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in exts])
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


def beat_to_pattern(
    beat,
    max_tab_fret: int,
    max_render_fret: int,
    bank: Dict[Tuple[int, int], List[FirebrandSample]],
    require_available_samples: bool = True,
) -> Optional[Pattern]:
    by_string: Dict[int, Tuple[int, int, int, int]] = {}
    for note in getattr(beat, "notes", []):
        gp_string = int(getattr(note, "string", 0))
        fret = int(getattr(note, "value", -1))
        if gp_string < 1 or gp_string > 6:
            continue
        if fret < 0 or fret > max_tab_fret or fret > max_render_fret:
            continue
        if require_available_samples and not sample_exists(bank, gp_string, fret):
            continue
        tab_string = gp_string_to_tab_index(gp_string)
        midi = midi_from_string_fret(tab_string, fret)
        if midi < LOW_E_MIDI or midi >= LOW_E_MIDI + N_F0_CLASSES:
            continue
        by_string[tab_string] = (tab_string, gp_string, fret, midi)
    if not by_string:
        return None
    return Pattern(notes=tuple(sorted(by_string.values(), key=lambda x: x[0])))


def mine_patterns_from_file(
    gp_path: Path,
    bank: Dict[Tuple[int, int], List[FirebrandSample]],
    max_tab_fret: int,
    max_render_fret: int,
    require_available_samples: bool,
) -> List[Pattern]:
    song = gp.parse(str(gp_path))
    patterns: List[Pattern] = []
    for track in getattr(song, "tracks", []):
        if not is_probably_guitar_track(track):
            continue
        for measure in getattr(track, "measures", []):
            for voice in getattr(measure, "voices", []):
                for beat in getattr(voice, "beats", []):
                    pat = beat_to_pattern(beat, max_tab_fret, max_render_fret, bank, require_available_samples)
                    if pat is not None:
                        patterns.append(pat)
    return patterns


def pattern_from_key(key: str) -> Pattern:
    if key == "REST":
        return Pattern(notes=tuple())
    notes = []
    for part in key.split("+"):
        m = re.match(r"s(\d+)f(\d+)$", part)
        if not m:
            raise ValueError(f"Bad pattern key: {key}")
        tab_s = int(m.group(1))
        fret = int(m.group(2))
        gp_s = tab_index_to_gp_string(tab_s)
        midi = midi_from_string_fret(tab_s, fret)
        notes.append((tab_s, gp_s, fret, midi))
    return Pattern(notes=tuple(sorted(notes, key=lambda x: x[0])))


def augment_patterns_by_transposition(
    pattern_counts: Counter,
    bank: Dict[Tuple[int, int], List[FirebrandSample]],
    max_tab_fret: int,
    max_render_fret: int,
    max_shift_up: int = 12,
    max_shift_down: int = 12,
) -> Counter:
    augmented = Counter(pattern_counts)
    original = []
    for key, count in pattern_counts.items():
        try:
            pat = pattern_from_key(key)
        except Exception:
            continue
        if pat.notes:
            original.append((pat, count))

    for pat, count in original:
        # Avoid transposing open-string patterns; keep them as mined.
        if any(fret == 0 for tab_s, gp_s, fret, midi in pat.notes):
            continue
        for shift in range(-max_shift_down, max_shift_up + 1):
            if shift == 0:
                continue
            new_notes = []
            possible = True
            for tab_s, gp_s, fret, midi in pat.notes:
                nf = fret + shift
                if nf < 1 or nf > max_tab_fret or nf > max_render_fret:
                    possible = False
                    break
                if not sample_exists(bank, gp_s, nf):
                    possible = False
                    break
                nmidi = midi_from_string_fret(tab_s, nf)
                if nmidi < LOW_E_MIDI or nmidi >= LOW_E_MIDI + N_F0_CLASSES:
                    possible = False
                    break
                new_notes.append((tab_s, gp_s, nf, nmidi))
            if possible:
                new_pat = Pattern(tuple(sorted(new_notes, key=lambda x: x[0])))
                augmented[pattern_key(new_pat)] += max(1, count // 2)
    return augmented


def empty_tab(length: int) -> np.ndarray:
    y = np.zeros((length, N_STRINGS, N_TAB_CLASSES), dtype=np.float32)
    y[:, :, NO_PLAY] = 1.0
    return y


def add_tab_sustain(y: np.ndarray, t0: int, t1: int, s: int, fret: int) -> None:
    if fret < 0 or fret >= NO_PLAY:
        return
    t0 = max(0, min(y.shape[0], int(t0)))
    t1 = max(0, min(y.shape[0], int(t1)))
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


def make_pattern_labels(
    pattern: Pattern,
    len_notes: int,
    frame_len: int,
    onset_note_idx: int,
    sustain_note_steps: int,
    sr: int,
    hop: int,
    tempo: float,
    note_resolution: int,
) -> Dict[str, np.ndarray]:
    tab = empty_tab(len_notes)
    tab_onset = empty_tab(len_notes)
    frame_tab = empty_tab(frame_len)
    frame_tab_onset = empty_tab(frame_len)
    F0 = np.zeros((len_notes, N_F0_CLASSES), dtype=np.float32)
    F0_onset = np.zeros((len_notes, N_F0_CLASSES), dtype=np.float32)
    frame_F0 = np.zeros((frame_len, N_F0_CLASSES), dtype=np.float32)
    frame_F0_onset = np.zeros((frame_len, N_F0_CLASSES), dtype=np.float32)

    note_dur = 60.0 / float(tempo) / float(note_resolution) * 4.0
    onset_sec = onset_note_idx * note_dur
    sustain_sec = sustain_note_steps * note_dur
    onset_frame_idx = int(round(onset_sec * sr / hop))
    end_note_idx = min(len_notes, onset_note_idx + sustain_note_steps)
    end_frame_idx = min(frame_len, int(math.ceil((onset_sec + sustain_sec) * sr / hop)))

    for tab_s, gp_s, fret, midi in pattern.notes:
        add_tab_sustain(tab, onset_note_idx, end_note_idx, tab_s, fret)
        add_tab_onset(tab_onset, onset_note_idx, tab_s, fret)
        add_tab_sustain(frame_tab, onset_frame_idx, end_frame_idx, tab_s, fret)
        add_tab_onset(frame_tab_onset, onset_frame_idx, tab_s, fret)
        add_f0_sustain(F0, onset_note_idx, end_note_idx, midi)
        add_f0_onset(F0_onset, onset_note_idx, midi)
        add_f0_sustain(frame_F0, onset_frame_idx, end_frame_idx, midi)
        add_f0_onset(frame_F0_onset, onset_frame_idx, midi)

    return dict(
        tab=tab, tab_onset=tab_onset,
        frame_tab=frame_tab, frame_tab_onset=frame_tab_onset,
        F0=F0, F0_onset=F0_onset,
        frame_F0=frame_F0, frame_F0_onset=frame_F0_onset,
    )


def hand_heatmap(active_frets: Sequence[int], n_frets: int, rng, sigma: float = 1.35) -> Tuple[np.ndarray, float]:
    fretted = [int(f) for f in active_frets if 0 < int(f) < n_frets]
    if not fretted:
        return np.zeros(n_frets, dtype=np.float32), 0.0
    min_f, max_f = min(fretted), max(fretted)
    center = int(round((min_f + max_f) / 2.0))
    shifts = np.array([-2, -1, 0, 1, 2])
    probs = np.array([0.08, 0.18, 0.48, 0.18, 0.08])
    center += int(rng.choice(shifts, p=probs))
    center = int(np.clip(center, 0, n_frets - 1))
    if rng.random() < 0.15:
        sigma *= rng.uniform(1.4, 2.2)
    x = np.arange(n_frets, dtype=np.float32)
    h = np.exp(-0.5 * ((x - center) / sigma) ** 2).astype(np.float32)
    h[max(0, min_f - 1):min(n_frets, max_f + 2)] = np.maximum(
        h[max(0, min_f - 1):min(n_frets, max_f + 2)], 0.65
    )
    if rng.random() < 0.10:
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
    if T >= 3:
        k = np.array([0.2, 0.6, 0.2], dtype=np.float32)
        for f in range(n_frets):
            hand[:, f] = np.convolve(hand[:, f], k, mode="same")
        conf = np.convolve(conf, k, mode="same").astype(np.float32)
    return hand.astype(np.float32), conf.astype(np.float32)


def render_pattern_audio(
    pattern: Pattern,
    bank: Dict[Tuple[int, int], List[FirebrandSample]],
    sr: int,
    segment_sec: float,
    onset_sec: float,
    sustain_sec: float,
    rng,
    tail_sec: float = 0.25,
) -> np.ndarray:
    n = int(round(segment_sec * sr))
    audio = np.zeros(n, dtype=np.float32)
    insert_at = int(round(onset_sec * sr))
    desired_len = max(1, int(round((sustain_sec + tail_sec) * sr)))

    for tab_s, gp_s, fret, midi in pattern.notes:
        sample = choose_sample(bank, gp_s, fret, rng)
        if sample is None:
            continue
        sample = ensure_len(sample, desired_len)
        sample = sample * float(rng.uniform(0.75, 1.05))
        jitter = int(rng.normal(0, 0.003 * sr))
        start = max(0, insert_at + jitter)
        if start >= n:
            continue
        sample = sample[:n - start]
        sample = fade_edges(sample, min(128, sample.size // 8))
        audio[start:start + sample.size] += sample

    return normalize_audio(audio, peak=0.95)


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


def atomic_savez_compressed(path: Path, **kwargs) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        np.savez_compressed(tmp_path, **kwargs)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def build(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parent = out_dir.parent
    parent.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading note samples...")
    onset_root = Path(args.onset_root) if args.onset_root else None
    bank = load_note_samples(
        sample_root=Path(args.sample_root),
        onset_root=onset_root,
        sr=args.sr,
        max_render_fret=args.max_render_fret,
        guitar_prefix=args.guitar_prefix,
        n_instances=args.n_instances,
    )
    if not bank:
        raise SystemExit("No note samples loaded. Check --sample-root and folder structure.")
    print(f"Loaded {sum(len(v) for v in bank.values())} note instances across {len(bank)} string/fret keys.")

    print("[2/6] Selecting DadaGP files...")
    gp_files = collect_gp_files(Path(args.dadagp_root), args.data_percentage, args.seed)
    print(f"Selected {len(gp_files)} files from {args.dadagp_root}")

    print("[3/6] Mining patterns...")
    pattern_counts = Counter()
    failed = []

    for gp_path in tqdm(gp_files):
        try:
            pats = mine_patterns_from_file(
                gp_path=gp_path,
                bank=bank,
                max_tab_fret=args.max_tab_fret,
                max_render_fret=args.max_render_fret,
                require_available_samples=not args.allow_patterns_without_samples,
            )
            for pat in pats:
                if len(pat.notes) == 0:
                    continue
                if args.min_notes_per_pattern and len(pat.notes) < args.min_notes_per_pattern:
                    continue
                if args.max_notes_per_pattern and len(pat.notes) > args.max_notes_per_pattern:
                    continue
                if args.skip_wide_fingering_patterns and fretted_span(pat) > args.max_fret_span:
                    continue
                pattern_counts[pattern_key(pat)] += 1
        except Exception as exc:
            failed.append({"file": str(gp_path), "error": repr(exc)})

    original_unique = len(pattern_counts)
    print(f"Unique mined patterns before augmentation: {original_unique}")

    if args.augment_transpositions:
        print("[4/6] Augmenting patterns by fret transposition...")
        pattern_counts = augment_patterns_by_transposition(
            pattern_counts=pattern_counts,
            bank=bank,
            max_tab_fret=args.max_tab_fret,
            max_render_fret=args.max_render_fret,
            max_shift_up=args.max_transpose_up,
            max_shift_down=args.max_transpose_down,
        )

    ranked = pattern_counts.most_common()
    total_unique = len(ranked)
    print(f"Unique patterns available after augmentation: {total_unique}")
    if total_unique < args.target_patterns:
        print(f"[warn] Requested {args.target_patterns} patterns, but only {total_unique} are available.")

    selected = ranked[:args.target_patterns]

    print(f"[5/6] Rendering {len(selected)} pattern .npz files...")
    tempo = float(args.tempo)
    note_dur = 60.0 / tempo / float(args.note_resolution) * 4.0
    len_notes = args.note_resolution * 4
    segment_sec = len_notes * note_dur
    onset_note_idx = int(args.onset_note_idx)
    sustain_note_steps = int(args.pattern_note_steps)
    onset_sec = onset_note_idx * note_dur
    sustain_sec = sustain_note_steps * note_dur
    frame_len = int(round(segment_sec * args.sr / args.hop_length))

    written = 0
    skipped = Counter()

    for i, (key, count) in enumerate(tqdm(selected)):
        try:
            pat = pattern_from_key(key)
        except Exception:
            skipped["bad_pattern_key"] += 1
            continue

        missing = [(gp_s, fret) for tab_s, gp_s, fret, midi in pat.notes if not sample_exists(bank, gp_s, fret)]
        if missing:
            skipped["missing_samples"] += 1
            continue

        labels = make_pattern_labels(
            pattern=pat,
            len_notes=len_notes,
            frame_len=frame_len,
            onset_note_idx=onset_note_idx,
            sustain_note_steps=sustain_note_steps,
            sr=args.sr,
            hop=args.hop_length,
            tempo=tempo,
            note_resolution=args.note_resolution,
        )
        audio = render_pattern_audio(
            pattern=pat,
            bank=bank,
            sr=args.sr,
            segment_sec=segment_sec,
            onset_sec=onset_sec,
            sustain_sec=sustain_sec,
            rng=rng,
            tail_sec=args.sample_tail_sec,
        )
        feats = compute_features(audio, args.sr, args.hop_length, args.cqt_n_bins, args.bins_per_octave, frame_len)

        fold = f"{i % args.n_folds:02d}"
        out_name = f"{fold}_pattern_{i:05d}_count{count}_{safe_name_text(key, 80)}.npz"
        out_path = out_dir / out_name

        payload = dict(
            cqt=feats["cqt"].astype(np.float32),
            log_cqt=feats["log_cqt"].astype(np.float32),
            mel_spec=feats["mel_spec"].astype(np.float32),
            tab=labels["tab"].astype(np.float32),
            tab_onset=labels["tab_onset"].astype(np.float32),
            frame_tab=labels["frame_tab"].astype(np.float32),
            frame_tab_onset=labels["frame_tab_onset"].astype(np.float32),
            F0=labels["F0"].astype(np.float32),
            F0_onset=labels["F0_onset"].astype(np.float32),
            frame_F0=labels["frame_F0"].astype(np.float32),
            frame_F0_onset=labels["frame_F0_onset"].astype(np.float32),
            tempo=np.array(tempo, dtype=np.float32),
            len_in_notes=np.array(len_notes, dtype=np.int32),
            pattern_key=np.array(key),
            pattern_count=np.array(count, dtype=np.int32),
            pattern_index=np.array(i, dtype=np.int32),
            source_type=np.array("dadagp_pattern"),
            onset_note_idx=np.array(onset_note_idx, dtype=np.int32),
            pattern_note_steps=np.array(sustain_note_steps, dtype=np.int32),
        )

        if args.hand_mode == "phantom":
            phantom_hand, phantom_conf = hand_from_tab(labels["tab"], args.hand_frets, rng)
            frame_phantom_hand, frame_phantom_conf = hand_from_tab(labels["frame_tab"], args.hand_frets, rng)
            payload.update(
                phantom_hand=phantom_hand.astype(np.float32),
                frame_phantom_hand=frame_phantom_hand.astype(np.float32),
                phantom_hand_conf=phantom_conf.astype(np.float32),
                frame_phantom_hand_conf=frame_phantom_conf.astype(np.float32),
            )

        atomic_savez_compressed(out_path, **payload)

        if args.write_mp3:
            mp3_dir = Path(args.mp3_dir) if args.mp3_dir else (parent / "mp3_debug")
            mp3_path = mp3_dir / out_path.with_suffix(".mp3").name
            try:
                write_low_quality_mp3(audio, args.sr, mp3_path, args.mp3_bitrate)
            except Exception as exc:
                skipped["mp3_write_failed"] += 1
                if args.fail_on_mp3_error:
                    raise
                print(f"[warn] MP3 write failed for {mp3_path}: {exc}")

        written += 1

    print("[6/6] Saving logs...")
    with (parent / "pattern_counts.json").open("w", encoding="utf-8") as f:
        json.dump(pattern_counts.most_common(), f, indent=2)
    with (parent / "failed_files.json").open("w", encoding="utf-8") as f:
        json.dump(failed, f, indent=2)
    stats = dict(
        target_patterns=args.target_patterns,
        original_unique_patterns=original_unique,
        total_unique_patterns_after_augmentation=total_unique,
        rendered_npz_files=written,
        skipped=dict(skipped),
        selected_gp_files=len(gp_files),
        output_dir=str(out_dir),
        hand_mode=args.hand_mode,
        args=vars(args),
    )
    with (parent / "dataset_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print("Done.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mine DadaGP fingering/chord patterns and render Tab-Estimator-style .npz files.")
    p.add_argument("--dadagp-root", required=True)
    p.add_argument("--sample-root", required=True)
    p.add_argument("--onset-root", default=None)
    p.add_argument("--out-dir", default="data/npz/dadagp_patterns/split")
    p.add_argument("--data-percentage", type=float, default=0.05)
    p.add_argument("--target-patterns", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-folds", type=int, default=6)
    p.add_argument("--guitar-prefix", default="fender")
    p.add_argument("--n-instances", type=int, default=1)
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--hop-length", type=int, default=512)
    p.add_argument("--cqt-n-bins", type=int, default=192)
    p.add_argument("--bins-per-octave", type=int, default=24)
    p.add_argument("--note-resolution", type=int, default=16)
    p.add_argument("--tempo", type=float, default=120.0)
    p.add_argument("--max-tab-fret", type=int, default=19)
    p.add_argument("--max-render-fret", type=int, default=19)
    p.add_argument("--hand-frets", type=int, default=20)
    p.add_argument("--min-notes-per-pattern", type=int, default=1)
    p.add_argument("--max-notes-per-pattern", type=int, default=6)
    p.add_argument("--max-fret-span", type=int, default=5)
    p.add_argument("--skip-wide-fingering-patterns", action="store_true")
    p.add_argument("--allow-patterns-without-samples", action="store_true")
    p.add_argument("--augment-transpositions", action="store_true", default=True)
    p.add_argument("--no-augment-transpositions", dest="augment_transpositions", action="store_false")
    p.add_argument("--max-transpose-up", type=int, default=12)
    p.add_argument("--max-transpose-down", type=int, default=12)
    p.add_argument("--onset-note-idx", type=int, default=0)
    p.add_argument("--pattern-note-steps", type=int, default=16)
    p.add_argument("--sample-tail-sec", type=float, default=0.25)
    p.add_argument("--hand-mode", choices=["none", "phantom"], default="none")
    p.add_argument("--write-mp3", action="store_true")
    p.add_argument("--mp3-dir", default=None)
    p.add_argument("--mp3-bitrate", default="32k")
    p.add_argument("--fail-on-mp3-error", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
