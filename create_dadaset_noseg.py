#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_dadaset_noseg_v2.py

Create a GuitarSet-like synthetic dataset called DadaSet from DadaGP,
WITHOUT creating NPZ files, WITHOUT custom pattern examples, and WITHOUT
splitting tracks into segments.

Output:
    DadaSet/
        annotation/
            00_piece-120-DadaSet_track001_Guitar_2.jams
            ...
        audio_mono-mic/
            00_piece-120-DadaSet_track001_Guitar_2_mic.wav
            ...

This mimics the two parts of GuitarSet that Tab-Estimator actually needs:

    GuitarSet/annotation/*.jams
    GuitarSet/audio_mono-mic/*_mic.wav

Main design:
    one selected DadaGP guitar track -> one full .jams + one full _mic.wav

This script does NOT:
    - create .npz files directly
    - create custom pattern examples
    - split tracks into 4-bar segments

Important for Tab-Estimator:
    Its original scripts are hardcoded to GuitarSet unless you patched them.
    If you patched them, use:
        python src/jams_to_midi.py --dataset-dir DadaSet
        python src/midi_to_numpy.py --dataset-dir DadaSet

Install:
    pip install numpy librosa soundfile jams pyguitarpro tqdm

Example:
    python create_dadaset_noseg_v2.py \
      --dadagp-root data/DadaGP-v1.1 \
      --sample-root data/fender_samples \
      --out-root DadaSet \
      --data-percentage 0.05 \
      --max-render-fret 19 \
      --max-tab-fret 19 \
      --min-notes-per-track 50 \
      --clean-output
"""

import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import jams
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

try:
    import guitarpro as gp
except ImportError as exc:
    raise SystemExit("Install PyGuitarPro with: pip install pyguitarpro") from exc


DEFAULT_TICKS_PER_QUARTER = 960

LOW_E_MIDI = 40
N_F0_CLASSES = 44

# Tab/GuitarSet order: low E -> high e.
OPEN_MIDI_LOW_TO_HIGH = np.array([40, 45, 50, 55, 59, 64], dtype=np.int32)
STRING_NAMES_LOW_TO_HIGH = [
    "E string",
    "A string",
    "D string",
    "G string",
    "B string",
    "e string",
]


@dataclass
class NoteEvent:
    start_sec: float
    end_sec: float
    tab_string: int     # 0=low E, 5=high e
    gp_string: int      # 1=high e, 6=low E
    fret: int
    midi: int
    velocity: float
    source_file: str
    track_name: str


@dataclass
class NoteSample:
    audio: np.ndarray
    sr: int
    gp_string: int
    fret: int
    source_path: str


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------

def safe_text(x, max_len: int = 80) -> str:
    if isinstance(x, Path):
        s = x.stem
    else:
        s = str(x)

    # Remove "-" so the first dash in filename is before tempo.
    # Tab-Estimator often parses tempo using basename.split("-")[1].
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", s).strip("_")
    return (s or "item")[:max_len]


def parse_keyword_list(text: str) -> List[str]:
    if text is None:
        return []
    text = text.strip()
    if not text:
        return []
    return [x.strip().lower() for x in text.split(",") if x.strip()]


def gp_string_to_tab_index(gp_string: int) -> int:
    """PyGuitarPro: 1=high e..6=low E. Tab/GuitarSet: 0=low E..5=high e."""
    return 6 - int(gp_string)


def midi_from_tab_string_fret(tab_string: int, fret: int) -> int:
    return int(OPEN_MIDI_LOW_TO_HIGH[tab_string] + fret)


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
        # Heuristic: large values are sample indices; small values are seconds.
        return val / sr if val > 5.0 else max(0.0, val)
    except Exception:
        return 0.0


def atomic_write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    """
    Write WAV atomically.

    The temporary file ends with .wav because soundfile uses the extension
    unless format='WAV' is passed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(path.stem + ".tmp.wav")

    try:
        sf.write(
            str(tmp),
            audio.astype(np.float32),
            sr,
            format="WAV",
            subtype="PCM_16",
        )
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


def atomic_jams_save(jam: jams.JAMS, path: Path) -> None:
    """
    Write JAMS atomically and verify it can be loaded after saving.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(path.stem + ".tmp.jams")

    try:
        jam.save(str(tmp))
        _ = jams.load(str(tmp))     # verify temp file
        os.replace(tmp, path)
        _ = jams.load(str(path))    # verify final file
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


# ---------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------

def ticks_per_quarter() -> int:
    try:
        return int(gp.models.Duration.quarterTime)
    except Exception:
        return DEFAULT_TICKS_PER_QUARTER


def get_song_tempo(song) -> float:
    tempo = getattr(song, "tempo", 120)
    try:
        return float(getattr(tempo, "value"))
    except Exception:
        try:
            return float(tempo)
        except Exception:
            return 120.0


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

    value = int(getattr(dur, "value", 4) or 4)
    base = int(ticks_per_quarter() * 4 / value)

    if bool(getattr(dur, "isDotted", False)):
        base = int(round(base * 1.5))

    tuplet = getattr(dur, "tuplet", None)
    if tuplet is not None:
        enters = getattr(tuplet, "enters", 1) or 1
        times = getattr(tuplet, "times", 1) or 1
        base = int(round(base * times / enters))

    return max(1, base)


# ---------------------------------------------------------------------
# Sample bank
# ---------------------------------------------------------------------

def load_note_samples(
    sample_root: Path,
    onset_root: Optional[Path],
    sr: int,
    max_render_fret: int,
    guitar_prefix: str = "fender",
    n_instances: int = 1,
) -> Dict[Tuple[int, int], List[NoteSample]]:
    """
    Supports:

    Flat layout:
        sample_root/string1/0.wav
        sample_root/string1/0.txt

    Multi-instance layout:
        sample_root/fender1/string1/0.wav
        sample_root/fender1/string1/0.txt
    """
    bank: Dict[Tuple[int, int], List[NoteSample]] = defaultdict(list)

    sample_root = Path(sample_root)
    onset_root = Path(onset_root) if onset_root else sample_root

    flat_layout = all((sample_root / f"string{s}").exists() for s in range(1, 7))
    sources = []

    if flat_layout:
        print("[info] Detected flat sample layout: string1..string6 under sample root.")
        for gp_string in range(1, 7):
            string_dir = sample_root / f"string{gp_string}"
            candidate_onset_dir = onset_root / f"string{gp_string}"
            onset_string_dir = candidate_onset_dir if candidate_onset_dir.exists() else string_dir
            sources.append((gp_string, string_dir, onset_string_dir))
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
                    sources.append((gp_string, string_dir, onset_string_dir))

    for gp_string, string_dir, onset_string_dir in sources:
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
                    try:
                        y = librosa.resample(
                            y.astype(np.float32),
                            orig_sr=src_sr,
                            target_sr=sr,
                        )
                    except TypeError:
                        # librosa 0.8.x fallback
                        y = librosa.resample(y.astype(np.float32), src_sr, sr)

                y = normalize_audio(y, peak=1.0)

                bank[(gp_string, fret)].append(
                    NoteSample(
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


def sample_exists(bank: Dict[Tuple[int, int], List[NoteSample]], gp_string: int, fret: int) -> bool:
    return len(bank.get((gp_string, fret), [])) > 0


def choose_sample(
    bank: Dict[Tuple[int, int], List[NoteSample]],
    gp_string: int,
    fret: int,
    rng,
) -> Optional[np.ndarray]:
    candidates = bank.get((gp_string, fret), [])
    if not candidates:
        return None
    return candidates[int(rng.integers(0, len(candidates)))].audio


# ---------------------------------------------------------------------
# DadaGP parsing and filtering
# ---------------------------------------------------------------------

def collect_gp_files(root: Path, percentage: float, seed: int) -> List[Path]:
    exts = {".gp", ".gp3", ".gp4", ".gp5"}
    files = sorted(
        [
            p for p in Path(root).rglob("*")
            if p.is_file() and p.suffix.lower() in exts
        ]
    )

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


def should_exclude_track_name(track_name: str, excluded_keywords: Sequence[str]) -> bool:
    name = str(track_name).lower().replace("_", " ")
    for kw in excluded_keywords:
        kw = kw.lower().strip()
        if kw and kw in name:
            return True
    return False


def parse_guitarpro_file(
    gp_path: Path,
    max_render_fret: int,
    max_tab_fret: int,
    bank: Dict[Tuple[int, int], List[NoteSample]],
    excluded_keywords: Sequence[str],
) -> Tuple[float, Dict[str, List[NoteEvent]], Counter]:
    """
    Parse one Guitar Pro file.

    Note:
        We do not apply min_notes here. We parse first, then build()
        performs final filtering on the effective event count before writing.
    """
    song = gp.parse(str(gp_path))
    tempo = get_song_tempo(song)
    sec_per_tick = (60.0 / tempo) / float(ticks_per_quarter())

    tracks_events: Dict[str, List[NoteEvent]] = {}
    skipped = Counter()

    for track_idx, track in enumerate(getattr(song, "tracks", [])):
        track_name = getattr(track, "name", f"track{track_idx}")

        if not is_probably_guitar_track(track):
            skipped["non_guitar_track"] += 1
            continue

        if should_exclude_track_name(track_name, excluded_keywords):
            skipped["excluded_track_name"] += 1
            continue

        safe_track_name = safe_text(track_name, max_len=40)
        track_key = f"track{track_idx:03d}_{safe_track_name}"

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

                        if gp_string < 1 or gp_string > 6:
                            skipped["invalid_string"] += 1
                            continue
                        if fret < 0:
                            skipped["invalid_fret"] += 1
                            continue
                        if fret > max_tab_fret:
                            skipped["fret_above_max_tab_fret"] += 1
                            continue
                        if fret > max_render_fret:
                            skipped["fret_above_max_render_fret"] += 1
                            continue
                        if not sample_exists(bank, gp_string, fret):
                            skipped["missing_sample_for_note"] += 1
                            continue

                        tab_string = gp_string_to_tab_index(gp_string)
                        midi = midi_from_tab_string_fret(tab_string, fret)

                        if midi < LOW_E_MIDI or midi >= LOW_E_MIDI + N_F0_CLASSES:
                            skipped["midi_out_of_range"] += 1
                            continue

                        velocity = float(getattr(note, "velocity", 95)) / 95.0
                        velocity = max(0.1, min(1.2, velocity))

                        start_tick = int(beat_start)
                        end_tick = int(
                            beat_start
                            + dur_ticks * float(getattr(note, "durationPercent", 1.0))
                        )

                        if end_tick <= start_tick:
                            end_tick = start_tick + 1

                        min_tick = start_tick if min_tick is None else min(min_tick, start_tick)
                        raw.append((start_tick, end_tick, tab_string, gp_string, fret, midi, velocity))

        if min_tick is None or len(raw) == 0:
            skipped["track_no_valid_notes"] += 1
            continue

        events: List[NoteEvent] = []

        for st, en, tab_s, gp_s, fret, midi, vel in raw:
            start_sec = (st - min_tick) * sec_per_tick
            end_sec = max(start_sec + 0.03, (en - min_tick) * sec_per_tick)

            events.append(
                NoteEvent(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    tab_string=tab_s,
                    gp_string=gp_s,
                    fret=fret,
                    midi=midi,
                    velocity=vel,
                    source_file=str(gp_path),
                    track_name=str(track_name),
                )
            )

        events.sort(key=lambda e: (e.start_sec, e.tab_string, e.fret))
        tracks_events[track_key] = events

    return tempo, tracks_events, skipped


# ---------------------------------------------------------------------
# Rendering and JAMS creation
# ---------------------------------------------------------------------

def render_events_audio(
    events: Sequence[NoteEvent],
    duration: float,
    bank: Dict[Tuple[int, int], List[NoteSample]],
    sr: int,
    rng,
    tail_sec: float,
) -> np.ndarray:
    n = max(1, int(math.ceil(duration * sr)))
    audio = np.zeros(n, dtype=np.float32)

    for ev in events:
        if ev.end_sec <= 0 or ev.start_sec >= duration:
            continue

        insert_at = int(round(max(0.0, ev.start_sec) * sr))

        desired_len = int(
            round(
                (
                    min(duration, ev.end_sec)
                    - max(0.0, ev.start_sec)
                    + tail_sec
                )
                * sr
            )
        )
        desired_len = max(1, desired_len)

        sample = choose_sample(bank, ev.gp_string, ev.fret, rng)
        if sample is None:
            continue

        sample = ensure_len(sample, desired_len)
        sample = sample * ev.velocity * float(rng.uniform(0.75, 1.05))

        jitter = int(rng.normal(0, 0.004 * sr))
        insert_at = max(0, insert_at + jitter)

        if insert_at >= n:
            continue

        sample = sample[:n - insert_at]
        sample = fade_edges(sample, min(128, sample.size // 8))
        audio[insert_at:insert_at + sample.size] += sample

    return normalize_audio(audio, peak=0.95)


def make_jams_for_track(
    events: Sequence[NoteEvent],
    duration: float,
    tempo: float,
    title: str,
    source_gp: str,
    track_name: str,
) -> jams.JAMS:
    """
    Create a minimal GuitarSet-like JAMS file for Tab-Estimator.

    Important:
    - Tab-Estimator only needs note_midi annotations.
    - We create exactly six note_midi annotations.
    - Their order is low-to-high:
        E, A, D, G, B, e
    - No beat_position is added.
    - No pitch_contour is added.
    """
    jam = jams.JAMS()
    jam.file_metadata.title = title
    jam.file_metadata.duration = float(duration)

    jam.sandbox.dadaset = {
        "source_kind": "DadaGP full track rendered with note samples",
        "source_gp": str(source_gp),
        "track_name": str(track_name),
        "tempo": float(tempo),
    }

    anns = []

    for string_idx, string_name in enumerate(STRING_NAMES_LOW_TO_HIGH):
        ann = jams.Annotation(namespace="note_midi")

        # Make annotation closer to GuitarSet.
        ann.time = 0.0
        ann.duration = float(duration)
        ann.annotation_metadata = jams.AnnotationMetadata()
        ann.annotation_metadata.data_source = str(string_idx)

        ann.sandbox.string_index = int(string_idx)
        ann.sandbox.string_name = str(string_name)
        ann.sandbox.string_order = "low_to_high"

        anns.append(ann)

    for ev in events:
        st = max(0.0, float(ev.start_sec))
        en = min(float(duration), float(ev.end_sec))

        if en <= st:
            continue

        anns[ev.tab_string].append(
            time=st,
            duration=max(0.03, en - st),
            value=float(ev.midi),
            confidence=None,
        )

    for ann in anns:
        jam.annotations.append(ann)

    return jam


def validate_jams_note_midi_count(jams_path: Path) -> int:
    jam = jams.load(str(jams_path))
    annos = jam.search(namespace="note_midi")
    if len(annos) != 6:
        raise RuntimeError(f"Expected 6 note_midi annotations, got {len(annos)} in {jams_path}")
    return sum(len(ann.data) for ann in annos)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def build(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)

    out_root = Path(args.out_root)
    ann_dir = out_root / "annotation"
    audio_dir = out_root / "audio_mono-mic"

    if args.clean_output:
        import shutil
        if ann_dir.exists():
            shutil.rmtree(ann_dir)
        if audio_dir.exists():
            shutil.rmtree(audio_dir)

    ann_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    excluded_keywords = parse_keyword_list(args.exclude_track_keywords)

    print("[1/5] Loading note samples...")
    bank = load_note_samples(
        sample_root=Path(args.sample_root),
        onset_root=Path(args.onset_root) if args.onset_root else None,
        sr=args.sr,
        max_render_fret=args.max_render_fret,
        guitar_prefix=args.guitar_prefix,
        n_instances=args.n_instances,
    )

    if not bank:
        raise SystemExit("No note samples loaded. Check --sample-root.")

    print(f"Loaded {sum(len(v) for v in bank.values())} note instances across {len(bank)} string/fret keys.")
    print("Excluded track keywords:", excluded_keywords if excluded_keywords else "(none)")

    print("[2/5] Selecting DadaGP files...")
    gp_files = collect_gp_files(Path(args.dadagp_root), args.data_percentage, args.seed)
    print(f"Selected {len(gp_files)} files.")

    failed = []
    skipped = Counter()
    written_tracks = 0
    written_notes = 0

    print("[3/5] Rendering full tracks with no segmentation...")
    for file_idx, gp_path in enumerate(tqdm(gp_files)):
        fold = f"{file_idx % args.n_folds:02d}"

        try:
            tempo, tracks, local_skipped = parse_guitarpro_file(
                gp_path=gp_path,
                max_render_fret=args.max_render_fret,
                max_tab_fret=args.max_tab_fret,
                bank=bank,
                excluded_keywords=excluded_keywords,
            )
            skipped.update(local_skipped)

        except Exception as exc:
            failed.append({"file": str(gp_path), "error": repr(exc)})
            skipped["parse_error"] += 1
            continue

        if tempo < args.min_tempo or tempo > args.max_tempo:
            skipped["tempo_out_of_range"] += 1
            continue

        piece_name = safe_text(gp_path)
        tempo_for_name = int(round(tempo))

        for track_key, events in tracks.items():
            if not events:
                skipped["empty_track"] += 1
                continue

            # Final effective note-count filter.
            if len(events) < args.min_notes_per_track:
                skipped["too_few_valid_events_after_filtering"] += 1
                continue

            duration = max(e.end_sec for e in events) + args.sample_tail_sec

            if args.max_track_duration_sec > 0 and duration > args.max_track_duration_sec:
                skipped["track_too_long"] += 1
                continue

            if duration <= 0:
                skipped["invalid_duration"] += 1
                continue

            # First dash is before tempo for Tab-Estimator's filename parsing.
            base = f"{fold}_{piece_name}-{tempo_for_name}-DadaSet_{track_key}"

            wav_path = audio_dir / f"{base}_mic.wav"
            jams_path = ann_dir / f"{base}.jams"

            audio = render_events_audio(
                events=events,
                duration=duration,
                bank=bank,
                sr=args.sr,
                rng=rng,
                tail_sec=args.sample_tail_sec,
            )

            if np.max(np.abs(audio)) < args.min_audio_peak:
                skipped["audio_too_quiet"] += 1
                continue

            jam = make_jams_for_track(
                events=events,
                duration=duration,
                tempo=tempo,
                title=base,
                source_gp=str(gp_path),
                track_name=track_key,
            )

            try:
                # Write audio first, then JAMS.
                atomic_write_wav(wav_path, audio, args.sr)
                atomic_jams_save(jam, jams_path)

                if not wav_path.exists():
                    raise RuntimeError(f"WAV not created: {wav_path}")
                if not jams_path.exists():
                    raise RuntimeError(f"JAMS not created: {jams_path}")

                n_jams_notes = validate_jams_note_midi_count(jams_path)
                if n_jams_notes < args.min_notes_per_track:
                    raise RuntimeError(
                        f"JAMS has too few note_midi events after save: {n_jams_notes}"
                    )

            except Exception as exc:
                failed.append(
                    {
                        "file": str(gp_path),
                        "item": base,
                        "event_count": len(events),
                        "wav_path": str(wav_path),
                        "jams_path": str(jams_path),
                        "error": repr(exc),
                    }
                )
                skipped["write_error"] += 1

                # Remove half-paired output if one side was written.
                try:
                    if wav_path.exists():
                        wav_path.unlink()
                    if jams_path.exists():
                        jams_path.unlink()
                except Exception:
                    pass

                continue

            written_tracks += 1
            written_notes += len(events)

            if args.max_tracks is not None and written_tracks >= args.max_tracks:
                break

        if args.max_tracks is not None and written_tracks >= args.max_tracks:
            break

    print("[4/5] Writing logs...")
    stats = {
        "out_root": str(out_root),
        "annotation_dir": str(ann_dir),
        "audio_dir": str(audio_dir),
        "selected_gp_files": len(gp_files),
        "written_full_tracks": written_tracks,
        "written_note_events_before_jams_conversion": written_notes,
        "failed_items": len(failed),
        "skipped": dict(skipped),
        "args": vars(args),
    }

    with (out_root / "dadaset_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    with (out_root / "dadaset_failed.json").open("w", encoding="utf-8") as f:
        json.dump(failed, f, indent=2)

    print("[5/5] Done.")
    print(json.dumps(stats, indent=2))

    print("\nCreated:")
    print(f"  JAMS: {ann_dir}")
    print(f"  WAV:  {audio_dir}")

    print("\nCheck:")
    print(f"  find {ann_dir} -name '*.jams' | wc -l")
    print(f"  find {audio_dir} -name '*_mic.wav' | wc -l")

    print("\nUse with your patched Tab-Estimator scripts:")
    print("  cd Tab-estimator")
    print(f"  python src/jams_to_midi.py --dataset-dir {out_root}")
    print(f"  python src/midi_to_numpy.py --dataset-dir {out_root}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create no-segmentation DadaSet JAMS/WAV dataset from DadaGP."
    )

    p.add_argument("--dadagp-root", required=True)
    p.add_argument("--sample-root", required=True)
    p.add_argument("--onset-root", default=None)
    p.add_argument("--out-root", default="DadaSet")

    p.add_argument("--data-percentage", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-folds", type=int, default=6)

    p.add_argument("--guitar-prefix", default="fender")
    p.add_argument("--n-instances", type=int, default=1)

    p.add_argument("--sr", type=int, default=22050)

    p.add_argument("--max-tab-fret", type=int, default=19)
    p.add_argument("--max-render-fret", type=int, default=19)

    # Default 50 avoids weak examples like 3-note/5-note tracks.
    p.add_argument("--min-notes-per-track", type=int, default=50)
    p.add_argument("--max-tracks", type=int, default=None)

    p.add_argument(
        "--max-track-duration-sec",
        type=float,
        default=0.0,
        help="0 means no limit. If >0, skip tracks longer than this; never segment.",
    )

    p.add_argument("--min-tempo", type=float, default=40.0)
    p.add_argument("--max-tempo", type=float, default=240.0)
    p.add_argument("--sample-tail-sec", type=float, default=0.25)

    p.add_argument(
        "--exclude-track-keywords",
        default="vocal,backvocal,voice,melody,bass,drum,perc,keyboard,piano,synth",
        help=(
            "Comma-separated track-name keywords to skip. "
            "Use empty string \"\" to disable."
        ),
    )

    p.add_argument(
        "--min-audio-peak",
        type=float,
        default=1e-5,
        help="Skip rendered tracks whose absolute peak is below this value.",
    )

    p.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete DadaSet/annotation and DadaSet/audio_mono-mic before writing.",
    )

    return p.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
