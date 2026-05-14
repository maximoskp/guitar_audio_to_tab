#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_dadaset_noseg_with_patterns.py

Create a GuitarSet-like synthetic dataset called DadaSet from DadaGP,
WITHOUT segmenting DadaGP tracks.

Output:
    DadaSet/
        annotation/
            00_piece-130-DadaSet_track000_Clean_Guitar.jams
            CP00000-120-DadaSet_pattern_s0f3_s1f2.jams
            ...
        audio_mono-mic/
            00_piece-130-DadaSet_track000_Clean_Guitar_mic.wav
            CP00000-120-DadaSet_pattern_s0f3_s1f2_mic.wav
            ...

This is designed to mimic the parts of GuitarSet that Tab-Estimator reads:
    GuitarSet/annotation/*.jams
    GuitarSet/audio_mono-mic/*_mic.wav

Important:
    Tab-Estimator's original scripts are hardcoded to "GuitarSet".
    After creating DadaSet, either patch those paths or symlink:

        cd Tab-estimator
        mv GuitarSet GuitarSet_real     # if it exists and you want to keep it
        ln -s ../DadaSet GuitarSet
        python src/jams_to_midi.py
        python src/midi_to_numpy.py

This script creates two types of examples:
    1. Full DadaGP tracks rendered as one whole JAMS/WAV pair.
       No 4-bar segmentation is done.

    2. Custom chord/fingering patterns mined from DadaGP beats, rendered as
       separate GuitarSet-like JAMS/WAV pairs. These are standalone synthetic
       examples, not source-track segments.

Filenames are intentionally formatted with the first dash before tempo, because
Tab-Estimator's src/jams_to_midi.py does:
    tempo = float(jams_filename.split('-')[1])

Install:
    pip install numpy librosa soundfile jams pyguitarpro tqdm

Example:
    python create_dadaset_noseg_with_patterns.py \
      --dadagp-root data/DadaGP-v1.1/DadaGP-v1.1 \
      --sample-root data/fender_samples \
      --out-root DadaSet \
      --data-percentage 0.05 \
      --max-render-fret 19 \
      --max-tab-fret 19 \
      --custom-patterns 1000
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

N_STRINGS = 6
LOW_E_MIDI = 40
N_F0_CLASSES = 44
OPEN_MIDI_LOW_TO_HIGH = np.array([40, 45, 50, 55, 59, 64], dtype=np.int32)
STRING_NAMES_LOW_TO_HIGH = ["E string", "A string", "D string", "G string", "B string", "e string"]


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


@dataclass(frozen=True)
class Pattern:
    notes: Tuple[Tuple[int, int, int, int], ...]
    # each note: (tab_string, gp_string, fret, midi)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def safe_text(x, max_len: int = 80) -> str:
    if isinstance(x, Path):
        s = x.stem
    else:
        s = str(x)
    # Remove '-' so the first dash in filename is before tempo.
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", s).strip("_")
    return (s or "item")[:max_len]


def safe_pattern_text(key: str, max_len: int = 90) -> str:
    s = key.replace("+", "_")
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", s).strip("_")
    return (s or "pattern")[:max_len]


def gp_string_to_tab_index(gp_string: int) -> int:
    return 6 - int(gp_string)


def tab_index_to_gp_string(tab_string: int) -> int:
    return 6 - int(tab_string)


def midi_from_tab_string_fret(tab_string: int, fret: int) -> int:
    return int(OPEN_MIDI_LOW_TO_HIGH[tab_string] + fret)


def pattern_key(pattern: Pattern) -> str:
    if not pattern.notes:
        return "REST"
    return "+".join(f"s{tab_s}f{fret}" for tab_s, gp_s, fret, midi in pattern.notes)


def pattern_from_key(key: str) -> Pattern:
    if key == "REST":
        return Pattern(tuple())
    notes = []
    for part in key.split("+"):
        m = re.match(r"s(\d+)f(\d+)$", part)
        if not m:
            raise ValueError(f"Bad pattern key: {key}")
        tab_s = int(m.group(1))
        fret = int(m.group(2))
        gp_s = tab_index_to_gp_string(tab_s)
        midi = midi_from_tab_string_fret(tab_s, fret)
        notes.append((tab_s, gp_s, fret, midi))
    return Pattern(tuple(sorted(notes, key=lambda x: x[0])))


def fretted_span(pattern: Pattern) -> int:
    frets = [fret for tab_s, gp_s, fret, midi in pattern.notes if fret > 0]
    if not frets:
        return 0
    return max(frets) - min(frets)


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


def atomic_write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Keep .wav as actual extension so soundfile knows the format.
    tmp = path.with_name(path.stem + ".tmp.wav")

    try:
        sf.write(str(tmp), audio.astype(np.float32), sr, format="WAV", subtype="PCM_16")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


def atomic_jams_save(jam: jams.JAMS, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Keep .jams as actual extension.
    tmp = path.with_name(path.stem + ".tmp.jams")

    try:
        jam.save(str(tmp))
        os.replace(tmp, path)
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
                        y = librosa.resample(y.astype(np.float32), orig_sr=src_sr, target_sr=sr)
                    except TypeError:
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


def choose_sample(bank: Dict[Tuple[int, int], List[NoteSample]], gp_string: int, fret: int, rng) -> Optional[np.ndarray]:
    candidates = bank.get((gp_string, fret), [])
    if not candidates:
        return None
    return candidates[int(rng.integers(0, len(candidates)))].audio


# ---------------------------------------------------------------------
# DadaGP parsing
# ---------------------------------------------------------------------

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


def parse_guitarpro_file(
    gp_path: Path,
    max_render_fret: int,
    max_tab_fret: int,
    min_notes: int,
    bank: Dict[Tuple[int, int], List[NoteSample]],
) -> Tuple[float, Dict[str, List[NoteEvent]]]:
    song = gp.parse(str(gp_path))
    tempo = get_song_tempo(song)
    sec_per_tick = (60.0 / tempo) / float(ticks_per_quarter())

    tracks_events: Dict[str, List[NoteEvent]] = {}

    for track_idx, track in enumerate(getattr(song, "tracks", [])):
        if not is_probably_guitar_track(track):
            continue

        track_name = getattr(track, "name", f"track{track_idx}")
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

                        if gp_string < 1 or gp_string > 6 or fret < 0:
                            continue
                        if fret > max_tab_fret or fret > max_render_fret:
                            continue
                        if not sample_exists(bank, gp_string, fret):
                            continue

                        tab_string = gp_string_to_tab_index(gp_string)
                        midi = midi_from_tab_string_fret(tab_string, fret)

                        if midi < LOW_E_MIDI or midi >= LOW_E_MIDI + N_F0_CLASSES:
                            continue

                        velocity = float(getattr(note, "velocity", 95)) / 95.0
                        velocity = max(0.1, min(1.2, velocity))

                        start_tick = int(beat_start)
                        end_tick = int(beat_start + dur_ticks * float(getattr(note, "durationPercent", 1.0)))
                        if end_tick <= start_tick:
                            end_tick = start_tick + 1

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


def beat_to_pattern(
    beat,
    max_tab_fret: int,
    max_render_fret: int,
    bank: Dict[Tuple[int, int], List[NoteSample]],
) -> Optional[Pattern]:
    by_string: Dict[int, Tuple[int, int, int, int]] = {}

    for note in getattr(beat, "notes", []):
        gp_string = int(getattr(note, "string", 0))
        fret = int(getattr(note, "value", -1))

        if gp_string < 1 or gp_string > 6 or fret < 0:
            continue
        if fret > max_tab_fret or fret > max_render_fret:
            continue
        if not sample_exists(bank, gp_string, fret):
            continue

        tab_string = gp_string_to_tab_index(gp_string)
        midi = midi_from_tab_string_fret(tab_string, fret)

        if midi < LOW_E_MIDI or midi >= LOW_E_MIDI + N_F0_CLASSES:
            continue

        by_string[tab_string] = (tab_string, gp_string, fret, midi)

    if not by_string:
        return None

    return Pattern(tuple(sorted(by_string.values(), key=lambda x: x[0])))


def mine_patterns_from_file(
    gp_path: Path,
    bank: Dict[Tuple[int, int], List[NoteSample]],
    max_tab_fret: int,
    max_render_fret: int,
) -> List[Pattern]:
    song = gp.parse(str(gp_path))
    patterns: List[Pattern] = []

    for track in getattr(song, "tracks", []):
        if not is_probably_guitar_track(track):
            continue
        for measure in getattr(track, "measures", []):
            for voice in getattr(measure, "voices", []):
                for beat in getattr(voice, "beats", []):
                    pat = beat_to_pattern(beat, max_tab_fret, max_render_fret, bank)
                    if pat is not None:
                        patterns.append(pat)

    return patterns


def augment_patterns_by_transposition(
    pattern_counts: Counter,
    bank: Dict[Tuple[int, int], List[NoteSample]],
    max_tab_fret: int,
    max_render_fret: int,
    max_shift_up: int,
    max_shift_down: int,
) -> Counter:
    out = Counter(pattern_counts)

    for key, count in list(pattern_counts.items()):
        pat = pattern_from_key(key)

        # Do not transpose open-string shapes.
        if any(fret == 0 for tab_s, gp_s, fret, midi in pat.notes):
            continue

        for shift in range(-max_shift_down, max_shift_up + 1):
            if shift == 0:
                continue

            new_notes = []
            ok = True

            for tab_s, gp_s, fret, midi in pat.notes:
                nf = fret + shift
                if nf < 1 or nf > max_tab_fret or nf > max_render_fret:
                    ok = False
                    break
                if not sample_exists(bank, gp_s, nf):
                    ok = False
                    break
                nmidi = midi_from_tab_string_fret(tab_s, nf)
                if nmidi < LOW_E_MIDI or nmidi >= LOW_E_MIDI + N_F0_CLASSES:
                    ok = False
                    break
                new_notes.append((tab_s, gp_s, nf, nmidi))

            if ok:
                new_pat = Pattern(tuple(sorted(new_notes, key=lambda x: x[0])))
                out[pattern_key(new_pat)] += max(1, count // 2)

    return out


# ---------------------------------------------------------------------
# Rendering and JAMS writing
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
        desired_len = int(round((min(duration, ev.end_sec) - max(0.0, ev.start_sec) + tail_sec) * sr))
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


def render_pattern_audio(
    pattern: Pattern,
    duration: float,
    onset_sec: float,
    sustain_sec: float,
    bank: Dict[Tuple[int, int], List[NoteSample]],
    sr: int,
    rng,
    tail_sec: float,
) -> np.ndarray:
    events = []
    for tab_s, gp_s, fret, midi in pattern.notes:
        events.append(
            NoteEvent(
                start_sec=onset_sec,
                end_sec=onset_sec + sustain_sec,
                tab_string=tab_s,
                gp_string=gp_s,
                fret=fret,
                midi=midi,
                velocity=1.0,
                source_file="custom_pattern",
                track_name="custom_pattern",
            )
        )
    return render_events_audio(events, duration, bank, sr, rng, tail_sec)


def make_jams(
    events: Sequence[NoteEvent],
    duration: float,
    tempo: float,
    title: str,
    source_kind: str,
    source_gp: str = "",
    track_name: str = "",
) -> jams.JAMS:
    jam = jams.JAMS()
    jam.file_metadata.title = title
    jam.file_metadata.duration = float(duration)

    jam.sandbox.dadaset = {
        "source_kind": source_kind,
        "source_gp": source_gp,
        "track_name": track_name,
        "tempo": float(tempo),
    }

    anns = []
    for string_idx, string_name in enumerate(STRING_NAMES_LOW_TO_HIGH):
        ann = jams.Annotation(namespace="note_midi")
        ann.annotation_metadata = jams.AnnotationMetadata(
            curator=jams.Curator(name="DadaSet generator"),
            data_source=source_kind,
        )
        ann.sandbox.string_index = int(string_idx)
        ann.sandbox.string_name = string_name
        anns.append(ann)

    for ev in events:
        st = max(0.0, ev.start_sec)
        en = min(float(duration), ev.end_sec)
        if en <= st:
            continue
        anns[ev.tab_string].append(
            time=float(st),
            duration=float(max(0.03, en - st)),
            value=float(ev.midi),
            confidence=1.0,
        )

    for ann in anns:
        jam.annotations.append(ann)

    beat_ann = jams.Annotation(namespace="beat_position")
    beat_ann.annotation_metadata = jams.AnnotationMetadata(
        curator=jams.Curator(name="DadaSet generator"),
        data_source="synthetic beat grid",
    )
    beat_dur = 60.0 / float(tempo)
    for b in range(int(math.ceil(duration / beat_dur)) + 1):
        t = b * beat_dur
        if t > duration:
            break
        beat_ann.append(
            time=float(t),
            duration=0.0,
            value={"position": int((b % 4) + 1)},
            confidence=1.0,
        )
    jam.annotations.append(beat_ann)

    return jam


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def build(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)

    out_root = Path(args.out_root)
    ann_dir = out_root / "annotation"
    audio_dir = out_root / "audio_mono-mic"
    ann_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading note samples...")
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

    print("[2/6] Selecting DadaGP files...")
    gp_files = collect_gp_files(Path(args.dadagp_root), args.data_percentage, args.seed)
    print(f"Selected {len(gp_files)} files.")

    failed = []
    skipped = Counter()
    written_tracks = 0
    written_patterns = 0

    pattern_counts = Counter()

    print("[3/6] Rendering full DadaGP tracks, no segmentation...")
    for file_idx, gp_path in enumerate(tqdm(gp_files)):
        fold = f"{file_idx % args.n_folds:02d}"

        try:
            tempo, tracks = parse_guitarpro_file(
                gp_path=gp_path,
                max_render_fret=args.max_render_fret,
                max_tab_fret=args.max_tab_fret,
                min_notes=args.min_notes_per_track,
                bank=bank,
            )
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

            duration = max(e.end_sec for e in events) + args.sample_tail_sec
            if args.max_track_duration_sec > 0 and duration > args.max_track_duration_sec:
                skipped["track_too_long"] += 1
                continue

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

            jam = make_jams(
                events=events,
                duration=duration,
                tempo=tempo,
                title=base,
                source_kind="DadaGP full track rendered with note samples",
                source_gp=str(gp_path),
                track_name=track_key,
            )

            try:
                atomic_write_wav(wav_path, audio, args.sr)
                atomic_jams_save(jam, jams_path)
            except Exception as exc:
                failed.append({"file": str(gp_path), "item": base, "error": repr(exc)})
                skipped["write_error"] += 1
                continue

            written_tracks += 1

            if args.max_tracks is not None and written_tracks >= args.max_tracks:
                break

        if args.max_tracks is not None and written_tracks >= args.max_tracks:
            break

    print("[4/6] Mining custom chord/fingering patterns...")
    if args.custom_patterns > 0:
        for gp_path in tqdm(gp_files):
            try:
                pats = mine_patterns_from_file(
                    gp_path=gp_path,
                    bank=bank,
                    max_tab_fret=args.max_tab_fret,
                    max_render_fret=args.max_render_fret,
                )
                for pat in pats:
                    if len(pat.notes) < args.min_notes_per_pattern:
                        continue
                    if len(pat.notes) > args.max_notes_per_pattern:
                        continue
                    if args.skip_wide_fingering_patterns and fretted_span(pat) > args.max_fret_span:
                        continue
                    pattern_counts[pattern_key(pat)] += 1
            except Exception as exc:
                failed.append({"file": str(gp_path), "pattern_mining_error": repr(exc)})
                skipped["pattern_mining_error"] += 1

        unique_before_aug = len(pattern_counts)

        if args.augment_transpositions:
            pattern_counts = augment_patterns_by_transposition(
                pattern_counts=pattern_counts,
                bank=bank,
                max_tab_fret=args.max_tab_fret,
                max_render_fret=args.max_render_fret,
                max_shift_up=args.max_transpose_up,
                max_shift_down=args.max_transpose_down,
            )

        ranked_patterns = pattern_counts.most_common(args.custom_patterns)
    else:
        unique_before_aug = 0
        ranked_patterns = []

    print(f"Unique mined patterns before augmentation: {unique_before_aug}")
    print(f"Patterns available after augmentation: {len(pattern_counts)}")
    print(f"Rendering custom patterns requested/selected: {args.custom_patterns}/{len(ranked_patterns)}")

    print("[5/6] Rendering custom patterns the same way as DadaSet items...")
    tempo = float(args.pattern_tempo)
    tempo_for_name = int(round(tempo))
    beat_sec = 60.0 / tempo
    pattern_duration = args.pattern_bars * 4 * beat_sec
    onset_sec = args.pattern_onset_beats * beat_sec
    sustain_sec = args.pattern_sustain_beats * beat_sec

    for idx, (key, count) in enumerate(tqdm(ranked_patterns)):
        try:
            pat = pattern_from_key(key)
        except Exception:
            skipped["bad_pattern_key"] += 1
            continue

        missing = [(gp_s, fret) for tab_s, gp_s, fret, midi in pat.notes if not sample_exists(bank, gp_s, fret)]
        if missing:
            skipped["pattern_missing_samples"] += 1
            continue

        events = []
        for tab_s, gp_s, fret, midi in pat.notes:
            events.append(
                NoteEvent(
                    start_sec=onset_sec,
                    end_sec=onset_sec + sustain_sec,
                    tab_string=tab_s,
                    gp_string=gp_s,
                    fret=fret,
                    midi=midi,
                    velocity=1.0,
                    source_file="custom_pattern",
                    track_name="custom_pattern",
                )
            )

        pat_text = safe_pattern_text(key, 90)
        base = f"CP{idx:05d}-{tempo_for_name}-DadaSet_pattern_{pat_text}"

        wav_path = audio_dir / f"{base}_mic.wav"
        jams_path = ann_dir / f"{base}.jams"

        audio = render_pattern_audio(
            pattern=pat,
            duration=pattern_duration,
            onset_sec=onset_sec,
            sustain_sec=sustain_sec,
            bank=bank,
            sr=args.sr,
            rng=rng,
            tail_sec=args.sample_tail_sec,
        )

        jam = make_jams(
            events=events,
            duration=pattern_duration,
            tempo=tempo,
            title=base,
            source_kind="DadaGP custom pattern rendered with note samples",
            source_gp="custom_pattern",
            track_name=key,
        )
        jam.sandbox.dadaset["pattern_key"] = key
        jam.sandbox.dadaset["pattern_count"] = int(count)

        try:
            atomic_write_wav(wav_path, audio, args.sr)
            atomic_jams_save(jam, jams_path)
        except Exception as exc:
            failed.append({"pattern": key, "item": base, "error": repr(exc)})
            skipped["pattern_write_error"] += 1
            continue

        written_patterns += 1

    print("[6/6] Writing logs...")
    stats = {
        "out_root": str(out_root),
        "annotation_dir": str(ann_dir),
        "audio_dir": str(audio_dir),
        "selected_gp_files": len(gp_files),
        "written_full_tracks": written_tracks,
        "written_custom_patterns": written_patterns,
        "custom_patterns_requested": args.custom_patterns,
        "unique_patterns_before_augmentation": unique_before_aug,
        "unique_patterns_after_augmentation": len(pattern_counts),
        "failed_items": len(failed),
        "skipped": dict(skipped),
        "args": vars(args),
    }

    with (out_root / "dadaset_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    with (out_root / "dadaset_failed.json").open("w", encoding="utf-8") as f:
        json.dump(failed, f, indent=2)

    with (out_root / "custom_pattern_counts.json").open("w", encoding="utf-8") as f:
        json.dump(pattern_counts.most_common(), f, indent=2)

    print(json.dumps(stats, indent=2))
    print("\nNext steps:")
    print("  cd Tab-estimator")
    print("  mv GuitarSet GuitarSet_real  # if needed")
    print(f"  ln -s {out_root.resolve()} GuitarSet")
    print("  python src/jams_to_midi.py")
    print("  python src/midi_to_numpy.py")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create no-segmentation DadaSet plus custom pattern JAMS/WAV examples."
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

    p.add_argument("--min-notes-per-track", type=int, default=8)
    p.add_argument("--max-tracks", type=int, default=None)
    p.add_argument(
        "--max-track-duration-sec",
        type=float,
        default=0.0,
        help="0 means no limit. If >0, skip tracks longer than this; never segment them.",
    )

    p.add_argument("--min-tempo", type=float, default=40.0)
    p.add_argument("--max-tempo", type=float, default=240.0)
    p.add_argument("--sample-tail-sec", type=float, default=0.25)

    # Custom pattern rendering.
    p.add_argument("--custom-patterns", type=int, default=1000)
    p.add_argument("--pattern-tempo", type=float, default=120.0)
    p.add_argument("--pattern-bars", type=int, default=4)
    p.add_argument("--pattern-onset-beats", type=float, default=0.0)
    p.add_argument("--pattern-sustain-beats", type=float, default=4.0)

    p.add_argument("--min-notes-per-pattern", type=int, default=1)
    p.add_argument("--max-notes-per-pattern", type=int, default=6)
    p.add_argument("--max-fret-span", type=int, default=5)
    p.add_argument("--skip-wide-fingering-patterns", action="store_true")

    p.add_argument("--augment-transpositions", action="store_true", default=True)
    p.add_argument("--no-augment-transpositions", dest="augment_transpositions", action="store_false")
    p.add_argument("--max-transpose-up", type=int, default=12)
    p.add_argument("--max-transpose-down", type=int, default=12)

    return p.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
