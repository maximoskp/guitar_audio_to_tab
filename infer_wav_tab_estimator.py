#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_dadaset_from_dadagp.py

Create a GuitarSet-like synthetic dataset called DadaSet from DadaGP.

Output structure:
    DadaSet/
        annotation/
            00_piece-120-DadaSet_t00_seg0000.jams
            ...
        audio_mono-mic/
            00_piece-120-DadaSet_t00_seg0000_mic.wav
            ...

This is designed to mimic the parts of GuitarSet that Tab-Estimator reads:

    GuitarSet/annotation/*.jams
    GuitarSet/audio_mono-mic/*_mic.wav

The generated JAMS contain six note_midi annotations, one per string,
in Tab-Estimator/GuitarSet order:

    0: E string  low E
    1: A string
    2: D string
    3: G string
    4: B string
    5: e string  high e

Important:
    Tab-Estimator's original src/jams_to_midi.py is hardcoded to read:
        GuitarSet/annotation/*.jams

    After creating DadaSet, either:
        1. temporarily symlink DadaSet as GuitarSet, or
        2. patch jams_to_midi.py to read DadaSet/annotation.

Recommended symlink approach:
    cd Tab-estimator
    mv GuitarSet GuitarSet_real   # if you already have one
    ln -s ../DadaSet GuitarSet

Then:
    python src/jams_to_midi.py
    python src/midi_to_numpy.py

But midi_to_numpy.py expects:
    GuitarSet/audio_mono-mic/<basename>_mic.wav

This script writes exactly that naming pattern.

Sample layout support:
    New flat layout:
        sample_root/string1/0.wav
        sample_root/string1/0.txt
        sample_root/string2/0.wav
        ...

    Old multi-instance layout:
        sample_root/fender1/string1/0.wav
        sample_root/fender2/string1/0.wav
        ...

Install:
    pip install numpy librosa soundfile jams pyguitarpro tqdm

Example:
    python create_dadaset_from_dadagp.py \
      --dadagp-root data/DadaGP-v1.1/DadaGP-v1.1 \
      --sample-root data/fender_samples \
      --out-root DadaSet \
      --data-percentage 0.05 \
      --max-render-fret 19 \
      --max-tab-fret 19 \
      --max-segments 1000
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


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

DEFAULT_TICKS_PER_QUARTER = 960

N_STRINGS = 6
LOW_E_MIDI = 40
N_F0_CLASSES = 44

# Tab/GuitarSet order: low E -> high e.
OPEN_MIDI_LOW_TO_HIGH = np.array([40, 45, 50, 55, 59, 64], dtype=np.int32)
STRING_NAMES_LOW_TO_HIGH = ["E string", "A string", "D string", "G string", "B string", "e string"]

# PyGuitarPro string order: 1=high e ... 6=low E.
GP_STRING_TO_OPEN_MIDI = {
    6: 40,
    5: 45,
    4: 50,
    3: 55,
    2: 59,
    1: 64,
}


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
# Utilities
# ---------------------------------------------------------------------

def safe_name(path_or_text, max_len: int = 80) -> str:
    if isinstance(path_or_text, Path):
        s = path_or_text.stem
    else:
        s = str(path_or_text)

    # Remove hyphens deliberately. Tab-Estimator's jams_to_midi.py extracts
    # tempo using jams_filename.split('-')[1], so the first dash must be the
    # dash immediately before tempo.
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", s)
    s = s.strip("_")
    return (s or "piece")[:max_len]


def gp_string_to_tab_index(gp_string: int) -> int:
    """GP: 1=high e..6=low E. Tab/GuitarSet: 0=low E..5=high e."""
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
        return val / sr if val > 5.0 else max(0.0, val)
    except Exception:
        return 0.0


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


def atomic_write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    sf.write(str(tmp), audio.astype(np.float32), sr)
    os.replace(tmp, path)


def atomic_jams_save(jam: jams.JAMS, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    jam.save(str(tmp))
    os.replace(tmp, path)


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
    Load note samples from either the new flat layout or the old multi-instance layout.

    New:
        sample_root/string1/0.wav
        sample_root/string1/0.txt

    Old:
        sample_root/fender1/string1/0.wav
    """
    bank: Dict[Tuple[int, int], List[NoteSample]] = defaultdict(list)

    sample_root = Path(sample_root)
    onset_root = Path(onset_root) if onset_root is not None else sample_root

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
                    y = librosa.resample(y.astype(np.float32), orig_sr=src_sr, target_sr=sr)

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
    require_sample_bank: Dict[Tuple[int, int], List[NoteSample]],
) -> Tuple[float, Dict[str, List[NoteEvent]]]:
    song = gp.parse(str(gp_path))
    tempo = get_song_tempo(song)
    sec_per_tick = (60.0 / tempo) / float(ticks_per_quarter())

    tracks_events: Dict[str, List[NoteEvent]] = {}

    for track_idx, track in enumerate(getattr(song, "tracks", [])):
        if not is_probably_guitar_track(track):
            continue

        track_name = getattr(track, "name", f"track{track_idx}")
        safe_track_name = safe_name(track_name, max_len=40)
        track_key = f"t{track_idx:02d}_{safe_track_name}"

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
                        if not require_sample_bank.get((gp_string, fret)):
                            continue

                        tab_string = gp_string_to_tab_index(gp_string)
                        midi = midi_from_tab_string_fret(tab_string, fret)

                        if midi < LOW_E_MIDI or midi >= LOW_E_MIDI + N_F0_CLASSES:
                            continue

                        velocity = float(getattr(note, "velocity", 95)) / 95.0
                        velocity = max(0.1, min(1.2, velocity))

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
                    track_name=track_name,
                )
            )

        events.sort(key=lambda e: (e.start_sec, e.tab_string, e.fret))
        tracks_events[track_key] = events

    return tempo, tracks_events


def segment_starts(events: Sequence[NoteEvent], tempo: float, note_resolution: int) -> List[Tuple[float, float]]:
    if not events:
        return []
    note_dur = 60.0 / tempo / note_resolution * 4.0
    seg_dur = note_resolution * 4 * note_dur  # 4 bars if note_resolution=16
    end = max(e.end_sec for e in events)
    return [(i * seg_dur, seg_dur) for i in range(int(math.ceil(end / seg_dur)))]


def events_in_segment(events: Sequence[NoteEvent], st: float, dur: float) -> List[NoteEvent]:
    en = st + dur
    return [e for e in events if e.end_sec > st and e.start_sec < en]


# ---------------------------------------------------------------------
# Rendering and JAMS writing
# ---------------------------------------------------------------------

def render_segment_audio(
    events: Sequence[NoteEvent],
    seg_start: float,
    seg_dur: float,
    bank: Dict[Tuple[int, int], List[NoteSample]],
    sr: int,
    rng,
    tail_sec: float,
) -> np.ndarray:
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
        sample = sample * ev.velocity * float(rng.uniform(0.75, 1.05))

        jitter = int(rng.normal(0, 0.004 * sr))
        insert_at = max(0, insert_at + jitter)

        if insert_at >= n:
            continue

        sample = sample[:n - insert_at]
        sample = fade_edges(sample, min(128, sample.size // 8))
        audio[insert_at:insert_at + sample.size] += sample

    return normalize_audio(audio, peak=0.95)


def make_jams_for_segment(
    events: Sequence[NoteEvent],
    seg_start: float,
    seg_dur: float,
    tempo: float,
    title: str,
    source_gp: str,
    track_name: str,
    note_resolution: int,
) -> jams.JAMS:
    jam = jams.JAMS()

    jam.file_metadata.title = title
    jam.file_metadata.duration = float(seg_dur)

    # Store useful provenance in sandbox metadata.
    jam.sandbox.dadaset = {
        "source_gp": source_gp,
        "track_name": track_name,
        "segment_start_sec": float(seg_start),
        "tempo": float(tempo),
        "note_resolution": int(note_resolution),
    }

    # Six note_midi annotations, low-to-high string order.
    anns = []
    for string_idx, string_name in enumerate(STRING_NAMES_LOW_TO_HIGH):
        ann = jams.Annotation(namespace="note_midi")
        ann.annotation_metadata = jams.AnnotationMetadata(
            curator=jams.Curator(name="DadaSet synthetic generator"),
            data_source="DadaGP rendered with recorded single-note samples",
        )
        ann.sandbox.string_index = int(string_idx)
        ann.sandbox.string_name = string_name
        anns.append(ann)

    for ev in events:
        local_start = ev.start_sec - seg_start
        local_end = ev.end_sec - seg_start

        if local_end <= 0 or local_start >= seg_dur:
            continue

        st = max(0.0, local_start)
        en = min(seg_dur, local_end)
        dur = max(0.03, en - st)

        anns[ev.tab_string].append(
            time=float(st),
            duration=float(dur),
            value=float(ev.midi),
            confidence=1.0,
        )

    for ann in anns:
        jam.annotations.append(ann)

    # Optional beat_position annotation. Tab-Estimator's jams_to_midi.py does not
    # need this, but keeping it makes the JAMS more GuitarSet-like.
    beat_ann = jams.Annotation(namespace="beat_position")
    beat_ann.annotation_metadata = jams.AnnotationMetadata(
        curator=jams.Curator(name="DadaSet synthetic generator"),
        data_source="synthetic beat grid",
    )

    beat_dur = 60.0 / float(tempo)
    n_beats = int(math.ceil(seg_dur / beat_dur)) + 1

    for b in range(n_beats):
        t = b * beat_dur
        if t > seg_dur:
            break
        beat_ann.append(
            time=float(t),
            duration=0.0,
            value={"position": int((b % 4) + 1)},
            confidence=1.0,
        )

    jam.annotations.append(beat_ann)

    return jam


def count_active_steps(events: Sequence[NoteEvent], seg_start: float, seg_dur: float, tempo: float, note_resolution: int) -> int:
    note_dur = 60.0 / tempo / note_resolution * 4.0
    len_notes = note_resolution * 4
    active = np.zeros(len_notes, dtype=bool)

    for ev in events:
        local_start = ev.start_sec - seg_start
        local_end = ev.end_sec - seg_start

        if local_end <= 0 or local_start >= seg_dur:
            continue

        n0 = int(round(max(0.0, local_start) / note_dur))
        n1 = int(math.ceil(min(seg_dur, local_end) / note_dur))
        n0 = max(0, min(len_notes, n0))
        n1 = max(0, min(len_notes, n1))

        if n1 <= n0:
            n1 = min(len_notes, n0 + 1)

        active[n0:n1] = True

    return int(active.sum())


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

    print("[1/5] Loading note samples...")
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
        raise SystemExit("No note samples loaded. Check --sample-root.")

    print(f"Loaded {sum(len(v) for v in bank.values())} note instances across {len(bank)} string/fret keys.")

    print("[2/5] Selecting DadaGP files...")
    gp_files = collect_gp_files(Path(args.dadagp_root), args.data_percentage, args.seed)
    print(f"Selected {len(gp_files)} files.")

    failed = []
    skipped = Counter()
    written = 0

    print("[3/5] Rendering DadaSet segments...")
    for file_idx, gp_path in enumerate(tqdm(gp_files)):
        fold = f"{file_idx % args.n_folds:02d}"

        try:
            tempo, tracks = parse_guitarpro_file(
                gp_path=gp_path,
                max_render_fret=args.max_render_fret,
                max_tab_fret=args.max_tab_fret,
                min_notes=args.min_notes_per_track,
                require_sample_bank=bank,
            )
        except Exception as exc:
            failed.append({"file": str(gp_path), "error": repr(exc)})
            skipped["parse_error"] += 1
            continue

        if tempo < args.min_tempo or tempo > args.max_tempo:
            skipped["tempo_out_of_range"] += 1
            continue

        tempo_for_name = int(round(tempo))
        piece_name = safe_name(gp_path)

        for track_key, events in tracks.items():
            for seg_idx, (seg_st, seg_dur) in enumerate(segment_starts(events, tempo, args.note_resolution)):
                seg_events = events_in_segment(events, seg_st, seg_dur)

                if not seg_events:
                    skipped["empty_segment"] += 1
                    continue

                active_steps = count_active_steps(
                    seg_events,
                    seg_st,
                    seg_dur,
                    tempo,
                    args.note_resolution,
                )

                if active_steps < args.min_active_steps:
                    skipped["not_enough_active_steps"] += 1
                    continue

                # Filename must have first dash before tempo:
                #   00_piece-120-DadaSet_t00_seg0000.jams
                # because Tab-Estimator does: float(jams_filename.split('-')[1])
                base = f"{fold}_{piece_name}-{tempo_for_name}-DadaSet_{track_key}_seg{seg_idx:04d}"

                jams_path = ann_dir / f"{base}.jams"
                wav_path = audio_dir / f"{base}_mic.wav"

                audio = render_segment_audio(
                    events=seg_events,
                    seg_start=seg_st,
                    seg_dur=seg_dur,
                    bank=bank,
                    sr=args.sr,
                    rng=rng,
                    tail_sec=args.sample_tail_sec,
                )

                jam = make_jams_for_segment(
                    events=seg_events,
                    seg_start=seg_st,
                    seg_dur=seg_dur,
                    tempo=tempo,
                    title=base,
                    source_gp=str(gp_path),
                    track_name=track_key,
                    note_resolution=args.note_resolution,
                )

                try:
                    atomic_write_wav(wav_path, audio, args.sr)
                    atomic_jams_save(jam, jams_path)
                except Exception as exc:
                    failed.append({"file": str(gp_path), "segment": base, "error": repr(exc)})
                    skipped["write_error"] += 1
                    continue

                written += 1

                if args.max_segments is not None and written >= args.max_segments:
                    break

            if args.max_segments is not None and written >= args.max_segments:
                break

        if args.max_segments is not None and written >= args.max_segments:
            break

    print("[4/5] Writing logs...")
    stats = {
        "out_root": str(out_root),
        "annotation_dir": str(ann_dir),
        "audio_dir": str(audio_dir),
        "selected_gp_files": len(gp_files),
        "written_segments": written,
        "failed_files": len(failed),
        "skipped": dict(skipped),
        "args": vars(args),
    }

    with (out_root / "dadaset_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    with (out_root / "dadaset_failed.json").open("w", encoding="utf-8") as f:
        json.dump(failed, f, indent=2)

    print("[5/5] Done.")
    print(json.dumps(stats, indent=2))

    print("\nNext steps:")
    print("  cd Tab-estimator")
    print("  mv GuitarSet GuitarSet_real  # only if you already have GuitarSet there")
    print(f"  ln -s {out_root.resolve()} GuitarSet")
    print("  python src/jams_to_midi.py")
    print("  python src/midi_to_numpy.py")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create GuitarSet-like DadaSet/annotation and DadaSet/audio_mono-mic from DadaGP."
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
    p.add_argument("--note-resolution", type=int, default=16)

    p.add_argument("--max-tab-fret", type=int, default=19)
    p.add_argument("--max-render-fret", type=int, default=19)

    p.add_argument("--min-notes-per-track", type=int, default=8)
    p.add_argument("--min-active-steps", type=int, default=2)

    p.add_argument("--min-tempo", type=float, default=40.0)
    p.add_argument("--max-tempo", type=float, default=240.0)

    p.add_argument("--sample-tail-sec", type=float, default=0.25)
    p.add_argument("--max-segments", type=int, default=None)

    return p.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
