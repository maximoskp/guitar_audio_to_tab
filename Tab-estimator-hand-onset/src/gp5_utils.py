#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gp5_utils.py

Utilities for converting Guitar Pro GP3/GP4/GP5 tablature files into
chord-aware VexTab and event dictionaries for the DeepVideoPanel frontend.

Install:
    pip install PyGuitarPro

Main entry point:
    parse_gp_file_to_vextab(...)

Returned event convention matches the rest of your backend/frontend:
    string_index_low_e_first:
        0 = low E
        5 = high e

    string_number_low_e_first:
        1 = low E
        6 = high e

    vex_string:
        1 = high e
        6 = low E

The generated VexTab supports chords using syntax like:
    notes (3/6.5/5.5/4) 7/3 8/2
"""

from __future__ import annotations

import os
import traceback
from typing import Any, Dict, List, Optional


ALLOWED_GP_EXTENSIONS = {"gp","gp3", "gp4", "gp5"}

# Guitar Pro string convention:
#   1 = high e
#   6 = low E
# VexTab convention is also:
#   1 = high e
#   6 = low E
GP_STRING_TO_VEXTAB_STRING = {
    1: 1,  # high e
    2: 2,  # B
    3: 3,  # G
    4: 4,  # D
    5: 5,  # A
    6: 6,  # low E
}

STANDARD_OPEN_MIDI_BY_GP_STRING = {
    1: 64,  # high e
    2: 59,  # B
    3: 55,  # G
    4: 50,  # D
    5: 45,  # A
    6: 40,  # low E
}


class GP5ConversionError(RuntimeError):
    """Raised when a Guitar Pro file cannot be converted safely."""


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def detect_probably_zipped_gp(path: str) -> bool:
    with open(path, "rb") as f:
        head = f.read(4)
    return head.startswith(b"PK\x03\x04")


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)



def allowed_gp_filename(filename: str) -> bool:
    if not filename or "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in ALLOWED_GP_EXTENSIONS



def import_guitarpro():
    """
    Import PyGuitarPro lazily so your FastAPI app can still start even if the
    optional dependency is missing.
    """
    try:
        import guitarpro  # type: ignore
    except Exception as exc:
        raise GP5ConversionError(
            "Failed to import PyGuitarPro. Install it with: pip install PyGuitarPro"
        ) from exc

    return guitarpro


# -----------------------------------------------------------------------------
# Track / tuning helpers
# -----------------------------------------------------------------------------


def gp_track_tuning_by_string(track: Any) -> Dict[int, int]:
    """
    Return open MIDI pitch by Guitar Pro string number.

    GP convention:
        1 = high e
        6 = low E

    Falls back to standard guitar tuning if the file does not expose tuning.
    """
    tuning = dict(STANDARD_OPEN_MIDI_BY_GP_STRING)

    for string_obj in getattr(track, "strings", []) or []:
        number = getattr(string_obj, "number", None)
        value = getattr(string_obj, "value", None)

        if number is None or value is None:
            continue

        number = safe_int(number, -1)
        value = safe_int(value, -1)

        if 1 <= number <= 6 and value >= 0:
            tuning[number] = value

    return tuning



def is_standard_guitar_tuning(tuning: Dict[int, int]) -> bool:
    return all(
        int(tuning.get(string_number, -999)) == int(STANDARD_OPEN_MIDI_BY_GP_STRING[string_number])
        for string_number in range(1, 7)
    )



def is_guitar_like_gp_track(track: Any) -> bool:
    if bool(getattr(track, "isPercussionTrack", False)):
        return False

    strings = getattr(track, "strings", []) or []
    return len(strings) >= 6



def describe_gp_tracks(song: Any) -> List[Dict[str, Any]]:
    """
    Return useful track metadata for debugging or frontend display.
    """
    rows: List[Dict[str, Any]] = []

    for index, track in enumerate(getattr(song, "tracks", []) or [], start=1):
        tuning = gp_track_tuning_by_string(track)
        rows.append(
            {
                "index": int(index),
                "track_number": safe_int(getattr(track, "number", index), index),
                "track_name": str(getattr(track, "name", "") or ""),
                "is_percussion": bool(getattr(track, "isPercussionTrack", False)),
                "string_count": int(len(getattr(track, "strings", []) or [])),
                "is_guitar_like": bool(is_guitar_like_gp_track(track)),
                "standard_tuning": bool(is_standard_guitar_tuning(tuning)),
                "tuning": {str(k): int(v) for k, v in tuning.items()},
            }
        )

    return rows



def select_gp_tracks(
    song: Any,
    track_index: Optional[int] = None,
    track_name_contains: Optional[str] = None,
    all_guitar_tracks: bool = False,
) -> List[Any]:
    """
    Select one or more tracks from a parsed Guitar Pro song.

    Priority:
      1. explicit track_index
      2. track_name_contains
      3. all guitar-like tracks if requested
      4. first guitar-like track
      5. first non-percussion track
    """
    tracks = list(getattr(song, "tracks", []) or [])

    if not tracks:
        raise GP5ConversionError("The Guitar Pro file does not contain any tracks.")

    if track_index is not None:
        idx = int(track_index)
        if idx < 1 or idx > len(tracks):
            raise ValueError(f"track_index must be between 1 and {len(tracks)}, got {idx}")
        return [tracks[idx - 1]]

    if track_name_contains:
        needle = str(track_name_contains).lower()
        matches = [
            track for track in tracks
            if needle in str(getattr(track, "name", "")).lower()
        ]
        if not matches:
            raise ValueError(f"No GP track name contains: {track_name_contains!r}")
        return matches

    guitar_tracks = [track for track in tracks if is_guitar_like_gp_track(track)]

    if all_guitar_tracks:
        if not guitar_tracks:
            raise GP5ConversionError("No guitar-like tracks found in the Guitar Pro file.")
        return guitar_tracks

    if guitar_tracks:
        return [guitar_tracks[0]]

    for track in tracks:
        if not bool(getattr(track, "isPercussionTrack", False)):
            return [track]

    raise GP5ConversionError("Could not find a usable non-percussion GP track.")


# -----------------------------------------------------------------------------
# Note / beat helpers
# -----------------------------------------------------------------------------


def gp_note_type_name(note: Any) -> str:
    return str(getattr(note, "type", "")).lower()



def should_skip_gp_note(note: Any, include_tied: bool = False) -> bool:
    note_type = gp_note_type_name(note)

    if "dead" in note_type or "rest" in note_type:
        return True

    # For TabEstimator evaluation, tied notes should usually not be treated as
    # new note attacks.
    if not include_tied and "tie" in note_type:
        return True

    return False



def gp_beat_sort_key(beat: Any, fallback_index: int) -> float:
    start = getattr(beat, "start", None)
    if start is None:
        return float(fallback_index)

    try:
        return float(start)
    except Exception:
        return float(fallback_index)



def dedupe_step_notes(notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate notes that target the same fret/string inside a single beat.
    """
    seen = set()
    out = []

    for note in notes:
        key = (int(note["fret"]), int(note["vex_string"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(note)

    return sorted(out, key=lambda x: (int(x["vex_string"]), int(x["fret"])))


# -----------------------------------------------------------------------------
# Conversion
# -----------------------------------------------------------------------------


def gp5_song_to_vextab_and_events(
    song: Any,
    track_index: Optional[int] = None,
    track_name_contains: Optional[str] = None,
    all_guitar_tracks: bool = False,
    require_standard_tuning: bool = True,
    max_fret: int = 19,
    include_tied: bool = False,
) -> Dict[str, Any]:
    """
    Convert a parsed PyGuitarPro song into chord-aware VexTab and event dicts.

    Args:
        song:
            Parsed object from guitarpro.parse(...).
        track_index:
            Optional 1-based track index.
        track_name_contains:
            Optional substring to select track(s) by name.
        all_guitar_tracks:
            If True, combine all guitar-like tracks.
        require_standard_tuning:
            If True, skip non-standard tuning tracks. This is safer because your
            current TabEstimator target space assumes standard tuning.
        max_fret:
            Default 19 because current TabEstimator predicts frets 0..19 plus
            rest class 20.
        include_tied:
            If False, tied GP notes are skipped as new attacks.

    Returns:
        dict with keys:
            title, tempo, vextab, vextab_text, events, stats, selected_tracks, width
    """
    tempo = float(getattr(song, "tempo", 120) or 120)
    title = str(getattr(song, "title", "") or "Uploaded GP Reference")

    selected = select_gp_tracks(
        song,
        track_index=track_index,
        track_name_contains=track_name_contains,
        all_guitar_tracks=all_guitar_tracks,
    )

    stats: Dict[str, int] = {
        "tracks_selected": int(len(selected)),
        "raw_notes": 0,
        "written_notes": 0,
        "written_steps": 0,
        "skipped_tied_or_dead": 0,
        "skipped_bad_string": 0,
        "skipped_bad_fret": 0,
        "skipped_nonstandard_tuning_track": 0,
    }

    selected_tracks: List[Dict[str, Any]] = []
    step_groups: List[Dict[str, Any]] = []

    for track in selected:
        track_name = str(getattr(track, "name", "") or "Track")
        track_number = safe_int(getattr(track, "number", 0), 0)
        tuning = gp_track_tuning_by_string(track)
        standard = is_standard_guitar_tuning(tuning)

        selected_tracks.append(
            {
                "track_number": int(track_number),
                "track_name": track_name,
                "standard_tuning": bool(standard),
                "tuning": {str(k): int(v) for k, v in tuning.items()},
            }
        )

        if require_standard_tuning and not standard:
            stats["skipped_nonstandard_tuning_track"] += 1
            continue

        beat_counter = 0

        for measure in getattr(track, "measures", []) or []:
            for voice in getattr(measure, "voices", []) or []:
                for beat in getattr(voice, "beats", []) or []:
                    beat_counter += 1
                    notes_for_step: List[Dict[str, Any]] = []

                    for gp_note in getattr(beat, "notes", []) or []:
                        stats["raw_notes"] += 1

                        if should_skip_gp_note(gp_note, include_tied=include_tied):
                            stats["skipped_tied_or_dead"] += 1
                            continue

                        gp_string = safe_int(getattr(gp_note, "string", -1), -1)
                        fret = safe_int(getattr(gp_note, "value", -999), -999)

                        if gp_string not in GP_STRING_TO_VEXTAB_STRING:
                            stats["skipped_bad_string"] += 1
                            continue

                        if fret < 0 or fret > int(max_fret):
                            stats["skipped_bad_fret"] += 1
                            continue

                        vex_string = GP_STRING_TO_VEXTAB_STRING[gp_string]

                        notes_for_step.append(
                            {
                                "fret": int(fret),
                                "vex_string": int(vex_string),
                                "gp_string": int(gp_string),
                                "track_name": track_name,
                                "track_number": int(track_number),
                            }
                        )

                    if not notes_for_step:
                        continue

                    step_groups.append(
                        {
                            "sort_key": gp_beat_sort_key(beat, beat_counter),
                            "notes": dedupe_step_notes(notes_for_step),
                        }
                    )

    step_groups = sorted(step_groups, key=lambda x: float(x["sort_key"]))

    vex_tokens: List[str] = []
    events: List[Dict[str, Any]] = []

    for global_step, group in enumerate(step_groups):
        notes = list(group["notes"])

        note_tokens = [
            f'{int(note["fret"])}/{int(note["vex_string"])}'
            for note in notes
        ]

        if len(note_tokens) == 1:
            vex_tokens.append(note_tokens[0])
        else:
            vex_tokens.append(f'({".".join(note_tokens)})')

        for note in notes:
            # Frontend/backend convention:
            #   low-E-first index 0 = low E
            #   VexTab string 6 = low E
            string_index_low_e_first = 6 - int(note["vex_string"])

            events.append(
                {
                    "global_step": int(global_step),
                    "step": int(global_step),
                    "time": None,
                    "fret": int(note["fret"]),
                    "vex_string": int(note["vex_string"]),
                    "gp_string": int(note["gp_string"]),
                    "string_index_low_e_first": int(string_index_low_e_first),
                    "string_number_low_e_first": int(string_index_low_e_first + 1),
                    "track_name": str(note["track_name"]),
                    "track_number": int(note["track_number"]),
                }
            )

    stats["written_steps"] = int(len(step_groups))
    stats["written_notes"] = int(len(events))

    note_text = " ".join(vex_tokens) if vex_tokens else "=:|"
    width = max(1240, 180 + len(vex_tokens) * 58)

    vextab = f"""
options width={width}
tabstave notation=true
notes {note_text}
""".strip()

    return {
        "title": title,
        "tempo": float(tempo),
        "vextab": vextab,
        "vextab_text": vextab,
        "events": events,
        "stats": stats,
        "selected_tracks": selected_tracks,
        "available_tracks": describe_gp_tracks(song),
        "width": int(width),
    }



def parse_gp_file(
    path: str,
    encoding: Optional[str] = None,
) -> Any:
    """
    Parse a Guitar Pro file. Tries default parser first, then cp1252 fallback.
    """
    guitarpro = import_guitarpro()

    if encoding:
        return guitarpro.parse(path, encoding=encoding)

    try:
        return guitarpro.parse(path)
    except UnicodeDecodeError:
        return guitarpro.parse(path, encoding="cp1252")



def parse_gp_file_to_vextab(
    path: str,
    track_index: Optional[int] = None,
    track_name_contains: Optional[str] = None,
    all_guitar_tracks: bool = False,
    require_standard_tuning: bool = True,
    max_fret: int = 19,
    include_tied: bool = False,
    encoding: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Parse a GP3/GP4/GP5 file and return chord-aware VexTab + events.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    if not allowed_gp_filename(path):
        raise ValueError(
            f"Unsupported Guitar Pro extension for {path!r}. "
            f"Allowed: {sorted(ALLOWED_GP_EXTENSIONS)}"
        )
    if detect_probably_zipped_gp(path):
        raise GP5ConversionError(
            "This .gp file appears to be a newer ZIP-based Guitar Pro format "
            "(GP6/GP7/GP8-style), not an old GP3/GP4/GP5 binary file. "
            "Please export or convert it to .gp5 first, then upload the .gp5 file."
        )

    try:
        song = parse_gp_file(path, encoding=encoding)
        return gp5_song_to_vextab_and_events(
            song=song,
            track_index=track_index,
            track_name_contains=track_name_contains,
            all_guitar_tracks=all_guitar_tracks,
            require_standard_tuning=require_standard_tuning,
            max_fret=int(max_fret),
            include_tied=bool(include_tied),
        )
    except Exception as exc:
        # Preserve original error in FastAPI traces while giving a useful message.
        if isinstance(exc, (ValueError, FileNotFoundError, GP5ConversionError)):
            raise

        raise GP5ConversionError(
            f"Failed to parse/convert Guitar Pro file: {path}\n{repr(exc)}\n{traceback.format_exc()}"
        ) from exc
