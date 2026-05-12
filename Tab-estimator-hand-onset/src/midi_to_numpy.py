#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
midi_to_numpy.py

Argument-based version for Tab-Estimator.

Examples
--------
DadaSet, non-quantized MIDI:
    python src/midi_to_numpy.py --dataset-dir DadaSet --n-cores 1

DadaSet, quantized MIDI:
    python src/midi_to_numpy.py --dataset-dir DadaSet --quantized --n-cores 1

Expected paths for --dataset-dir DadaSet:
    reads MIDI:
        data/midi/dadaset/*.mid
    reads audio:
        DadaSet/audio_mono-mic/*_mic.wav
    writes NPZ:
        data/npz/dadaset/
        data/npz/dadaset/split/

Expected paths for --dataset-dir DadaSet --quantized:
    reads MIDI:
        data/midi/auto_quantized_16_dadaset/*.mid
    reads audio:
        DadaSet/audio_mono-mic/*_mic.wav
    writes NPZ:
        data/npz/auto_quantized_16_dadaset/
        data/npz/auto_quantized_16_dadaset/split/
"""

import argparse
import glob
import math
import os
from itertools import repeat
from multiprocessing import Pool

import librosa
import numpy as np
import pretty_midi
import tqdm
import yaml
from scipy.io import wavfile


STRING_BASE_PITCH = {
    "E string": 40,
    "A string": 45,
    "D string": 50,
    "G string": 55,
    "B string": 59,
    "e string": 64,
}

STRING_INDEX = {
    "E string": 0,
    "A string": 1,
    "D string": 2,
    "G string": 3,
    "B string": 4,
    "e string": 5,
}


def dataset_name_from_dir(dataset_dir):
    return os.path.basename(os.path.normpath(dataset_dir)).lower()


def pitch_to_nfrets(pitch, string_name):
    if string_name not in STRING_BASE_PITCH:
        raise ValueError(f"Unknown MIDI instrument/string name: {string_name}")

    fret = int(pitch) - STRING_BASE_PITCH[string_name]
    string_n = STRING_INDEX[string_name]
    return fret, string_n


def load_audio_mono(audio_filename):
    sr_original, audio_file = wavfile.read(audio_filename)
    audio_file = audio_file.astype(np.float32)

    if audio_file.ndim == 2:
        audio_file = np.mean(audio_file, axis=1)

    return sr_original, audio_file


def resample_audio(data, sr_original, target_sr):
    data = data.astype(np.float32)

    if sr_original == target_sr:
        return data

    try:
        return librosa.resample(data, orig_sr=sr_original, target_sr=target_sr)
    except TypeError:
        # librosa 0.8.x fallback
        return librosa.resample(data, sr_original, target_sr)


def process_cqt(data, sr_original, **kwargs):
    down_sampling_rate = kwargs["down_sampling_rate"]
    bins_per_octave = kwargs["bins_per_octave"]
    n_bins = kwargs["n_bins"]
    hop_length = kwargs["hop_length"]

    data = librosa.util.normalize(data.astype(np.float32))
    data = resample_audio(data, sr_original, down_sampling_rate)

    cqt = np.abs(
        librosa.cqt(
            data,
            hop_length=hop_length,
            sr=down_sampling_rate,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
        )
    )

    return cqt


def process_mel_spec(data, sr_original, **kwargs):
    down_sampling_rate = kwargs["down_sampling_rate"]
    hop_length = kwargs["hop_length"]

    data = librosa.util.normalize(data.astype(np.float32))
    data = resample_audio(data, sr_original, down_sampling_rate)

    mel_spec = np.abs(
        librosa.feature.melspectrogram(
            y=data,
            sr=down_sampling_rate,
            n_fft=2048,
            hop_length=hop_length,
        )
    )

    return mel_spec


def pad_or_trim_2d(x, target_len):
    if x.shape[0] >= target_len:
        return x[:target_len].astype(np.float32)

    pad = np.zeros((target_len - x.shape[0], x.shape[1]), dtype=np.float32)
    return np.vstack([x.astype(np.float32), pad])


def safe_add_tab(tab_array, start, end, string_n, fret):
    if fret < 0 or fret >= 20:
        return

    start = int(start)
    end = int(end)

    start = max(0, min(tab_array.shape[0], start))
    end = max(0, min(tab_array.shape[0], end))

    if end <= start:
        end = min(tab_array.shape[0], start + 1)

    if start >= tab_array.shape[0]:
        return

    tab_array[start:end, string_n, 20] = 0
    tab_array[start:end, string_n, fret] = 1


def safe_add_tab_onset(tab_array, t, string_n, fret):
    if fret < 0 or fret >= 20:
        return

    t = int(t)

    if 0 <= t < tab_array.shape[0]:
        tab_array[t, string_n, 20] = 0
        tab_array[t, string_n, fret] = 1


def safe_add_f0(f0_array, start, end, pitch_index):
    if pitch_index < 0 or pitch_index >= 44:
        return

    start = int(start)
    end = int(end)

    start = max(0, min(f0_array.shape[0], start))
    end = max(0, min(f0_array.shape[0], end))

    if end <= start:
        end = min(f0_array.shape[0], start + 1)

    if start < f0_array.shape[0]:
        f0_array[start:end, pitch_index] = 1


def safe_add_f0_onset(f0_array, t, pitch_index):
    if pitch_index < 0 or pitch_index >= 44:
        return

    t = int(t)

    if 0 <= t < f0_array.shape[0]:
        f0_array[t, pitch_index] = 1


def save_npz_atomic(npz_path_no_ext, **payload):
    final_path = npz_path_no_ext + ".npz"
    tmp_path = npz_path_no_ext + ".tmp.npz"

    try:
        np.savez_compressed(tmp_path, **payload)
        os.replace(tmp_path, final_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def split_save(
    npz_path_no_ext,
    cqt,
    log_cqt,
    mel_spec,
    tab,
    tab_onset,
    frame_tab,
    frame_tab_onset,
    F0,
    F0_onset,
    frame_F0,
    frame_F0_onset,
    tempo,
    note_resolution,
):
    feature_len = cqt.shape[0]
    note_len = tab.shape[0]

    note_len_4bars = int(round(float(note_len) / float(note_resolution * 4)))

    if note_len_4bars <= 0:
        return

    feature_len_4bars = int(round(float(feature_len) / float(note_len_4bars)))

    split_npz_path = os.path.join(os.path.split(npz_path_no_ext)[0], "split")
    os.makedirs(split_npz_path, exist_ok=True)

    for n_4bars in range(note_len_4bars):
        note_start = (note_resolution * 4) * n_4bars
        note_end = (note_resolution * 4) * (n_4bars + 1)

        frame_start = feature_len_4bars * n_4bars
        frame_end = feature_len_4bars * (n_4bars + 1)

        split_cqt = cqt[frame_start:frame_end]
        split_log_cqt = log_cqt[frame_start:frame_end]
        split_mel_spec = mel_spec[frame_start:frame_end]
        split_frame_tab = frame_tab[frame_start:frame_end]
        split_frame_tab_onset = frame_tab_onset[frame_start:frame_end]
        split_frame_F0 = frame_F0[frame_start:frame_end]
        split_frame_F0_onset = frame_F0_onset[frame_start:frame_end]

        split_tab = tab[note_start:note_end]
        split_tab_onset = tab_onset[note_start:note_end]
        split_F0 = F0[note_start:note_end]
        split_F0_onset = F0_onset[note_start:note_end]

        split_npz_filename = os.path.join(
            split_npz_path,
            os.path.split(npz_path_no_ext)[1] + f"_0{n_4bars}",
        )

        np.savez_compressed(
            split_npz_filename,
            cqt=split_cqt.astype(np.float32),
            log_cqt=split_log_cqt.astype(np.float32),
            mel_spec=split_mel_spec.astype(np.float32),
            tab=split_tab.astype(np.float32),
            tab_onset=split_tab_onset.astype(np.float32),
            frame_tab=split_frame_tab.astype(np.float32),
            frame_tab_onset=split_frame_tab_onset.astype(np.float32),
            F0=split_F0.astype(np.float32),
            F0_onset=split_F0_onset.astype(np.float32),
            frame_F0=split_frame_F0.astype(np.float32),
            frame_F0_onset=split_frame_F0_onset.astype(np.float32),
            tempo=tempo,
            len_in_notes=(note_resolution * 4),
        )


def process_midi_file(midi_filename, kwargs):
    note_resolution = kwargs["note_resolution"]
    down_sampling_rate = kwargs["down_sampling_rate"]
    hop_length = kwargs["hop_length"]
    audio_dir = kwargs["audio_dir"]
    npz_dir = kwargs["npz_dir"]

    norm_len = down_sampling_rate / hop_length

    midi_base = os.path.split(midi_filename)[1][:-4]
    audio_filename = os.path.join(audio_dir, midi_base + "_mic.wav")

    if not os.path.exists(audio_filename):
        print(f"[missing audio] {audio_filename}")
        return None

    try:
        sr_original, audio_file = load_audio_mono(audio_filename)

        cqt = process_cqt(audio_file, sr_original, **kwargs)
        log_cqt = librosa.amplitude_to_db(np.abs(cqt))
        mel_spec = process_mel_spec(audio_file, sr_original, **kwargs)

        cqt = cqt.T.astype(np.float32)
        log_cqt = log_cqt.T.astype(np.float32)
        mel_spec = mel_spec.T.astype(np.float32)

        midi_file = pretty_midi.PrettyMIDI(midi_filename)

        tempo = float(os.path.basename(midi_filename).split("-")[1])
        note_dur = 60.0 / tempo / note_resolution * 4.0

        len_in_notes = int(
            math.ceil(
                round(midi_file.get_end_time() / note_dur)
                / (note_resolution * 4)
            )
            * (note_resolution * 4)
        )

        if len_in_notes <= 0:
            print(f"[empty midi duration] {midi_filename}")
            return None

        feature_len = int(
            len_in_notes * note_dur * (down_sampling_rate / hop_length)
        )

        if feature_len <= 0:
            print(f"[empty feature len] {midi_filename}")
            return None

        cqt = pad_or_trim_2d(cqt, feature_len)
        log_cqt = pad_or_trim_2d(log_cqt, feature_len)
        mel_spec = pad_or_trim_2d(mel_spec, feature_len)

        tab = np.zeros((len_in_notes, 6, 21), dtype=np.float32)
        tab[:, :, 20] = 1

        tab_onset = np.zeros((len_in_notes, 6, 21), dtype=np.float32)
        tab_onset[:, :, 20] = 1

        frame_tab = np.zeros((feature_len, 6, 21), dtype=np.float32)
        frame_tab[:, :, 20] = 1

        frame_tab_onset = np.zeros((feature_len, 6, 21), dtype=np.float32)
        frame_tab_onset[:, :, 20] = 1

        F0 = np.zeros((len_in_notes, 44), dtype=np.float32)
        F0_onset = np.zeros((len_in_notes, 44), dtype=np.float32)

        frame_F0 = np.zeros((feature_len, 44), dtype=np.float32)
        frame_F0_onset = np.zeros((feature_len, 44), dtype=np.float32)

        for midi_string in midi_file.instruments:
            string_name = midi_string.name

            if string_name not in STRING_BASE_PITCH:
                print(f"[skip unknown instrument] {string_name} in {midi_filename}")
                continue

            for note in midi_string.notes:
                fret, string_n = pitch_to_nfrets(note.pitch, string_name)
                pitch_index = int(note.pitch) - 40

                n0 = int(round(note.start / note_dur))
                n1 = int(round(note.end / note_dur))
                f0 = int(round(note.start * norm_len))
                f1 = int(round(note.end * norm_len))

                safe_add_tab(tab, n0, n1, string_n, fret)
                safe_add_tab_onset(tab_onset, n0, string_n, fret)

                safe_add_tab(frame_tab, f0, f1, string_n, fret)
                safe_add_tab_onset(frame_tab_onset, f0, string_n, fret)

                safe_add_f0(F0, n0, n1, pitch_index)
                safe_add_f0_onset(F0_onset, n0, pitch_index)

                safe_add_f0(frame_F0, f0, f1, pitch_index)
                safe_add_f0_onset(frame_F0_onset, f0, pitch_index)

        npz_path_no_ext = os.path.join(npz_dir, midi_base)

        save_npz_atomic(
            npz_path_no_ext,
            cqt=cqt,
            log_cqt=log_cqt,
            mel_spec=mel_spec,
            tab=tab,
            tab_onset=tab_onset,
            frame_tab=frame_tab,
            frame_tab_onset=frame_tab_onset,
            F0=F0,
            F0_onset=F0_onset,
            frame_F0=frame_F0,
            frame_F0_onset=frame_F0_onset,
            tempo=tempo,
            len_in_notes=len_in_notes,
        )

        split_save(
            npz_path_no_ext,
            cqt,
            log_cqt,
            mel_spec,
            tab,
            tab_onset,
            frame_tab,
            frame_tab_onset,
            F0,
            F0_onset,
            frame_F0,
            frame_F0_onset,
            tempo,
            note_resolution,
        )

        print("finished", midi_base)
        return npz_path_no_ext + ".npz"

    except Exception as exc:
        print(f"[error] {midi_filename}: {repr(exc)}")
        return None


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        default="GuitarSet",
        help="Dataset root containing audio_mono-mic/, e.g. GuitarSet or DadaSet.",
    )

    parser.add_argument(
        "--quantized",
        action="store_true",
        help=(
            "Use data/midi/auto_quantized_<note_resolution>_<dataset_name>/ "
            "and write data/npz/auto_quantized_<note_resolution>_<dataset_name>/."
        ),
    )

    parser.add_argument(
        "--midi-dir",
        default=None,
        help="Explicit MIDI input directory. Overrides dataset-based MIDI path.",
    )

    parser.add_argument(
        "--audio-dir",
        default=None,
        help="Explicit audio directory. Default: <dataset-dir>/audio_mono-mic.",
    )

    parser.add_argument(
        "--npz-dir",
        default=None,
        help="Explicit output NPZ directory. Overrides dataset-based NPZ path.",
    )

    parser.add_argument(
        "--config",
        default="src/config.yaml",
        help="Path to config.yaml.",
    )

    parser.add_argument(
        "--n-cores",
        type=int,
        default=None,
        help="Override n_cores from config. Use 1 for easier debugging.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        obj = yaml.safe_load(f)

    note_resolution = int(obj["note_resolution"])
    down_sampling_rate = int(obj["down_sampling_rate"])
    bins_per_octave = int(obj["bins_per_octave"])
    n_bins = int(obj["cqt_n_bins"])
    hop_length = int(obj["hop_length"])
    n_cores = int(args.n_cores if args.n_cores is not None else obj["n_cores"])

    dataset_name = dataset_name_from_dir(args.dataset_dir)

    if args.quantized:
        midi_dir_default = os.path.join(
            "data",
            "midi",
            f"auto_quantized_{note_resolution}_{dataset_name}",
        )
        npz_dir_default = os.path.join(
            "data",
            "npz",
            f"auto_quantized_{note_resolution}_{dataset_name}",
        )
    else:
        midi_dir_default = os.path.join("data", "midi", dataset_name)
        npz_dir_default = os.path.join("data", "npz", dataset_name)

    midi_dir = args.midi_dir if args.midi_dir else midi_dir_default
    audio_dir = args.audio_dir if args.audio_dir else os.path.join(
        args.dataset_dir,
        "audio_mono-mic",
    )
    npz_dir = args.npz_dir if args.npz_dir else npz_dir_default

    os.makedirs(npz_dir, exist_ok=True)

    midi_filename_list = sorted(glob.glob(os.path.join(midi_dir, "*.mid")))

    print("dataset_dir:", args.dataset_dir)
    print("dataset_name:", dataset_name)
    print("quantized:", args.quantized)
    print("midi_dir:", midi_dir)
    print("audio_dir:", audio_dir)
    print("npz_dir:", npz_dir)
    print("midi files:", len(midi_filename_list))

    kwargs = {
        "note_resolution": note_resolution,
        "down_sampling_rate": down_sampling_rate,
        "bins_per_octave": bins_per_octave,
        "n_bins": n_bins,
        "hop_length": hop_length,
        "audio_dir": audio_dir,
        "npz_dir": npz_dir,
    }

    if n_cores <= 1:
        for midi_filename in tqdm.tqdm(midi_filename_list):
            process_midi_file(midi_filename, kwargs)
    else:
        p = Pool(n_cores)
        p.starmap(process_midi_file, zip(midi_filename_list, repeat(kwargs)))
        p.close()
        p.join()

    missing = 0

    for midi_filename in midi_filename_list:
        name = os.path.split(midi_filename)[1][:-4]
        expected = os.path.join(npz_dir, name + ".npz")

        if not os.path.exists(expected):
            print(f"{name} does not exist!")
            missing += 1

    print("missing npz files:", missing)


if __name__ == "__main__":
    main()
