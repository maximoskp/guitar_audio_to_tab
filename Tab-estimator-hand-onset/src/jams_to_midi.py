import argparse
import jams
import numpy as np
import os
import glob
import jams_interpreter
import pretty_midi
import tqdm
import yaml


def dataset_name_from_dir(dataset_dir):
    return os.path.basename(os.path.normpath(dataset_dir)).lower()


def main(note_resolution, dataset_dir):
    dataset_name = dataset_name_from_dir(dataset_dir)

    jams_file_path = os.path.join(dataset_dir, "annotation", "*")
    jams_filename_list = glob.glob(jams_file_path)
    jams_filename_list.sort()

    midi_dir = os.path.join("data", "midi")
    midi_dir_dataset = os.path.join(midi_dir, dataset_name)
    midi_dir_quantized = os.path.join(
        midi_dir, f"auto_quantized_{note_resolution}_{dataset_name}"
    )

    os.makedirs(midi_dir_dataset, exist_ok=True)
    os.makedirs(midi_dir_quantized, exist_ok=True)

    for jams_filename in tqdm.tqdm(jams_filename_list):
        jams_file = jams.load(jams_filename)

        print(f"Processing {jams_filename}...")
        # Keeps original Tab-Estimator filename convention:
        # something-120-something.jams
        tempo = float(os.path.basename(jams_filename).split("-")[1])

        midi_file_original = jams_interpreter.jams_to_midi(
            jams_file, tempo=tempo, q=0
        )
        print(f"Original MIDI file for {jams_filename} has {len(midi_file_original.instruments[0].notes)} notes.")
        
        midi_file_quantized = jams_interpreter.jams_to_midi(
            jams_file, tempo=tempo, q=0, quantization=note_resolution
        )

        print(f"Saving MIDI files for {jams_filename}...")
        midi_filename_dataset = os.path.join(
            midi_dir_dataset,
            os.path.split(jams_filename)[1][:-5],
        )
        midi_filename_quantized = os.path.join(
            midi_dir_quantized,
            os.path.split(jams_filename)[1][:-5],
        )

        midi_file_original.write(midi_filename_dataset + ".mid")
        midi_file_quantized.write(midi_filename_quantized + ".mid")

    visualize_path = os.path.join("visualize")
    os.makedirs(visualize_path, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        default="GuitarSet",
        help="Dataset root containing annotation/*.jams, e.g. GuitarSet or DadaSet",
    )
    args = parser.parse_args()

    with open("src/config.yaml") as f:
        obj = yaml.safe_load(f)
        note_resolution = obj["note_resolution"]

    main(note_resolution, args.dataset_dir)