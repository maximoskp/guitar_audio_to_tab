#!/bin/bash

# Navigate to the EGDB directory
cd /home/gbastas/guitar_audio_to_tab/Tab-estimator-hand/EGDB

# Function to rename files in a directory
rename_files() {
    local dir=$1
    local suffix=$2
    for file in "$dir"/*.wav; do
        if [[ -f "$file" ]]; then
            # Extract the base name without extension
            base=$(basename "$file" .wav)
            # New name
            new_name="${base}_${suffix}.wav"
            # Rename
            mv "$file" "$dir/$new_name"
            echo "Renamed $file to $dir/$new_name"
        fi
    done
}

# Rename in audio_DI
rename_files "audio_DI" "DI"

# Rename in audio_Marshall
rename_files "audio_Marshall" "Marshall"

echo "Renaming complete."