python src/train.py   --npz-dir data/npz/guitarset_guitartechs_egdb_goat_idmt/split   --run-name guitarset_guitartechs_egdb_goat_idmt_onset_only   --test-num 6   --n-folds 7   --epoch 192

python src/predict.py  guitarset_guitartechs_egdb_goat_idmt/guitarset_guitartechs_egdb_goat_idmt_onset 192 --test-num 06  --onset-threshold 0.80 --n-folds 7 --npz-dir data/npz/guitarset_guitartechs_egdb_goat_idmt/split


# Onset-Detection

Audio-only onset detection for guitar. This is a simplified version of Tab-Estimator-Hand-Onset that focuses exclusively on predicting note onsets from audio without tablature estimation or hand-position input.

## Architecture

The model consists of:

1. **Audio Frontend**: ConvStack (Conv2D -> MaxPool -> FC)
2. **Encoder**: Conformer or Transformer encoder
3. **Onset Heads**: Per-string and global non-causal gated TCN heads
   - Per-string onset: (B, T, 6) logits
   - Global onset: (B, T) logits

Both onset heads optionally receive:
- Encoder memory states
- Projected raw CQT/mel features (if `onset_use_raw_features=True`)

## Model Configurations

The onset input can be configured via `--onset-input-mode`:

- **full** (default): Concatenate encoder memory and projected raw features
- **encoder**: Use encoder memory only
- **raw**: Use projected raw features only

## Training

### Basic Training Command

```bash
python src/train.py \
  --npz-dir data/npz/guitarset_guitartechs_egdb_goat_idmt/split \
  --run-name guitarset_guitartechs_egdb_goat_idmt_onset \
  --test-num 6 \
  --n-folds 7 \
  --epoch 192
```

### Key Training Arguments

- `--npz-dir`: Path to NPZ split containing CQT/mel features and onset targets
- `--run-name`: Name for the training run (used in model dir)
- `--test-num`: Fold number (0-indexed, uses `{test_num:02d}_*.npz` files)
- `--n-folds`: Total number of folds (default: 6)
- `--epoch`: Number of training epochs (default: from config)
- `--onset-loss-weight`: Weight for per-string onset loss (default: 0.25)
- `--global-onset-loss-weight`: Weight for global onset loss (default: 0.25)
- `--onset-positive-weight`: Positive class weight for per-string BCEWithLogitsLoss (default: 10.0)
- `--global-onset-positive-weight`: Positive class weight for global onset (default: 10.0)
- `--onset-target-radius-ms`: Widen training labels by this many ms (default: 25)
- `--onset-target-mode`: hard, triangular, or gaussian (default: hard)
- `--no-onset-raw-features`: Disable projected raw features in onset heads
- `--onset-raw-proj-dim`: Projection dimension for raw features (default: 64)
- `--onset-raw-dropout`: Dropout for raw feature projection (default: 0.10)
- `--onset-hidden-dim`: TCN hidden channels (default: 64)
- `--onset-dropout`: TCN dropout (default: 0.25)
- `--onset-kernel-size`: TCN kernel size (default: 3)
- `--onset-tcn-levels`: Number of TCN levels (default: 4)
- `--onset-input-mode`: full, encoder, or raw (default: full)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: from config)
- `--device`: cuda or cpu (default: cuda if available)
- `--freeze-encoder`: Freeze encoder parameters
- `--freeze-frontend`: Freeze ConvStack parameters
- `--pretrained-model`: Path or name of pretrained checkpoint

### Training Ablations

**Global-only onset detection** (no per-string supervision):

```bash
python src/train.py \
  --npz-dir data/npz/guitarset/split \
  --run-name guitarset_onset_global_only \
  --onset-loss-weight 0 \
  --global-onset-loss-weight 0.25 \
  --epoch 192
```

**Per-string-only onset detection** (no global supervision):

```bash
python src/train.py \
  --npz-dir data/npz/guitarset/split \
  --run-name guitarset_onset_per_string_only \
  --onset-loss-weight 0.25 \
  --global-onset-loss-weight 0 \
  --epoch 192
```

**Raw-features-only onset** (encoder-free):

```bash
python src/train.py \
  --npz-dir data/npz/guitarset/split \
  --run-name guitarset_onset_raw_only \
  --onset-input-mode raw \
  --epoch 192
```

**Encoder-only onset** (no raw features):

```bash
python src/train.py \
  --npz-dir data/npz/guitarset/split \
  --run-name guitarset_onset_encoder_only \
  --onset-input-mode encoder \
  --no-onset-raw-features \
  --epoch 192
```

## Evaluation

### Basic Prediction Command

```bash
python src/predict.py \
  guitarset_guitartechs_egdb_goat_idmt/guitarset_guitartechs_egdb_goat_idmt_onset \
  192 \
  --test-num 06 \
  --onset-threshold 0.80 \
  --n-folds 7 \
  --npz-dir data/npz/guitarset_guitartechs_egdb_goat_idmt/split
```

### Key Prediction Arguments

- `model`: Model run name (e.g., `dataset/run_name`)
- `epoch`: Checkpoint epoch number
- `--test-num`: Test fold number
- `--all-folds`: Evaluate all folds (0 to `--n-folds-1`)
- `--n-folds`: Number of folds (default: 6)
- `--npz-dir`: NPZ split directory
- `--onset-threshold`: Threshold for sigmoid(logits) (default: 0.5)
- `--onset-tolerance-ms`: Tolerant matching window in ms (default: 25)
- `--no-peak-picking`: Use simple thresholding instead of Madmom-style peak-picking
- `--peak-smooth-ms`: Pre-peak-picking smoothing window in ms (default: 0)
- `--peak-pre-max-ms`: Pre-peak context in ms (default: 50)
- `--peak-post-max-ms`: Post-peak context in ms (default: 50)
- `--peak-combine-ms`: Combine peaks within this many ms (default: 30)
- `--device`: cuda or cpu

### Output Metrics

The evaluation produces onset metrics:

**Headline metrics:**
- `frame_avg_onset_f`: Average per-string onset F1 across test files
- `global_onset_f`: Average global onset F1 across test files

**Detailed metrics (in `metrics.csv`):**
- Per-string onset precision/recall/F1 (tolerant matching, +/- tolerance_ms)
- Global onset precision/recall/F1 (tolerant matching)
- Frame-exact onset precision/recall/F1 (for diagnostic purposes)
- Per-metric TP/FP/FN counts

Results are saved in:
- CSV: `result/audio_onset_detection/{model}_epoch{epoch}/metrics.csv`
- NPZ: `result/audio_onset_detection/{model}_epoch{epoch}/npz/test_{fold_id}/`

## Dataset Format

The Onset-Detection model expects NPZ files with:

**Required:**
- `cqt` (if input_feature_type="cqt"): (T, n_bins=192)
- `mel_spec` (if input_feature_type="melspec"): (T, 128)

**Onset targets** (priority order):
- `frame_onset`: (T, 6) binary per-string onset labels
- `frame_onsets`: Alias for `frame_onset`
- `onset`: Alias for `frame_onset`
- `onsets`: Alias for `frame_onset`
- `frame_tab_onset`: (T, 6, 21) one-hot; max-collapsed to (T, 6)

**Fallback:**
- `frame_tab`: (T, 6, 21) one-hot tab labels; onset derived by detecting fret changes

If neither onset targets nor frame_tab are present, the NPZ is skipped with an error message.

## Project Structure

```
Onset-Detection/
  src/
    network.py        # OnsetDetector and OnsetOnlyLoss models
    train.py          # Training script
    predict.py        # Evaluation script
    config.yaml       # Default configuration
  model/              # Trained checkpoints (created during training)
  result/             # Evaluation results (created during prediction)
  data/               # Symlink or path to NPZ dataset splits
  requirements.txt    # Python dependencies
  README.md           # This file
```

## Dependencies

See `requirements.txt`. Main dependencies:
- torch
- numpy
- scipy
- espnet/espnet2 (for encoders)
- librosa (for audio feature extraction)
- pandas
- scikit-learn
- tqdm
- pyyaml

## Notes

- The model does NOT require hand-position input.
- The model does NOT produce tablature predictions.
- Training uses per-file and per-string onset targets. If onset labels are missing, they are derived from frame_tab (rest class = 20).
- Onset targets can be widened during training (e.g., +/-25 ms) to make the model more robust.
- Evaluation uses millisecond-based tolerant matching: a predicted onset matches a ground-truth onset if they occur within `--onset-tolerance-ms` on the same string.
- Peak-picking follows Madmom conventions: onsets are peaks of the activation that are above a moving average.

## Examples

### Train a basic onset detector on GuitarSet

```bash
# Assuming data/npz/guitarset/split/ contains NPZ files with CQT and onset labels
python src/train.py \
  --npz-dir data/npz/guitarset/split \
  --run-name guitarset_onset_baseline \
  --test-num 0 \
  --epoch 192 \
  --batch-size 32
```

### Evaluate the trained model

```bash
python src/predict.py \
  guitarset/guitarset_onset_baseline \
  192 \
  --test-num 00 \
  --onset-threshold 0.80 \
  --npz-dir data/npz/guitarset/split
```

### Evaluate all folds

```bash
python src/predict.py \
  guitarset/guitarset_onset_baseline \
  192 \
  --all-folds \
  --onset-threshold 0.80 \
  --npz-dir data/npz/guitarset/split
```

## References

- Original Tab-Estimator-Hand-Onset (with hand-position and tablature)
- Conformer Encoder: [Gulati et al., 2021]
- Madmom Peak-Picking: [Böck et al., 2016]
