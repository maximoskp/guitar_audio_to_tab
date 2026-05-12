#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py

BPM-free frame-level tablature + onset training script.

This script intentionally removes the legacy note-level TabEstimator path:
  - no bpm input to the model
  - no note_len input to the model
  - no note_pred target/loss
  - no note-level decimation

Training targets:
  - frame_tab:   (T, 6, 21), one-hot fret/rest class per string per frame
  - frame_onset: (T, 6), binary onset label per string per frame

The onset branch is a non-causal gated TCN head. It returns raw logits and is
trained with BCEWithLogitsLoss inside CustomLoss.

If frame_onset is not present in the NPZ, it is derived from frame_tab by
marking a string onset whenever the active fret changes from rest/different fret
to a played fret.
"""

import argparse
import datetime
import glob
import os
import random
import re
import shutil
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch_optimizer as optim
import yaml
from torch.utils.data import Dataset

try:
    import tensorboardX
except Exception:  # pragma: no cover
    tensorboardX = None

from network import TabEstimator, CustomLoss, REST_CLASS


# -----------------------------------------------------------------------------
# Dataset/path helpers
# -----------------------------------------------------------------------------


def dataset_name_from_dir(dataset_dir):
    return os.path.basename(os.path.normpath(dataset_dir)).lower()


def default_npz_dir(dataset_dir, note_resolution, quantized):
    dataset_name = dataset_name_from_dir(dataset_dir)
    if quantized:
        return os.path.join("data", "npz", f"auto_quantized_{note_resolution}_{dataset_name}", "split")
    return os.path.join("data", "npz", dataset_name, "split")


def npz_tag_from_path(npz_dir):
    p = os.path.normpath(npz_dir)
    parts = p.split(os.sep)
    if len(parts) >= 2 and parts[-1] == "split":
        return parts[-2]
    return os.path.basename(p).lower()


def safe_run_name(s):
    s = str(s)
    s = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)
    return s.strip("_") or "run"


# -----------------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------------


def _epoch_number_from_checkpoint(path):
    match = re.search(r"epoch(\d+)\.model$", os.path.basename(path))
    return int(match.group(1)) if match else -1


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint is not a state_dict-like mapping: {type(checkpoint)}")
    cleaned = {}
    for key, value in checkpoint.items():
        key = str(key)
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    return cleaned


def resolve_pretrained_model_path(pretrained_model, model_root, test_num=None):
    if pretrained_model is None:
        return None

    raw = str(pretrained_model)
    path = os.path.expanduser(raw)

    if os.path.isfile(path):
        return path

    if os.path.isdir(path):
        candidates = glob.glob(os.path.join(path, "*.model"))
        if not candidates:
            candidates = glob.glob(os.path.join(path, "**", "*.model"), recursive=True)
        if candidates:
            return sorted(candidates, key=lambda p: (_epoch_number_from_checkpoint(p), os.path.getmtime(p)))[-1]

    candidates = []
    if test_num is not None:
        candidates.extend(glob.glob(os.path.join(model_root, "*", raw, f"testNo{test_num:02d}", "*.model")))
    candidates.extend(glob.glob(os.path.join(model_root, "*", raw, "**", "*.model"), recursive=True))

    if candidates:
        return sorted(candidates, key=lambda p: (_epoch_number_from_checkpoint(p), os.path.getmtime(p)))[-1]

    raise FileNotFoundError(f"Could not resolve pretrained model: {pretrained_model}")


def load_pretrained_weights(model, checkpoint_path, strict=True):
    state = unwrap_state_dict(safe_torch_load(checkpoint_path, map_location="cpu"))

    if strict:
        model.load_state_dict(state, strict=True)
        print(f"Loaded pretrained checkpoint strictly: {checkpoint_path}")
        return

    model_state = model.state_dict()
    compatible = {
        k: v for k, v in state.items()
        if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)
    }
    skipped = [k for k in state.keys() if k not in compatible]
    model_state.update(compatible)
    model.load_state_dict(model_state, strict=True)

    print(f"Loaded pretrained checkpoint partially: {checkpoint_path}")
    print(f"Loaded tensors: {len(compatible)}")
    print(f"Skipped tensors: {len(skipped)}")
    if skipped:
        print("First skipped tensors:")
        for key in skipped[:20]:
            print("  ", key)


# -----------------------------------------------------------------------------
# Target helpers
# -----------------------------------------------------------------------------


def derive_frame_onsets_from_frame_tab(frame_tab, rest_class=REST_CLASS):
    """
    Derive per-string onset labels from frame_tab.

    frame_tab: (T, 6, 21), one-hot or probability-like
    output:    (T, 6), binary
    """
    frame_tab = np.asarray(frame_tab)
    if frame_tab.ndim != 3 or frame_tab.shape[1] != 6:
        raise ValueError(f"Expected frame_tab shape (T, 6, 21), got {frame_tab.shape}")

    classes = np.argmax(frame_tab, axis=2)
    onset = np.zeros(classes.shape, dtype=np.float32)

    for s in range(6):
        prev = int(rest_class)
        for t in range(classes.shape[0]):
            cur = int(classes[t, s])
            if cur != rest_class and cur != prev:
                onset[t, s] = 1.0
            prev = cur

    return onset


def load_frame_onset_from_npz(data, frame_tab):
    for key in ["frame_onset", "frame_onsets", "onset", "onsets", "frame_tab_onset"]:
        if key not in data.files:
            continue
        x = np.asarray(data[key]).astype(np.float32)
        if x.ndim == 3:
            # Either (T, 6, 21) or similar. Collapse fret dimension.
            x = np.max(x[:, :, :REST_CLASS], axis=2)
        if x.ndim != 2 or x.shape[1] != 6:
            raise ValueError(f"NPZ onset key {key!r} has unsupported shape {x.shape}")
        return x
    return derive_frame_onsets_from_frame_tab(frame_tab)


# -----------------------------------------------------------------------------
# Dataset and collation
# -----------------------------------------------------------------------------


class FrameOnsetDataset(Dataset):
    def __init__(
        self,
        data_list,
        input_feature_type,
        use_hand_position=False,
        hand_pos_dim=20,
        allow_missing_hand_pos=False,
    ):
        self.data_list = list(data_list)
        self.input_feature_type = str(input_feature_type)
        self.use_hand_position = bool(use_hand_position)
        self.hand_pos_dim = int(hand_pos_dim)
        self.allow_missing_hand_pos = bool(allow_missing_hand_pos)

    def __len__(self):
        return len(self.data_list)

    def _missing_hand_pos(self, length):
        return np.ones((length, self.hand_pos_dim), dtype=np.float32) / float(self.hand_pos_dim)

    def __getitem__(self, index):
        npz_path = self.data_list[index]
        data = np.load(npz_path, allow_pickle=True)

        if self.input_feature_type == "cqt":
            input_features = data["cqt"].astype(np.float32)
        elif self.input_feature_type == "melspec":
            input_features = data["mel_spec"].astype(np.float32)
        else:
            raise ValueError(f"Unknown input_feature_type: {self.input_feature_type}")

        if "frame_tab" not in data.files:
            raise KeyError(f"{npz_path} does not contain frame_tab")

        frame_tab = data["frame_tab"].astype(np.float32)
        frame_onset = load_frame_onset_from_npz(data, frame_tab).astype(np.float32)

        frame_len = int(input_features.shape[0])

        # Match target length to input feature length if the dataset is slightly inconsistent.
        target_len = min(frame_len, frame_tab.shape[0], frame_onset.shape[0])
        input_features = input_features[:target_len]
        frame_tab = frame_tab[:target_len]
        frame_onset = frame_onset[:target_len]
        frame_len = int(target_len)

        frame_hand_pos = None
        if self.use_hand_position:
            if "frame_hand_pos" in data.files:
                frame_hand_pos = data["frame_hand_pos"].astype(np.float32)[:target_len]
            elif self.allow_missing_hand_pos:
                frame_hand_pos = self._missing_hand_pos(frame_len)
            else:
                raise KeyError(
                    f"{npz_path} does not contain frame_hand_pos. "
                    "Use a hand-position NPZ split or --allow-missing-hand-pos for debugging."
                )

            if frame_hand_pos.shape[-1] != self.hand_pos_dim:
                raise ValueError(
                    f"{npz_path}: frame_hand_pos dim is {frame_hand_pos.shape[-1]}, "
                    f"expected {self.hand_pos_dim}."
                )

        return input_features, frame_tab, frame_onset, frame_len, frame_hand_pos


def _pad_2d(x, target_len):
    pad_len = target_len - x.shape[0]
    return np.pad(x, [(0, pad_len), (0, 0)], mode="constant")


def _pad_3d(x, target_len):
    pad_len = target_len - x.shape[0]
    return np.pad(x, [(0, pad_len), (0, 0), (0, 0)], mode="constant")


def frame_onset_collate(batch):
    input_features, frame_tab, frame_onset, frame_len, frame_hand_pos = zip(*batch)

    frame_len = np.asarray(frame_len, dtype=np.int64)
    max_len = int(max(frame_len))
    sort_idx = np.argsort(frame_len)[::-1]

    input_out = np.asarray([_pad_2d(x, max_len) for x in input_features], dtype=np.float32)
    tab_out = np.asarray([_pad_3d(x, max_len) for x in frame_tab], dtype=np.float32)
    onset_out = np.asarray([_pad_2d(x, max_len) for x in frame_onset], dtype=np.float32)

    input_out = np.take(input_out, sort_idx, axis=0)
    tab_out = np.take(tab_out, sort_idx, axis=0)
    onset_out = np.take(onset_out, sort_idx, axis=0)
    frame_len = np.take(frame_len, sort_idx, axis=0)

    any_hand = any(x is not None for x in frame_hand_pos)
    if any_hand:
        if not all(x is not None for x in frame_hand_pos):
            raise ValueError("Some batch items have frame_hand_pos and others do not.")
        hand_out = np.asarray([_pad_2d(x, max_len) for x in frame_hand_pos], dtype=np.float32)
        hand_out = np.take(hand_out, sort_idx, axis=0)
        hand_tensor = torch.from_numpy(hand_out)
    else:
        hand_tensor = None

    return (
        torch.from_numpy(input_out),
        torch.from_numpy(tab_out),
        torch.from_numpy(onset_out),
        torch.from_numpy(frame_len),
        hand_tensor,
    )


# -----------------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------------


def config_bool(config, key, default=False):
    return bool(config[key]) if key in config else bool(default)


def config_int(config, key, default):
    return int(config[key]) if key in config else int(default)


def config_float(config, key, default):
    return float(config[key]) if key in config else float(default)


def config_str(config, key, default):
    return str(config[key]) if key in config else str(default)


def build_tab_estimator(
    mode,
    input_feature_type,
    encoder_type,
    use_custom_decimation_func,
    use_conv_stack,
    n_bins,
    hop_length,
    sr,
    encoder_heads,
    encoder_layers,
    use_hand_position=False,
    hand_pos_dim=20,
    hand_position_fusion="hidden+prior",
    hand_hidden_gate_init=0.5,
    hand_prior_strength=0.35,
    hand_span=4,
    onset_hidden_dim=64,
    onset_dropout=0.25,
    onset_kernel_size=5,
    onset_tcn_levels=4,
):
    return TabEstimator(
        mode=mode,
        encoder_type=encoder_type,
        use_custom_decimation_func=False,
        use_conv_stack=use_conv_stack,
        n_bins=n_bins,
        hop_length=hop_length,
        sr=sr,
        encoder_heads=encoder_heads,
        encoder_layers=encoder_layers,
        use_hand_position=use_hand_position,
        hand_pos_dim=hand_pos_dim,
        hand_position_fusion=hand_position_fusion,
        hand_hidden_gate_init=hand_hidden_gate_init,
        hand_prior_strength=hand_prior_strength,
        hand_span=hand_span,
        onset_hidden_dim=onset_hidden_dim,
        onset_dropout=onset_dropout,
        onset_kernel_size=onset_kernel_size,
        onset_tcn_levels=onset_tcn_levels,
    )


def model_forward(model, padded_input_features, frame_len, frame_hand_pos=None):
    return model(
        padded_input_features.float(),
        frame_len,
        frame_hand_pos=frame_hand_pos,
    )


def get_encoder_attention(model, encoder_layers):
    try:
        attn_map = model.encoder.encoders._modules["0"]._modules["self_attn"].attn
        for n_layer in range(1, int(encoder_layers)):
            attn_map = torch.cat(
                (
                    attn_map,
                    model.encoder.encoders._modules[f"{n_layer}"]._modules["self_attn"].attn,
                ),
                dim=0,
            )
        return attn_map
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


class DummyWriter:
    def add_scalar(self, *args, **kwargs):
        pass
    def close(self):
        pass


def make_fold_lists(data_list, test_num, train_ratio, seed_value, n_folds):
    rng = random.Random(seed_value + test_num)

    if n_folds <= 1:
        data = list(data_list)
        rng.shuffle(data)
        split_idx = int(round(len(data) * train_ratio))
        return data[:split_idx], data[split_idx:]

    fold_prefix = f"{test_num:02d}_"
    dev_data_list = [p for p in data_list if not os.path.split(p)[1].startswith(fold_prefix)]

    if len(dev_data_list) == len(data_list):
        data = list(data_list)
        rng.shuffle(data)
        split_idx = int(round(len(data) * train_ratio))
        return data[:split_idx], data[split_idx:]

    rng.shuffle(dev_data_list)
    split_idx = int(round(len(dev_data_list) * train_ratio))
    return dev_data_list[:split_idx], dev_data_list[split_idx:]


def freeze_model_encoder(model):
    for param in model.encoder.parameters():
        param.requires_grad = False
    print("Encoder frozen.")


def freeze_audio_frontend(model):
    if hasattr(model, "convstack"):
        for param in model.convstack.parameters():
            param.requires_grad = False
        print("ConvStack frozen.")


def run_epoch(model, loader, criterion, optimizer, device, train_mode=True, freeze_encoder=False, freeze_frontend=False):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        input_features, frame_tab, frame_onset, frame_len, frame_hand_pos = batch
        input_features = input_features.to(device)
        frame_tab = frame_tab.to(device)
        frame_onset = frame_onset.to(device)
        frame_len = frame_len.to(device)
        if frame_hand_pos is not None:
            frame_hand_pos = frame_hand_pos.to(device)

        if train_mode:
            if freeze_encoder:
                model.encoder.eval()
            if freeze_frontend and hasattr(model, "convstack"):
                model.convstack.eval()
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            frame_tab_pred, frame_onset_pred, olens = model_forward(
                model,
                input_features,
                frame_len,
                frame_hand_pos=frame_hand_pos,
            )
            loss = criterion(frame_tab_pred, frame_tab, frame_onset_pred, frame_onset, olens)

            if train_mode:
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(1, n_batches)


def train(
    config,
    args,
    test_num,
    train_data_list,
    valid_data_list,
    tensorboard_dir,
    model_dir,
):
    hop_length = int(config["hop_length"])
    sr = int(config["down_sampling_rate"])
    cqt_n_bins = int(config["cqt_n_bins"])
    encoder_heads = int(config["encoder_heads"])
    encoder_layers = int(config["encoder_layers"])
    input_feature_type = str(config["input_feature_type"])
    mode = str(config["mode"])
    encoder_type = str(config["encoder_type"])
    use_conv_stack = bool(config["use_conv_stack"])

    if mode != "tab":
        raise ValueError("This BPM-free train.py supports config mode: tab only.")

    n_bins = cqt_n_bins if input_feature_type == "cqt" else 128

    use_hand_position = bool(args.use_hand_position or config_bool(config, "use_hand_position", False))
    hand_pos_dim = int(args.hand_pos_dim if args.hand_pos_dim is not None else config_int(config, "hand_pos_dim", 20))
    hand_position_fusion = str(args.hand_position_fusion if args.hand_position_fusion is not None else config_str(config, "hand_position_fusion", "hidden+prior"))
    hand_hidden_gate_init = float(args.hand_hidden_gate_init if args.hand_hidden_gate_init is not None else config_float(config, "hand_hidden_gate_init", 0.5))
    hand_prior_strength = float(args.hand_prior_strength if args.hand_prior_strength is not None else config_float(config, "hand_prior_strength", 0.35))
    hand_span = int(args.hand_span if args.hand_span is not None else config_int(config, "hand_span", 4))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise EnvironmentError("CUDA was requested but is not available")

    model = build_tab_estimator(
        mode=mode,
        input_feature_type=input_feature_type,
        encoder_type=encoder_type,
        use_custom_decimation_func=False,
        use_conv_stack=use_conv_stack,
        n_bins=n_bins,
        hop_length=hop_length,
        sr=sr,
        encoder_heads=encoder_heads,
        encoder_layers=encoder_layers,
        use_hand_position=use_hand_position,
        hand_pos_dim=hand_pos_dim,
        hand_position_fusion=hand_position_fusion,
        hand_hidden_gate_init=hand_hidden_gate_init,
        hand_prior_strength=hand_prior_strength,
        hand_span=hand_span,
        onset_hidden_dim=args.onset_hidden_dim,
        onset_dropout=args.onset_dropout,
        onset_kernel_size=args.onset_kernel_size,
        onset_tcn_levels=args.onset_tcn_levels,
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    pretrained_model_path = resolve_pretrained_model_path(args.pretrained_model, args.model_root, test_num=test_num) if args.pretrained_model else None
    if pretrained_model_path:
        load_pretrained_weights(model, pretrained_model_path, strict=not args.pretrained_allow_partial)

    if args.freeze_frontend:
        freeze_audio_frontend(model)
    if args.freeze_encoder:
        freeze_model_encoder(model)

    model.to(device)

    criterion = CustomLoss(
        onset_loss_weight=args.onset_loss_weight,
        onset_positive_weight=args.onset_positive_weight,
        tab_loss_weight=args.tab_loss_weight,
    ).to(device)

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters left after freezing.")

    optimizer = optim.RAdam(trainable_parameters, lr=float(args.lr if args.lr is not None else config["lr"]))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 32, gamma=0.5)

    train_dataset = FrameOnsetDataset(
        train_data_list,
        input_feature_type=input_feature_type,
        use_hand_position=use_hand_position,
        hand_pos_dim=hand_pos_dim,
        allow_missing_hand_pos=args.allow_missing_hand_pos,
    )
    valid_dataset = FrameOnsetDataset(
        valid_data_list,
        input_feature_type=input_feature_type,
        use_hand_position=use_hand_position,
        hand_pos_dim=hand_pos_dim,
        allow_missing_hand_pos=args.allow_missing_hand_pos,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=frame_onset_collate,
        num_workers=args.n_cores,
        pin_memory=args.pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=frame_onset_collate,
        num_workers=args.n_cores,
        pin_memory=args.pin_memory,
    )

    writer = tensorboardX.SummaryWriter(tensorboard_dir) if tensorboardX is not None else DummyWriter()

    os.makedirs(model_dir, exist_ok=True)
    max_epochs = int(args.epoch if args.epoch is not None else config["epoch"])

    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train_mode=True,
            freeze_encoder=args.freeze_encoder,
            freeze_frontend=args.freeze_frontend,
        )
        valid_loss = run_epoch(
            model,
            valid_loader,
            criterion,
            optimizer,
            device,
            train_mode=False,
        )

        scheduler.step()

        print(f"Epoch {epoch:03d} | train loss: {train_loss:.4f} | valid loss: {valid_loss:.4f}")
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("valid/loss", valid_loss, epoch)

        if epoch % int(args.save_every) == 0 or epoch == max_epochs:
            ckpt_path = os.path.join(model_dir, f"epoch{epoch}.model")
            torch.save(model.state_dict(), ckpt_path)
            print("Saved checkpoint:", ckpt_path)

    writer.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="BPM-free frame-tab + onset training")

    parser.add_argument("--dataset-dir", default="GuitarSet")
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--npz-dir", default=None)
    parser.add_argument("--model-root", default="model")
    parser.add_argument("--tensorboard-root", default="tensorboard")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--config", default="src/config.yaml")
    parser.add_argument("--n-folds", type=int, default=6)
    parser.add_argument("--test-num", type=int, default=None)
    parser.add_argument("--n-cores", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--pin-memory", action="store_true")

    parser.add_argument("--pretrained-model", default=None)
    parser.add_argument("--pretrained-allow-partial", action="store_true")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--freeze-frontend", action="store_true")

    parser.add_argument("--use-hand-position", action="store_true")
    parser.add_argument("--allow-missing-hand-pos", action="store_true")
    parser.add_argument("--hand-pos-dim", type=int, default=None)
    parser.add_argument("--hand-position-fusion", choices=["hidden", "prior", "hidden+prior", "none"], default=None)
    parser.add_argument("--hand-hidden-gate-init", type=float, default=None)
    parser.add_argument("--hand-prior-strength", type=float, default=None)
    parser.add_argument("--hand-span", type=int, default=None)

    parser.add_argument("--tab-loss-weight", type=float, default=1.0)
    parser.add_argument("--onset-loss-weight", type=float, default=0.25)
    parser.add_argument("--onset-positive-weight", type=float, default=10.0)
    parser.add_argument("--onset-hidden-dim", type=int, default=64, help="Channels per onset TCN layer.")
    parser.add_argument("--onset-dropout", type=float, default=0.25)
    parser.add_argument("--onset-kernel-size", type=int, default=5, help="Odd kernel size for the non-causal onset TCN.")
    parser.add_argument("--onset-tcn-levels", type=int, default=4, help="Number of dilated gated TCN blocks in the onset head.")

    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    note_resolution = int(config["note_resolution"])
    train_ratio = float(config["train_ratio"])
    seed_value = int(args.seed if args.seed is not None else config["seed_"])

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

    npz_dir = args.npz_dir or default_npz_dir(args.dataset_dir, note_resolution, args.quantized)
    data_list = np.array(sorted(glob.glob(os.path.join(npz_dir, "*.npz"), recursive=True)))
    if len(data_list) == 0:
        raise FileNotFoundError(f"No NPZ files found at: {os.path.join(npz_dir, '*.npz')}")

    run_tag = npz_tag_from_path(npz_dir)
    run_name = safe_run_name(args.run_name) if args.run_name else datetime.datetime.now().strftime("%Y%m%d%H%M")

    base_tensorboard_dir = os.path.join(args.tensorboard_root, run_tag, run_name)
    base_model_dir = os.path.join(args.model_root, run_tag, run_name)
    os.makedirs(base_tensorboard_dir, exist_ok=True)
    os.makedirs(base_model_dir, exist_ok=True)

    shutil.copyfile(args.config, os.path.join(base_model_dir, "config.yaml"))

    metadata = {
        "architecture": "bpm_free_frame_tab_onset",
        "npz_dir": npz_dir,
        "data_path": os.path.join(npz_dir, "*.npz"),
        "num_npz_files": int(len(data_list)),
        "run_tag": run_tag,
        "run_name": run_name,
        "base_model_dir": base_model_dir,
        "base_tensorboard_dir": base_tensorboard_dir,
        "uses_bpm": False,
        "uses_note_pred": False,
        "outputs": ["frame_tab_pred", "frame_onset_pred"],
        "use_hand_position": bool(args.use_hand_position or config_bool(config, "use_hand_position", False)),
        "hand_pos_dim": int(args.hand_pos_dim if args.hand_pos_dim is not None else config_int(config, "hand_pos_dim", 20)),
        "hand_position_fusion": str(args.hand_position_fusion if args.hand_position_fusion is not None else config_str(config, "hand_position_fusion", "hidden+prior")),
        "hand_prior_strength": float(args.hand_prior_strength if args.hand_prior_strength is not None else config_float(config, "hand_prior_strength", 0.35)),
        "hand_span": int(args.hand_span if args.hand_span is not None else config_int(config, "hand_span", 4)),
        "onset_loss_weight": float(args.onset_loss_weight),
        "onset_positive_weight": float(args.onset_positive_weight),
        "onset_head": "noncausal_gated_tcn",
        "onset_hidden_dim": int(args.onset_hidden_dim),
        "onset_dropout": float(args.onset_dropout),
        "onset_kernel_size": int(args.onset_kernel_size),
        "onset_tcn_levels": int(args.onset_tcn_levels),
        "args": vars(args),
    }
    with open(os.path.join(base_model_dir, "run_metadata.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f)

    print("npz_dir:", npz_dir)
    print("npz files:", len(data_list))
    print("model output:", base_model_dir)
    print("tensorboard output:", base_tensorboard_dir)
    print("architecture: bpm_free_frame_tab_onset_noncausal_tcn")

    n_folds = int(args.n_folds)
    if args.test_num is not None:
        if args.test_num < 0 or args.test_num >= n_folds:
            raise ValueError(f"--test-num must be between 0 and {n_folds - 1}")
        test_nums = [args.test_num]
    else:
        test_nums = range(n_folds)

    for test_num in test_nums:
        train_data_list, valid_data_list = make_fold_lists(
            data_list,
            test_num=test_num,
            train_ratio=train_ratio,
            seed_value=seed_value,
            n_folds=n_folds,
        )

        if len(train_data_list) == 0:
            print(f"[skip fold {test_num}] no training data")
            continue
        if len(valid_data_list) == 0:
            print(f"[warn fold {test_num}] no validation data; using last training item as validation")
            valid_data_list = train_data_list[-1:]
            train_data_list = train_data_list[:-1]

        tensorboard_dir = os.path.join(base_tensorboard_dir, f"testNo{test_num:02d}")
        model_dir = os.path.join(base_model_dir, f"testNo{test_num:02d}")

        print("\n" + "=" * 80)
        print(f"Fold {test_num}/{n_folds - 1}")
        print("train files:", len(train_data_list))
        print("valid files:", len(valid_data_list))
        print("model_dir:", model_dir)
        print("=" * 80)

        train(
            config=config,
            args=args,
            test_num=test_num,
            train_data_list=train_data_list,
            valid_data_list=valid_data_list,
            tensorboard_dir=tensorboard_dir,
            model_dir=model_dir,
        )


if __name__ == "__main__":
    main()
