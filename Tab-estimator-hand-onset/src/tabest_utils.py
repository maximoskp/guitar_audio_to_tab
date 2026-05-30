"""
tabest_utils.py

Utility helpers for running the hand-conditioned TabEstimator model from FastAPI.

Expected layout, if this file lives in backend/:

    backend/
    ├── fastAPI.py
    ├── tabest_utils.py
    └── Tab-estimator-hand-light/
        ├── src/
        │   └── network.py
        └── model/
            └── guitarset_phantom_handpos/
                └── guitarset_handpos/
                    ├── config.yaml
                    ├── run_metadata.yaml      # optional but recommended
                    └── testNo00/
                        └── epoch128.model

This module intentionally does not import FastAPI. It can be imported from a
backend endpoint or tested directly from Python.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import torch
import yaml


# =============================================================================
# TabEstimator repo/model paths
# =============================================================================


_THIS_DIR = Path(__file__).resolve().parent

# Allow override from environment if you ever move the light repo elsewhere:
#   set TABEST_ROOT=C:\path\to\Tab-estimator-hand-light
#   export TABEST_ROOT=/path/to/Tab-estimator-hand-light
TABEST_ROOT = Path(os.environ.get("TABEST_ROOT", _THIS_DIR / "Tab-estimator-hand-light")).resolve()
TABEST_SRC = TABEST_ROOT / "src"
DEFAULT_TABEST_MODEL_ROOT = TABEST_ROOT / "model"

# Optional BPM-free hand+onset TabEstimator repo. This is loaded lazily so the
# old hand endpoint keeps working even if the onset repo is absent.
TABEST_ONSET_ROOT = Path(os.environ.get("TABEST_ONSET_ROOT", _THIS_DIR / "Tab-estimator-hand-onset-light")).resolve()
TABEST_ONSET_SRC = TABEST_ONSET_ROOT / "src"
DEFAULT_TABEST_ONSET_MODEL_ROOT = TABEST_ONSET_ROOT / "model"

# Optional hidden-onset hand TabEstimator repo. This is loaded lazily so the
# old hand and BPM-free onset endpoints keep working even if the hidonset repo
# is absent.
TABEST_HIDONSET_ROOT = Path(os.environ.get("TABEST_HIDONSET_ROOT", _THIS_DIR / "Tab-estimator-hand-hidonset-light")).resolve()
TABEST_HIDONSET_SRC = TABEST_HIDONSET_ROOT / "src"
DEFAULT_TABEST_HIDONSET_MODEL_ROOT = TABEST_HIDONSET_ROOT / "model"

if not TABEST_SRC.is_dir():
    raise ImportError(
        "Could not find TabEstimator source directory. Expected:\n"
        f"  {TABEST_SRC}\n"
        "Either place Tab-estimator-hand-light next to tabest_utils.py, or set TABEST_ROOT."
    )

if str(TABEST_SRC) not in sys.path:
    sys.path.insert(0, str(TABEST_SRC))

try:
    from network import TabEstimator
except Exception as exc:
    raise ImportError(
        "Could not import TabEstimator from network.py. Checked:\n"
        f"  {TABEST_SRC}\n"
        f"Original error: {repr(exc)}"
    ) from exc


# =============================================================================
# Constants / cache
# =============================================================================


TABESTIMATOR_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}
TABESTIMATOR_ONSET_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}
TABESTIMATOR_HIDONSET_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}
TABESTIMATOR_REST_CLASS = 20  # current TabEstimator: frets 0..19, class 20 = not played


# =============================================================================
# Basic file/config helpers
# =============================================================================



def _as_path_or_none(value: Optional[str | os.PathLike]) -> Optional[Path]:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _safe_torch_load(path: str | os.PathLike, map_location: str = "cpu"):
    """
    torch.load wrapper that supports both older and newer PyTorch.
    """
    path = str(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _read_yaml_if_exists(path: Optional[str | os.PathLike]) -> Dict[str, Any]:
    if path is None:
        return {}

    path = Path(path)
    if not path.is_file():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)

    return obj or {}


def _get_config_required(config: Dict[str, Any], key: str):
    if key not in config:
        raise KeyError(f"Missing required config key: {key!r}")
    return config[key]


def _bool_from_config(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _finite_float_or_none(value: Any) -> Optional[float]:
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return None


# =============================================================================
# Model path resolution
# =============================================================================


def _resolve_tabestimator_paths(
    model_run: Optional[str] = None,
    epoch: Optional[int] = None,
    test_num: int = 0,
    model_root: Optional[str | os.PathLike] = None,
    checkpoint_path: Optional[str | os.PathLike] = None,
    config_path: Optional[str | os.PathLike] = None,
) -> Tuple[str, str, Optional[str]]:
    """
    Resolve config/checkpoint/metadata paths.

    Supported styles
    ----------------

    A. Run name under default model root:

        model_run="guitarset_phantom_handpos/guitarset_handpos"
        epoch=128
        test_num=0

    B. Run name under explicit model root.

    C. model_run is itself a directory.

    D. Explicit checkpoint_path/config_path.
    """
    fold_id = f"{int(test_num):02d}"

    model_root_path = Path(model_root).expanduser().resolve() if model_root else DEFAULT_TABEST_MODEL_ROOT

    run_dir: Optional[Path] = None

    if model_run:
        candidate = Path(model_run).expanduser()
        if candidate.is_dir():
            run_dir = candidate.resolve()
        else:
            run_dir = (model_root_path / model_run).resolve()

    checkpoint = _as_path_or_none(checkpoint_path)
    config = _as_path_or_none(config_path)

    if checkpoint is None:
        if run_dir is None:
            raise ValueError("Either model_run or checkpoint_path must be provided.")
        if epoch is None:
            raise ValueError("epoch is required when checkpoint_path is not provided.")
        checkpoint = run_dir / f"testNo{fold_id}" / f"epoch{int(epoch)}.model"

    if config is None:
        if run_dir is not None:
            config = run_dir / "config.yaml"
        else:
            try:
                config = checkpoint.parent.parent / "config.yaml"
            except Exception:
                raise ValueError("config_path is required when it cannot be inferred from checkpoint_path.")

    metadata_path: Optional[Path] = None
    if run_dir is not None:
        metadata_path = run_dir / "run_metadata.yaml"
    elif checkpoint is not None:
        metadata_path = checkpoint.parent.parent / "run_metadata.yaml"

    if not config.is_file():
        raise FileNotFoundError(
            "TabEstimator config not found:\n"
            f"  {config}\n"
            "Expected config.yaml in the model run directory."
        )

    if not checkpoint.is_file():
        raise FileNotFoundError(
            "TabEstimator checkpoint not found:\n"
            f"  {checkpoint}\n"
            "Check model_run, epoch, and test_num."
        )

    return str(config), str(checkpoint), str(metadata_path) if metadata_path and metadata_path.is_file() else None


# =============================================================================
# Model loading
# =============================================================================


def load_tabestimator_hand_model(
    model_run: Optional[str] = None,
    epoch: Optional[int] = None,
    test_num: int = 0,
    model_root: Optional[str | os.PathLike] = None,
    checkpoint_path: Optional[str | os.PathLike] = None,
    config_path: Optional[str | os.PathLike] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load and cache a hand-conditioned TabEstimator model.
    """
    if model_root is None:
        model_root = str(TABEST_ROOT / "model")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config_path_resolved, checkpoint_path_resolved, metadata_path = _resolve_tabestimator_paths(
        model_run=model_run,
        epoch=epoch,
        test_num=test_num,
        model_root=model_root,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )

    cache_key = f"{checkpoint_path_resolved}|{config_path_resolved}|{device}"
    if cache_key in TABESTIMATOR_MODEL_CACHE:
        return TABESTIMATOR_MODEL_CACHE[cache_key]

    config = _read_yaml_if_exists(config_path_resolved)
    metadata = _read_yaml_if_exists(metadata_path)

    mode = str(_get_config_required(config, "mode"))
    if mode != "tab":
        raise ValueError(f"TabEstimator endpoint requires mode='tab', got {mode!r}")

    input_feature_type = str(_get_config_required(config, "input_feature_type"))
    if input_feature_type == "cqt":
        n_bins = int(_get_config_required(config, "cqt_n_bins"))
    elif input_feature_type == "melspec":
        n_bins = 128
    else:
        raise ValueError(f"Unknown input_feature_type: {input_feature_type!r}")

    # Prefer metadata because it should reflect exactly how the checkpoint model was constructed.
    use_hand_position = _bool_from_config(
        metadata.get("use_hand_position", config.get("use_hand_position", True)),
        default=True,
    )
    hand_pos_dim = int(metadata.get("hand_pos_dim", config.get("hand_pos_dim", 20)))
    hand_position_fusion = str(metadata.get("hand_position_fusion", config.get("hand_position_fusion", "hidden+prior")))
    hand_prior_strength = float(metadata.get("hand_prior_strength", config.get("hand_prior_strength", 0.35)))
    hand_hidden_gate_init = float(metadata.get("hand_hidden_gate_init", config.get("hand_hidden_gate_init", 0.5)))
    hand_span = int(metadata.get("hand_span", config.get("hand_span", 4)))
    note_target_length = int(metadata.get("note_target_length", config.get("note_target_length", 64)))

    try:
        model = TabEstimator(
            mode,
            str(_get_config_required(config, "encoder_type")),
            _bool_from_config(_get_config_required(config, "use_custom_decimation_func")),
            _bool_from_config(_get_config_required(config, "use_conv_stack")),
            n_bins,
            int(_get_config_required(config, "hop_length")),
            int(_get_config_required(config, "down_sampling_rate")),
            encoder_heads=int(_get_config_required(config, "encoder_heads")),
            encoder_layers=int(_get_config_required(config, "encoder_layers")),
            use_hand_position=use_hand_position,
            hand_pos_dim=hand_pos_dim,
            hand_position_fusion=hand_position_fusion,
            hand_hidden_gate_init=hand_hidden_gate_init,
            hand_prior_strength=hand_prior_strength,
            hand_span=hand_span,
            note_target_length=note_target_length,
        )
    except TypeError as exc:
        raise TypeError(
            "Your Tab-estimator-hand-light/src/network.py does not appear to match the "
            "hand-conditioned TabEstimator constructor expected by this backend. "
            "Make sure network.py accepts arguments like use_hand_position, hand_pos_dim, "
            "hand_position_fusion, hand_prior_strength, and note_target_length.\n"
            f"Original error: {repr(exc)}"
        ) from exc

    state_dict = _safe_torch_load(checkpoint_path_resolved, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if isinstance(state_dict, dict):
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            new_key = key[7:] if str(key).startswith("module.") else key
            cleaned_state_dict[new_key] = value
        state_dict = cleaned_state_dict

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    bundle = {
        "model": model,
        "config": config,
        "metadata": metadata,
        "device": device,
        "config_path": config_path_resolved,
        "checkpoint_path": checkpoint_path_resolved,
        "metadata_path": metadata_path,
        "input_feature_type": input_feature_type,
        "hand_pos_dim": hand_pos_dim,
        "note_target_length": note_target_length,
        "use_hand_position": use_hand_position,
        "hand_span": hand_span,
        "hand_position_fusion": hand_position_fusion,
        "tabest_root": str(TABEST_ROOT),
        "tabest_src": str(TABEST_SRC),
        "model_root": str(Path(model_root).expanduser().resolve()),
    }

    TABESTIMATOR_MODEL_CACHE[cache_key] = bundle
    return bundle



# =============================================================================
# BPM-free hand+onset TabEstimator loading and runtime decoding
# =============================================================================


def _load_onset_network_class(tabestimator_root: str | os.PathLike):
    """
    Load TabEstimator from Tab-estimator-hand-onset-light/src/network.py without
    clobbering the already imported legacy hand-conditioned network module.
    """
    root = Path(tabestimator_root).expanduser().resolve()
    network_path = root / "src" / "network.py"

    if not network_path.is_file():
        raise FileNotFoundError(f"Onset network.py not found: {network_path}")

    module_name = "tabestimator_hand_onset_network"
    spec = importlib.util.spec_from_file_location(module_name, str(network_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    return module.TabEstimator


def _candidate_onset_model_roots(model_root=None):
    here = Path(__file__).resolve().parent
    cwd = Path(os.getcwd()).resolve()

    candidates = []
    if model_root:
        candidates.append(Path(model_root))

    candidates.extend(
        [
            here / "Tab-estimator-hand-onset-light" / "model",
            here.parent / "Tab-estimator-hand-onset-light" / "model",
            cwd / "backend_cv" / "Tab-estimator-hand-onset-light" / "model",
            cwd / "backend" / "Tab-estimator-hand-onset-light" / "model",
            cwd / "Tab-estimator-hand-onset-light" / "model",
            DEFAULT_TABEST_ONSET_MODEL_ROOT,
        ]
    )

    out = []
    seen = set()
    for path in candidates:
        norm = str(Path(path).expanduser().resolve())
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def resolve_tabestimator_hand_onset_paths(
    model_run=None,
    epoch=None,
    test_num=0,
    model_root=None,
    checkpoint_path=None,
    config_path=None,
):
    """
    Resolve a BPM-free hand+onset checkpoint under Tab-estimator-hand-onset-light.
    """
    if checkpoint_path and config_path:
        checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        config_path = str(Path(config_path).expanduser().resolve())
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Onset checkpoint not found: {checkpoint_path}")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Onset config not found: {config_path}")
        if model_root:
            resolved_model_root = str(Path(model_root).expanduser().resolve())
        else:
            resolved_model_root = str(Path(config_path).parent.parent.parent.resolve())
        return {
            "checkpoint_path": checkpoint_path,
            "config_path": config_path,
            "metadata_path": str(Path(config_path).parent / "run_metadata.yaml"),
            "model_root": resolved_model_root,
            "tabestimator_root": str(Path(resolved_model_root).parent.resolve()),
        }

    if not model_run:
        raise ValueError("model_run is required unless checkpoint_path/config_path are provided.")
    if epoch is None:
        raise ValueError("epoch is required unless checkpoint_path is provided.")

    fold_id = f"{int(test_num):02d}"
    errors = []

    for root in _candidate_onset_model_roots(model_root):
        run_dir = Path(root) / str(model_run)
        cfg = run_dir / "config.yaml"
        ckpt = run_dir / f"testNo{fold_id}" / f"epoch{int(epoch)}.model"
        metadata = run_dir / "run_metadata.yaml"

        if cfg.is_file() and ckpt.is_file():
            return {
                "checkpoint_path": str(ckpt.resolve()),
                "config_path": str(cfg.resolve()),
                "metadata_path": str(metadata.resolve()) if metadata.is_file() else None,
                "model_root": str(Path(root).resolve()),
                "tabestimator_root": str(Path(root).parent.resolve()),
            }

        errors.append(f"root={root} | config={cfg.is_file()} | checkpoint={ckpt.is_file()}")

    raise FileNotFoundError(
        "Could not resolve hand+onset TabEstimator model.\n"
        f"model_run={model_run}\n"
        f"epoch={epoch}\n"
        f"test_num={test_num}\n"
        "Tried:\n" + "\n".join(errors)
    )


def _metadata_or_config(metadata: Dict[str, Any], config: Dict[str, Any], key: str, default: Any) -> Any:
    if key in metadata and metadata[key] is not None:
        return metadata[key]
    if key in config and config[key] is not None:
        return config[key]
    return default


def load_tabestimator_hand_onset_model(
    model_run=None,
    epoch=None,
    test_num=0,
    model_root=None,
    checkpoint_path=None,
    config_path=None,
    device=None,
):
    """
    Load and cache the BPM-free hand+onset TabEstimator.
    """
    paths = resolve_tabestimator_hand_onset_paths(
        model_run=model_run,
        epoch=epoch,
        test_num=test_num,
        model_root=model_root,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_key = f"{paths['checkpoint_path']}|{paths['config_path']}|{device}"
    if cache_key in TABESTIMATOR_ONSET_MODEL_CACHE:
        return TABESTIMATOR_ONSET_MODEL_CACHE[cache_key]

    with open(paths["config_path"], "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    metadata = _read_yaml_if_exists(paths.get("metadata_path"))

    input_feature_type = str(_get_config_required(config, "input_feature_type"))
    if input_feature_type == "cqt":
        n_bins = int(_get_config_required(config, "cqt_n_bins"))
    elif input_feature_type == "melspec":
        n_bins = 128
    else:
        raise ValueError(f"Unknown input_feature_type: {input_feature_type!r}")

    OnsetTabEstimator = _load_onset_network_class(paths["tabestimator_root"])

    use_hand_position = _bool_from_config(
        _metadata_or_config(metadata, config, "use_hand_position", True),
        default=True,
    )
    hand_pos_dim = int(_metadata_or_config(metadata, config, "hand_pos_dim", 20))
    hand_position_fusion = str(_metadata_or_config(metadata, config, "hand_position_fusion", "hidden+prior"))
    hand_hidden_gate_init = float(_metadata_or_config(metadata, config, "hand_hidden_gate_init", 0.5))
    hand_prior_strength = float(_metadata_or_config(metadata, config, "hand_prior_strength", 0.35))
    hand_span = int(_metadata_or_config(metadata, config, "hand_span", 4))

    model = OnsetTabEstimator(
        str(_get_config_required(config, "mode")),
        str(_get_config_required(config, "encoder_type")),
        _bool_from_config(_get_config_required(config, "use_custom_decimation_func")),
        _bool_from_config(_get_config_required(config, "use_conv_stack")),
        n_bins,
        int(_get_config_required(config, "hop_length")),
        int(_get_config_required(config, "down_sampling_rate")),
        encoder_heads=int(_get_config_required(config, "encoder_heads")),
        encoder_layers=int(_get_config_required(config, "encoder_layers")),
        use_hand_position=use_hand_position,
        hand_pos_dim=hand_pos_dim,
        hand_position_fusion=hand_position_fusion,
        hand_hidden_gate_init=hand_hidden_gate_init,
        hand_prior_strength=hand_prior_strength,
        hand_span=hand_span,
        onset_hidden_dim=int(_metadata_or_config(metadata, config, "onset_hidden_dim", 64)),
        onset_dropout=float(_metadata_or_config(metadata, config, "onset_dropout", 0.25)),
        onset_kernel_size=int(_metadata_or_config(metadata, config, "onset_kernel_size", 3)),
        onset_tcn_levels=int(_metadata_or_config(metadata, config, "onset_tcn_levels", 4)),
        onset_use_raw_features=_bool_from_config(_metadata_or_config(metadata, config, "onset_use_raw_features", True), default=True),
        onset_raw_proj_dim=int(_metadata_or_config(metadata, config, "onset_raw_proj_dim", 64)),
        onset_raw_dropout=float(_metadata_or_config(metadata, config, "onset_raw_dropout", 0.10)),
    )

    state_dict = _safe_torch_load(paths["checkpoint_path"], map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if isinstance(state_dict, dict):
        state_dict = {str(k)[7:] if str(k).startswith("module.") else k: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    bundle = {
        "model": model,
        "config": config,
        "metadata": metadata,
        "device": device,
        "checkpoint_path": paths["checkpoint_path"],
        "config_path": paths["config_path"],
        "metadata_path": paths.get("metadata_path"),
        "model_root": paths["model_root"],
        "tabestimator_root": paths["tabestimator_root"],
        "input_feature_type": input_feature_type,
        "use_hand_position": bool(use_hand_position),
        "hand_pos_dim": int(hand_pos_dim),
        "hand_span": int(hand_span),
        "hand_position_fusion": hand_position_fusion,
    }

    TABESTIMATOR_ONSET_MODEL_CACHE[cache_key] = bundle
    return bundle


def _sigmoid_np(x):
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x))


def _ms_to_frames(ms: float, sr: int, hop_length: int, minimum: int = 0) -> int:
    frame_ms = 1000.0 * float(hop_length) / float(sr)
    if frame_ms <= 0:
        return int(minimum)
    return max(int(minimum), int(round(float(ms) / frame_ms)))


def _peak_picking_1d(
    activations,
    threshold=0.5,
    pre_max=1,
    post_max=1,
    combine=0,
):
    """
    Lightweight madmom-style local-maximum peak picking for one activation curve.
    """
    a = np.asarray(activations, dtype=np.float32).reshape(-1)
    n = int(a.shape[0])
    pre_max = int(max(0, pre_max))
    post_max = int(max(0, post_max))
    combine = int(max(0, combine))

    peaks = []
    for i in range(n):
        value = float(a[i])
        if value < float(threshold):
            continue
        lo = max(0, i - pre_max)
        hi = min(n, i + post_max + 1)
        local_max = float(np.max(a[lo:hi])) if hi > lo else value
        if value < local_max:
            continue
        # In plateaus keep the leftmost peak.
        if peaks and i - peaks[-1] <= combine:
            if value > float(a[peaks[-1]]):
                peaks[-1] = i
            continue
        peaks.append(i)
    return np.asarray(peaks, dtype=np.int64)


def _decode_onset_events_from_global_peaks(
    frame_tab_probs: np.ndarray,
    frame_onset_probs: np.ndarray,
    global_onset_probs: np.ndarray,
    global_peak_frames: Iterable[int],
    sr: int,
    hop_length: int,
    start_time: float = 0.0,
    start_frame: int = 0,
    label_window_ms: float = 50.0,
    label_delay_ms: float = 0.0,
    string_support_window_ms: float = 50.0,
    tab_threshold: float = 0.50,
    string_threshold: float = 0.30,
    rest_class: int = TABESTIMATOR_REST_CLASS,
    use_string_onset_filter: bool = True,
) -> List[Dict[str, Any]]:
    """
    Decode symbolic note events from global onset peaks. Timing comes from the
    global onset head; string/fret labels come from the frame tab head in a short
    post-onset window, with per-string onset probabilities used as optional support.

    If use_string_onset_filter=True, a string is emitted only when either:
      - the per-string onset head supports it, or
      - the frame-tab confidence is very strong.

    If use_string_onset_filter=False, the per-string onset gate is disabled and
    frame-tab confidence alone decides which strings/frets are emitted. This is
    useful for diagnosing missing chord tones.
    """
    frame_tab_probs = np.asarray(frame_tab_probs, dtype=np.float32)
    frame_onset_probs = np.asarray(frame_onset_probs, dtype=np.float32)
    global_onset_probs = np.asarray(global_onset_probs, dtype=np.float32).reshape(-1)

    if frame_tab_probs.ndim != 3 or frame_tab_probs.shape[1] != 6:
        raise ValueError(f"Expected frame_tab_probs shape (T, 6, 21), got {frame_tab_probs.shape}")

    T = int(frame_tab_probs.shape[0])
    seconds_per_frame = float(hop_length) / float(sr)
    label_window = max(0, _ms_to_frames(label_window_ms, sr, hop_length, minimum=0))
    label_delay = max(0, _ms_to_frames(label_delay_ms, sr, hop_length, minimum=0))
    support_window = max(0, _ms_to_frames(string_support_window_ms, sr, hop_length, minimum=0))

    events: List[Dict[str, Any]] = []

    for peak in global_peak_frames:
        peak = int(peak)
        if peak < 0 or peak >= T:
            continue

        label_start = min(T, peak + label_delay)
        label_end = min(T, label_start + label_window + 1)
        support_start = max(0, peak - support_window)
        support_end = min(T, peak + support_window + 1)

        onset_time = float(start_time) + float(peak) * seconds_per_frame
        global_frame = int(start_frame) + int(peak)

        emitted_for_peak = []

        for string_idx in range(6):
            tab_window = frame_tab_probs[label_start:label_end, string_idx, :int(rest_class)]
            if tab_window.size == 0:
                continue

            flat_idx = int(np.argmax(tab_window))
            local_frame, fret = np.unravel_index(flat_idx, tab_window.shape)
            tab_score = float(tab_window[local_frame, fret])

            if frame_onset_probs.ndim == 2 and frame_onset_probs.shape[1] == 6:
                string_score = float(np.max(frame_onset_probs[support_start:support_end, string_idx]))
            else:
                string_score = 0.0

            strong_tab = tab_score >= max(float(tab_threshold) + 0.20, 0.75)
            supported_string = string_score >= float(string_threshold)
            passes_string_filter = (
                not bool(use_string_onset_filter)
                or supported_string
                or strong_tab
            )

            if tab_score >= float(tab_threshold) and passes_string_filter:
                emitted_for_peak.append(
                    {
                        "time": onset_time,
                        "frame": int(peak),
                        "global_frame": int(global_frame),
                        "step": int(peak),
                        "global_step": int(global_frame),
                        "string_index_low_e_first": int(string_idx),
                        "string_number_low_e_first": int(string_idx + 1),
                        "fret": int(fret),
                        "source": "global_onset_window_decoder",
                        "global_onset_prob": float(global_onset_probs[peak]) if peak < len(global_onset_probs) else None,
                        "string_onset_prob": float(string_score),
                        "tab_prob": float(tab_score),
                        "label_frame": int(label_start + local_frame),
                        "label_global_frame": int(start_frame) + int(label_start + local_frame),
                    }
                )

        # Fallback: if the global onset fired but string support was too weak,
        # keep the single strongest non-rest tablature label if it is plausible.
        if not emitted_for_peak:
            tab_window_all = frame_tab_probs[label_start:label_end, :, :int(rest_class)]
            if tab_window_all.size:
                flat_idx = int(np.argmax(tab_window_all))
                local_frame, string_idx, fret = np.unravel_index(flat_idx, tab_window_all.shape)
                tab_score = float(tab_window_all[local_frame, string_idx, fret])
                if tab_score >= max(0.35, float(tab_threshold) * 0.75):
                    emitted_for_peak.append(
                        {
                            "time": onset_time,
                            "frame": int(peak),
                            "global_frame": int(global_frame),
                            "step": int(peak),
                            "global_step": int(global_frame),
                            "string_index_low_e_first": int(string_idx),
                            "string_number_low_e_first": int(string_idx + 1),
                            "fret": int(fret),
                            "source": "global_onset_window_decoder_fallback",
                            "global_onset_prob": float(global_onset_probs[peak]) if peak < len(global_onset_probs) else None,
                            "string_onset_prob": None,
                            "tab_prob": float(tab_score),
                            "label_frame": int(label_start + local_frame),
                            "label_global_frame": int(start_frame) + int(label_start + local_frame),
                        }
                    )

        events.extend(emitted_for_peak)

    return events




def _window_max_1d(values: np.ndarray, center: int, radius: int) -> Tuple[float, int]:
    """Return max value and local frame index in a 1D window around center."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0, int(center)
    center = int(max(0, min(int(center), values.size - 1)))
    radius = int(max(0, radius))
    lo = max(0, center - radius)
    hi = min(values.size, center + radius + 1)
    if hi <= lo:
        return float(values[center]), center
    rel = int(np.argmax(values[lo:hi]))
    idx = lo + rel
    return float(values[idx]), int(idx)


def _event_get_frame(event: Dict[str, Any]) -> int:
    for key in ("frame", "global_frame", "onset_frame", "display_frame"):
        if key in event and event[key] is not None:
            try:
                return int(event[key])
            except Exception:
                pass
    if event.get("time") is not None and event.get("frame_seconds") is not None:
        try:
            return int(round(float(event["time"]) / float(event["frame_seconds"])))
        except Exception:
            pass
    return 0


def _event_get_string(event: Dict[str, Any]) -> int:
    return int(event.get("string_index_low_e_first", event.get("string", -1)))


def _event_get_fret(event: Dict[str, Any]) -> int:
    return int(event.get("fret", 20))


def _event_get_string_onset_prob(event: Dict[str, Any]) -> float:
    for key in ("string_onset_prob", "onset_prob", "event_onset_prob"):
        if key in event and event[key] is not None:
            try:
                return float(event[key])
            except Exception:
                pass
    return 0.0


def _event_get_global_onset_prob(event: Dict[str, Any]) -> float:
    for key in ("global_onset_prob_nearby", "global_onset_prob", "global_prob"):
        if key in event and event[key] is not None:
            try:
                return float(event[key])
            except Exception:
                pass
    return 0.0


def _tab_classes_from_probs(frame_tab_probs: np.ndarray, rest_class: int = 20) -> np.ndarray:
    arr = np.asarray(frame_tab_probs)
    if arr.ndim != 3:
        raise ValueError(f"Expected frame_tab_probs shape (T,6,21), got {arr.shape}")
    return np.argmax(arr, axis=-1).astype(np.int64)


def _string_had_rest_or_fret_change_between(
    tab_classes: np.ndarray,
    string_idx: int,
    fret: int,
    prev_frame: int,
    cur_frame: int,
    rest_class: int = 20,
) -> bool:
    if tab_classes is None:
        return False

    T = int(tab_classes.shape[0])
    if T <= 0:
        return False

    s = int(string_idx)
    if s < 0 or s >= tab_classes.shape[1]:
        return False

    a = max(0, min(int(prev_frame), T - 1))
    b = max(0, min(int(cur_frame), T - 1))

    if b <= a + 1:
        return False

    path = tab_classes[a + 1 : b, s]
    if path.size == 0:
        return False

    return bool(np.any(path != int(fret)))


def _same_string_any_fret_too_close(
    event: Dict[str, Any],
    last_event_by_string: Dict[int, Dict[str, Any]],
    frame_seconds: float,
    same_string_any_fret_min_ms: float,
) -> bool:
    if same_string_any_fret_min_ms is None or float(same_string_any_fret_min_ms) <= 0:
        return False

    s = _event_get_string(event)
    if s not in last_event_by_string:
        return False

    cur_frame = _event_get_frame(event)
    prev_frame = _event_get_frame(last_event_by_string[s])
    dt_ms = abs(cur_frame - prev_frame) * float(frame_seconds) * 1000.0
    return dt_ms < float(same_string_any_fret_min_ms)


def filter_repeated_string_fret_events(
    events: Sequence[Dict[str, Any]],
    frame_tab_probs: np.ndarray,
    frame_seconds: float,
    rest_class: int = 20,
    repeat_same_fret_policy: str = "tab_change_or_strong_onset",
    min_repeat_ms: float = 90.0,
    repeat_onset_threshold: float = 0.90,
    repeat_global_threshold: float = 0.65,
    require_global_for_repeats: bool = True,
    same_string_any_fret_min_ms: float = 35.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    policy = str(repeat_same_fret_policy or "off").lower()
    if policy not in {
        "off",
        "refractory",
        "strong_onset",
        "tab_change_or_strong_onset",
    }:
        raise ValueError(f"Unknown repeat_same_fret_policy: {repeat_same_fret_policy}")

    if policy == "off":
        kept = [dict(ev) for ev in events]
        for ev in kept:
            ev["repeat_filter_action"] = "kept_policy_off"
        return kept, {
            "repeat_filter_policy": policy,
            "repeat_filter_input_events": int(len(events)),
            "repeat_filter_kept_events": int(len(kept)),
            "repeat_filter_suppressed_events": 0,
            "min_repeat_ms": float(min_repeat_ms),
            "repeat_onset_threshold": float(repeat_onset_threshold),
            "repeat_global_threshold": float(repeat_global_threshold),
            "require_global_for_repeats": bool(require_global_for_repeats),
            "same_string_any_fret_min_ms": float(same_string_any_fret_min_ms),
        }

    tab_classes = _tab_classes_from_probs(frame_tab_probs, rest_class=rest_class)
    events_sorted = sorted(
        [dict(ev) for ev in events],
        key=lambda ev: (
            _event_get_frame(ev),
            _event_get_string(ev),
            _event_get_fret(ev),
        ),
    )

    kept = []
    last_same_string_fret: Dict[Tuple[int, int], Dict[str, Any]] = {}
    last_event_by_string: Dict[int, Dict[str, Any]] = {}

    min_repeat_frames = max(
        0,
        int(round((float(min_repeat_ms) / 1000.0) / float(frame_seconds))),
    )

    suppressed_count = 0

    for ev in events_sorted:
        s = _event_get_string(ev)
        fret = _event_get_fret(ev)
        cur_frame = _event_get_frame(ev)
        key = (s, fret)

        if _same_string_any_fret_too_close(
            ev,
            last_event_by_string=last_event_by_string,
            frame_seconds=frame_seconds,
            same_string_any_fret_min_ms=same_string_any_fret_min_ms,
        ):
            prev_ev = last_event_by_string.get(s)
            prev_fret = _event_get_fret(prev_ev) if prev_ev is not None else None
            tab_prob = float(ev.get("tab_prob", ev.get("frame_tab_prob", 0.0)) or 0.0)
            if prev_fret == fret or tab_prob < 0.85:
                ev["repeat_filter_action"] = "suppressed_same_string_too_close"
                ev["repeat_filter_reason"] = (
                    f"same string event within {same_string_any_fret_min_ms} ms"
                )
                suppressed_count += 1
                continue

        if key not in last_same_string_fret:
            ev["repeat_filter_action"] = "kept_first_same_string_fret"
            kept.append(ev)
            last_same_string_fret[key] = ev
            last_event_by_string[s] = ev
            continue

        prev_ev = last_same_string_fret[key]
        prev_frame = _event_get_frame(prev_ev)
        frame_gap = cur_frame - prev_frame
        dt_ms = frame_gap * float(frame_seconds) * 1000.0

        had_tab_change = _string_had_rest_or_fret_change_between(
            tab_classes=tab_classes,
            string_idx=s,
            fret=fret,
            prev_frame=prev_frame,
            cur_frame=cur_frame,
            rest_class=rest_class,
        )

        string_prob = _event_get_string_onset_prob(ev)
        global_prob = _event_get_global_onset_prob(ev)

        strong_onset_ok = (
            frame_gap >= min_repeat_frames
            and string_prob >= float(repeat_onset_threshold)
            and (
                not bool(require_global_for_repeats)
                or global_prob >= float(repeat_global_threshold)
            )
        )

        if policy == "refractory":
            keep = frame_gap >= min_repeat_frames
            reason = "kept_after_refractory" if keep else "suppressed_refractory"
        elif policy == "strong_onset":
            keep = strong_onset_ok
            reason = "kept_strong_repeat_onset" if keep else "suppressed_weak_repeat_onset"
        else:
            keep = bool(had_tab_change or strong_onset_ok)
            if had_tab_change:
                reason = "kept_tab_changed_between_repeats"
            elif strong_onset_ok:
                reason = "kept_strong_repeat_onset"
            else:
                reason = "suppressed_sustained_duplicate"

        ev["repeat_filter_action"] = reason
        ev["repeat_filter_dt_ms"] = float(dt_ms)
        ev["repeat_filter_frame_gap"] = int(frame_gap)
        ev["repeat_filter_had_tab_change"] = bool(had_tab_change)
        ev["repeat_filter_string_onset_prob"] = float(string_prob)
        ev["repeat_filter_global_onset_prob"] = float(global_prob)
        ev["repeat_filter_min_repeat_ms"] = float(min_repeat_ms)
        ev["repeat_filter_repeat_onset_threshold"] = float(repeat_onset_threshold)
        ev["repeat_filter_repeat_global_threshold"] = float(repeat_global_threshold)

        if keep:
            kept.append(ev)
            last_same_string_fret[key] = ev
            last_event_by_string[s] = ev
        else:
            suppressed_count += 1

    return kept, {
        "repeat_filter_policy": policy,
        "repeat_filter_input_events": int(len(events)),
        "repeat_filter_kept_events": int(len(kept)),
        "repeat_filter_suppressed_events": int(suppressed_count),
        "min_repeat_ms": float(min_repeat_ms),
        "repeat_onset_threshold": float(repeat_onset_threshold),
        "repeat_global_threshold": float(repeat_global_threshold),
        "require_global_for_repeats": bool(require_global_for_repeats),
        "same_string_any_fret_min_ms": float(same_string_any_fret_min_ms),
    }


def _decode_onset_events_from_string_peaks(
    frame_tab_probs: np.ndarray,
    frame_onset_probs: np.ndarray,
    global_onset_probs: np.ndarray,
    per_string_peak_frames: Sequence[Sequence[int]],
    global_peak_frames: Iterable[int],
    sr: int,
    hop_length: int,
    start_time: float = 0.0,
    start_frame: int = 0,
    label_window_ms: float = 50.0,
    label_delay_ms: float = 0.0,
    tab_threshold: float = 0.50,
    rest_class: int = TABESTIMATOR_REST_CLASS,
    use_global_onset_confirmation: bool = False,
    global_confirm_window_ms: float = 40.0,
    global_onset_threshold: float = 0.50,
    use_global_onset_fallback: bool = True,
    global_fallback_max_notes: int = 1,
) -> List[Dict[str, Any]]:
    """
    Arpeggio-safe event decoder.

    Primary timing comes from the per-string onset head. Each string onset can
    emit only that same string, so sustained notes on other strings are not
    copied into a later onset. The global onset head is optional confirmation
    and/or a conservative single-note fallback.
    """
    frame_tab_probs = np.asarray(frame_tab_probs, dtype=np.float32)
    frame_onset_probs = np.asarray(frame_onset_probs, dtype=np.float32)
    global_onset_probs = np.asarray(global_onset_probs, dtype=np.float32).reshape(-1)

    if frame_tab_probs.ndim != 3 or frame_tab_probs.shape[1] != 6:
        raise ValueError(f"Expected frame_tab_probs shape (T, 6, 21), got {frame_tab_probs.shape}")
    if frame_onset_probs.ndim != 2 or frame_onset_probs.shape[1] != 6:
        raise ValueError(f"Expected frame_onset_probs shape (T, 6), got {frame_onset_probs.shape}")

    T = int(min(frame_tab_probs.shape[0], frame_onset_probs.shape[0], global_onset_probs.shape[0]))
    seconds_per_frame = float(hop_length) / float(sr)
    label_window = max(0, _ms_to_frames(label_window_ms, sr, hop_length, minimum=0))
    label_delay = max(0, _ms_to_frames(label_delay_ms, sr, hop_length, minimum=0))
    global_confirm_radius = max(0, _ms_to_frames(global_confirm_window_ms, sr, hop_length, minimum=0))

    events: List[Dict[str, Any]] = []

    # A. Main path: per-string onsets create per-string events.
    for string_idx in range(min(6, len(per_string_peak_frames))):
        for peak in per_string_peak_frames[string_idx]:
            peak = int(peak)
            if peak < 0 or peak >= T:
                continue

            global_support, global_support_frame = _window_max_1d(
                global_onset_probs[:T],
                center=peak,
                radius=global_confirm_radius,
            )
            if bool(use_global_onset_confirmation) and global_support < float(global_onset_threshold):
                continue

            label_start = min(T, peak + label_delay)
            label_end = min(T, label_start + label_window + 1)
            tab_window = frame_tab_probs[label_start:label_end, string_idx, :int(rest_class)]
            if tab_window.size == 0:
                continue

            flat_idx = int(np.argmax(tab_window))
            local_frame, fret = np.unravel_index(flat_idx, tab_window.shape)
            tab_score = float(tab_window[local_frame, fret])
            if tab_score < float(tab_threshold):
                continue

            label_frame = int(label_start + local_frame)
            onset_time = float(start_time) + float(peak) * seconds_per_frame
            global_frame = int(start_frame) + int(peak)

            events.append(
                {
                    "time": onset_time,
                    "frame": int(peak),
                    "global_frame": int(global_frame),
                    "step": int(peak),
                    "global_step": int(global_frame),
                    "string_index_low_e_first": int(string_idx),
                    "string_number_low_e_first": int(string_idx + 1),
                    "fret": int(fret),
                    "source": "string_onset_decoder",
                    "decoder": "string_onset",
                    "string_onset_prob": float(frame_onset_probs[peak, string_idx]),
                    "global_onset_prob_nearby": float(global_support),
                    "global_onset_support_frame": int(global_support_frame),
                    "global_onset_support_global_frame": int(start_frame) + int(global_support_frame),
                    "tab_prob": float(tab_score),
                    "label_frame": int(label_frame),
                    "label_global_frame": int(start_frame) + int(label_frame),
                }
            )

    # B. Conservative global fallback: if a strong global onset has no nearby
    # string-onset event, recover at most N strongest single notes. This never
    # emits all sustained active strings, which is what creates fake chords.
    if bool(use_global_onset_fallback) and int(global_fallback_max_notes) > 0:
        existing_frames = [int(ev["frame"]) for ev in events]
        fallback_radius = global_confirm_radius

        for peak in global_peak_frames:
            peak = int(peak)
            if peak < 0 or peak >= T:
                continue
            if any(abs(peak - f) <= fallback_radius for f in existing_frames):
                continue

            label_start = min(T, peak + label_delay)
            label_end = min(T, label_start + label_window + 1)
            tab_window_all = frame_tab_probs[label_start:label_end, :, :int(rest_class)]
            if tab_window_all.size == 0:
                continue

            # Rank candidates by frame-tab probability and emit at most N notes.
            flat_order = np.argsort(tab_window_all.reshape(-1))[::-1]
            emitted_here = 0
            used_strings = set()

            for flat_idx in flat_order:
                local_frame, string_idx, fret = np.unravel_index(int(flat_idx), tab_window_all.shape)
                string_idx = int(string_idx)
                if string_idx in used_strings:
                    continue
                tab_score = float(tab_window_all[local_frame, string_idx, fret])
                if tab_score < max(0.35, float(tab_threshold) * 0.75):
                    break

                label_frame = int(label_start + local_frame)
                onset_time = float(start_time) + float(peak) * seconds_per_frame
                global_frame = int(start_frame) + int(peak)

                events.append(
                    {
                        "time": onset_time,
                        "frame": int(peak),
                        "global_frame": int(global_frame),
                        "step": int(peak),
                        "global_step": int(global_frame),
                        "string_index_low_e_first": int(string_idx),
                        "string_number_low_e_first": int(string_idx + 1),
                        "fret": int(fret),
                        "source": "string_onset_decoder_global_fallback",
                        "decoder": "string_onset",
                        "string_onset_prob": float(frame_onset_probs[peak, string_idx]) if peak < frame_onset_probs.shape[0] else None,
                        "global_onset_prob_nearby": float(global_onset_probs[peak]) if peak < len(global_onset_probs) else None,
                        "tab_prob": float(tab_score),
                        "label_frame": int(label_frame),
                        "label_global_frame": int(start_frame) + int(label_frame),
                    }
                )
                used_strings.add(string_idx)
                emitted_here += 1
                if emitted_here >= int(global_fallback_max_notes):
                    break

    return sorted(
        events,
        key=lambda ev: (
            float(ev.get("time", 0.0)),
            int(ev.get("string_index_low_e_first", 0)),
            int(ev.get("fret", 0)),
        ),
    )

def run_tabestimator_hand_onset_npz_inference(
    model,
    npz_path: str | os.PathLike,
    config: Dict[str, Any],
    device: str,
    hand_pos_dim: int,
    feature_key: Optional[str] = None,
    normalize_hand_rows: bool = False,
    start_time: float = 0.0,
    start_frame: int = 0,
    frame_step_width: int = 2,
    onset_threshold: float = 0.5,
    global_onset_threshold: float = 0.5,
    peak_pre_max_ms: float = 90.0,
    peak_post_max_ms: float = 90.0,
    peak_combine_ms: float = 30.0,
    event_label_delay_ms: float = 0.0,
    event_label_window_ms: float = 50.0,
    event_string_window_ms: float = 50.0,
    event_tab_threshold: float = 0.50,
    event_string_threshold: float = 0.30,
    no_event_string_filter: bool = False,
    event_decode_mode: str = "string_onset",
    event_chord_group_ms: float = 25.0,
    use_global_onset_confirmation: bool = False,
    global_confirm_window_ms: float = 40.0,
    use_global_onset_fallback: bool = True,
    global_fallback_max_notes: int = 1,
    repeat_same_fret_policy: str = "tab_change_or_strong_onset",
    min_repeat_ms: float = 90.0,
    repeat_onset_threshold: float = 0.90,
    repeat_global_threshold: float = 0.65,
    require_global_for_repeats: bool = True,
    same_string_any_fret_min_ms: float = 35.0,
) -> Dict[str, Any]:
    """
    Run the BPM-free hand+onset TabEstimator on one runtime NPZ split.

    The returned `events` are symbolic note events. By default, event timing
    comes from per-string onset peaks, and each onset labels only its own string.
    This avoids treating sustained notes from previous attacks as part of a
    later arpeggio onset. The old global-onset-window decoder is still
    available with event_decode_mode="global_onset" for comparison.
    """
    npz_path = _resolve_existing_npz_path(npz_path)

    with np.load(npz_path, allow_pickle=True) as data:
        available_keys = list(data.files)
        key = _feature_key_from_config(config, feature_key=feature_key)
        if key not in data.files:
            raise KeyError(f"Feature key {key!r} not found in NPZ. Available keys: {available_keys}")
        if "frame_hand_pos" not in data.files:
            raise KeyError(f"Hand key 'frame_hand_pos' not found in NPZ. Available keys: {available_keys}")

        features = np.asarray(data[key], dtype=np.float32)
        frame_hand_pos = np.asarray(data["frame_hand_pos"], dtype=np.float32)
        frame_hand_pos_index = np.asarray(data["frame_hand_pos_index"], dtype=np.int64) if "frame_hand_pos_index" in data.files else None

    if features.ndim != 2:
        raise ValueError(f"Expected features shape (T, F), got {features.shape}")

    frame_count = int(features.shape[0])
    frame_hand_pos = _fit_2d_time_and_dim(
        frame_hand_pos,
        target_len=frame_count,
        target_dim=int(hand_pos_dim),
        normalize_rows=normalize_hand_rows,
    )

    src = torch.from_numpy(features).float().unsqueeze(0).to(device)
    src_len = torch.tensor([frame_count], dtype=torch.long, device=device)
    frame_hand = torch.from_numpy(frame_hand_pos).float().unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(src, src_len, frame_hand_pos=frame_hand)

    if not isinstance(outputs, (tuple, list)) or len(outputs) < 3:
        raise RuntimeError("Onset model forward must return frame_tab_pred, frame_onset_logits, global_onset_logits, olens")

    frame_tab_pred = outputs[0]
    frame_onset_logits = outputs[1]
    global_onset_logits = outputs[2]

    frame_tab_probs = torch.squeeze(frame_tab_pred, 0).detach().cpu().numpy().astype(np.float32)
    frame_onset_probs = _sigmoid_np(torch.squeeze(frame_onset_logits, 0).detach().cpu().numpy())
    global_onset_probs = _sigmoid_np(torch.squeeze(global_onset_logits, 0).detach().cpu().numpy()).reshape(-1)

    frame_classes = np.argmax(frame_tab_probs, axis=2).astype(np.int64)

    sr = int(_get_config_required(config, "down_sampling_rate"))
    hop_length = int(_get_config_required(config, "hop_length"))
    frame_ms = 1000.0 * float(hop_length) / float(sr)
    seconds_per_frame = float(hop_length) / float(sr)

    pre_max_frames = _ms_to_frames(peak_pre_max_ms, sr, hop_length, minimum=0)
    post_max_frames = _ms_to_frames(peak_post_max_ms, sr, hop_length, minimum=0)
    combine_frames = _ms_to_frames(peak_combine_ms, sr, hop_length, minimum=0)

    global_peak_frames = _peak_picking_1d(
        global_onset_probs,
        threshold=float(global_onset_threshold),
        pre_max=pre_max_frames,
        post_max=post_max_frames,
        combine=combine_frames,
    )

    per_string_peak_frames = []
    if frame_onset_probs.ndim == 2 and frame_onset_probs.shape[1] == 6:
        for s in range(6):
            peaks = _peak_picking_1d(
                frame_onset_probs[:, s],
                threshold=float(onset_threshold),
                pre_max=pre_max_frames,
                post_max=post_max_frames,
                combine=combine_frames,
            )
            per_string_peak_frames.append(peaks.astype(int).tolist())

    event_decode_mode = str(event_decode_mode or "string_onset").strip().lower()
    if event_decode_mode not in {"string_onset", "global_onset", "hybrid"}:
        raise ValueError(
            "event_decode_mode must be one of: 'string_onset', 'global_onset', 'hybrid'. "
            f"Got {event_decode_mode!r}."
        )

    repeat_filter_diagnostics = None
    if event_decode_mode == "global_onset":
        raw_onset_events = _decode_onset_events_from_global_peaks(
            frame_tab_probs=frame_tab_probs,
            frame_onset_probs=frame_onset_probs,
            global_onset_probs=global_onset_probs,
            global_peak_frames=global_peak_frames,
            sr=sr,
            hop_length=hop_length,
            start_time=float(start_time),
            start_frame=int(start_frame),
            label_window_ms=float(event_label_window_ms),
            label_delay_ms=float(event_label_delay_ms),
            string_support_window_ms=float(event_string_window_ms),
            tab_threshold=float(event_tab_threshold),
            string_threshold=float(event_string_threshold),
            rest_class=TABESTIMATOR_REST_CLASS,
            use_string_onset_filter=not bool(no_event_string_filter),
        )
    else:
        raw_onset_events = _decode_onset_events_from_string_peaks(
            frame_tab_probs=frame_tab_probs,
            frame_onset_probs=frame_onset_probs,
            global_onset_probs=global_onset_probs,
            per_string_peak_frames=per_string_peak_frames,
            global_peak_frames=global_peak_frames,
            sr=sr,
            hop_length=hop_length,
            start_time=float(start_time),
            start_frame=int(start_frame),
            label_window_ms=float(event_label_window_ms),
            label_delay_ms=float(event_label_delay_ms),
            tab_threshold=float(event_tab_threshold),
            rest_class=TABESTIMATOR_REST_CLASS,
            use_global_onset_confirmation=bool(use_global_onset_confirmation),
            global_confirm_window_ms=float(global_confirm_window_ms),
            global_onset_threshold=float(global_onset_threshold),
            use_global_onset_fallback=bool(use_global_onset_fallback or event_decode_mode == "hybrid"),
            global_fallback_max_notes=int(global_fallback_max_notes),
        )

        raw_onset_events, repeat_filter_diagnostics = filter_repeated_string_fret_events(
            raw_onset_events,
            frame_tab_probs=frame_tab_probs,
            frame_seconds=seconds_per_frame,
            rest_class=TABESTIMATOR_REST_CLASS,
            repeat_same_fret_policy=str(repeat_same_fret_policy),
            min_repeat_ms=float(min_repeat_ms),
            repeat_onset_threshold=float(repeat_onset_threshold),
            repeat_global_threshold=float(repeat_global_threshold),
            require_global_for_repeats=bool(require_global_for_repeats),
            same_string_any_fret_min_ms=float(same_string_any_fret_min_ms),
        )

    chord_group_window_seconds = max(0.0, float(event_chord_group_ms) / 1000.0)
    onset_events = copy_events_with_chord_grouping(
        raw_onset_events,
        window_steps=0,
        window_seconds=chord_group_window_seconds,
        step_key="global_step",
    )

    frame_events = tabestimator_frame_classes_to_events(
        frame_classes,
        chunk_start_time=float(start_time),
        sr=sr,
        hop_length=hop_length,
        collapse_repeats=True,
        start_frame=int(start_frame),
    )

    frame_ascii_tab = tabestimator_classes_to_ascii(
        frame_classes,
        step_width=int(frame_step_width),
        reverse_strings=True,
        collapse_repeats=True,
    )

    return {
        "npz_path": npz_path,
        "available_keys": available_keys,
        "feature_key": key,
        "start_time": float(start_time),
        "start_frame": int(start_frame),
        "frame_count": int(frame_count),
        "frame_pred_len": int(frame_classes.shape[0]),
        "frame_dur": float(hop_length) / float(sr),
        "frame_ms": float(frame_ms),
        "feature_shape": [int(x) for x in features.shape],
        "frame_hand_pos_shape": [int(x) for x in frame_hand_pos.shape],
        "frame_hand_pos_index": None if frame_hand_pos_index is None else frame_hand_pos_index[:frame_count].astype(int).tolist(),
        "frame_pred_classes_low_e_first": frame_classes.astype(int).tolist(),
        "frame_events": frame_events,
        "events": onset_events,
        "note_events": onset_events,
        "onset_events": onset_events,
        "raw_onset_events": raw_onset_events,
        "ungrouped_onset_events": raw_onset_events,
        "vextab": events_to_vextab_text(onset_events),
        "note_vextab": events_to_vextab_text(onset_events),
        "frame_ascii_tab": frame_ascii_tab,
        "global_onset_peak_frames": global_peak_frames.astype(int).tolist(),
        "per_string_onset_peak_frames": per_string_peak_frames,
        "onset_threshold": float(onset_threshold),
        "global_onset_threshold": float(global_onset_threshold),
        "peak_picking": {
            "pre_max_ms": float(peak_pre_max_ms),
            "post_max_ms": float(peak_post_max_ms),
            "combine_ms": float(peak_combine_ms),
            "pre_max_frames": int(pre_max_frames),
            "post_max_frames": int(post_max_frames),
            "combine_frames": int(combine_frames),
        },
        "event_decoder": {
            "mode": str(event_decode_mode),
            "primary_timing": "per_string_onset" if event_decode_mode in {"string_onset", "hybrid"} else "global_onset",
            "label_delay_ms": float(event_label_delay_ms),
            "label_window_ms": float(event_label_window_ms),
            "string_window_ms": float(event_string_window_ms),
            "tab_threshold": float(event_tab_threshold),
            "string_threshold": float(event_string_threshold),
            "no_event_string_filter": bool(no_event_string_filter),
            "use_string_onset_filter": not bool(no_event_string_filter),
            "chord_group_ms": float(event_chord_group_ms),
            "use_global_onset_confirmation": bool(use_global_onset_confirmation),
            "global_confirm_window_ms": float(global_confirm_window_ms),
            "use_global_onset_fallback": bool(use_global_onset_fallback or event_decode_mode == "hybrid"),
            "global_fallback_max_notes": int(global_fallback_max_notes),
            "repeat_same_fret_policy": str(repeat_same_fret_policy),
            "min_repeat_ms": float(min_repeat_ms),
            "repeat_onset_threshold": float(repeat_onset_threshold),
            "repeat_global_threshold": float(repeat_global_threshold),
            "require_global_for_repeats": bool(require_global_for_repeats),
            "same_string_any_fret_min_ms": float(same_string_any_fret_min_ms),
        },
        "count_onset_events": int(len(onset_events)),
        "count_raw_onset_events": int(len(raw_onset_events)),
        "repeat_filter_policy": str(repeat_same_fret_policy),
        "repeat_filter_input_events": int(repeat_filter_diagnostics["repeat_filter_input_events"]) if repeat_filter_diagnostics is not None else int(len(raw_onset_events)),
        "repeat_filter_kept_events": int(repeat_filter_diagnostics["repeat_filter_kept_events"]) if repeat_filter_diagnostics is not None else int(len(raw_onset_events)),
        "repeat_filter_suppressed_events": int(repeat_filter_diagnostics["repeat_filter_suppressed_events"]) if repeat_filter_diagnostics is not None else 0,
        "repeat_filter_diagnostics": repeat_filter_diagnostics,
    }



# =============================================================================
# Hidden-onset hand TabEstimator loading and runtime decoding
# =============================================================================


def _load_hidonset_network_class(tabestimator_root: str | os.PathLike):
    """
    Load TabEstimator from Tab-estimator-hand-hidonset-light/src/network.py without
    clobbering the already imported legacy hand-conditioned network module.
    """
    root = Path(tabestimator_root).expanduser().resolve()
    network_path = root / "src" / "network.py"

    if not network_path.is_file():
        raise FileNotFoundError(f"Hidden-onset network.py not found: {network_path}")

    module_name = "tabestimator_hand_hidonset_network"
    spec = importlib.util.spec_from_file_location(module_name, str(network_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    return module.TabEstimator


def _candidate_hidonset_model_roots(model_root=None):
    here = Path(__file__).resolve().parent
    cwd = Path(os.getcwd()).resolve()

    candidates = []
    if model_root:
        candidates.append(Path(model_root))

    candidates.extend(
        [
            here / "Tab-estimator-hand-hidonset-light" / "model",
            here.parent / "Tab-estimator-hand-hidonset-light" / "model",
            cwd / "backend_cv" / "Tab-estimator-hand-hidonset-light" / "model",
            cwd / "backend" / "Tab-estimator-hand-hidonset-light" / "model",
            cwd / "Tab-estimator-hand-hidonset-light" / "model",
            DEFAULT_TABEST_HIDONSET_MODEL_ROOT,
        ]
    )

    out = []
    seen = set()
    for path in candidates:
        norm = str(Path(path).expanduser().resolve())
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def resolve_tabestimator_hand_hidonset_paths(
    model_run=None,
    epoch=None,
    test_num=0,
    model_root=None,
    checkpoint_path=None,
    config_path=None,
):
    """
    Resolve a hidden-onset hand TabEstimator checkpoint under
    Tab-estimator-hand-hidonset-light.
    """
    if checkpoint_path and config_path:
        checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        config_path = str(Path(config_path).expanduser().resolve())
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Hidden-onset checkpoint not found: {checkpoint_path}")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Hidden-onset config not found: {config_path}")
        if model_root:
            resolved_model_root = str(Path(model_root).expanduser().resolve())
        else:
            resolved_model_root = str(Path(config_path).parent.parent.parent.resolve())
        return {
            "checkpoint_path": checkpoint_path,
            "config_path": config_path,
            "metadata_path": str(Path(config_path).parent / "run_metadata.yaml"),
            "model_root": resolved_model_root,
            "tabestimator_root": str(Path(resolved_model_root).parent.resolve()),
        }

    if not model_run:
        raise ValueError("model_run is required unless checkpoint_path/config_path are provided.")
    if epoch is None:
        raise ValueError("epoch is required unless checkpoint_path is provided.")

    fold_id = f"{int(test_num):02d}"
    errors = []

    for root in _candidate_hidonset_model_roots(model_root):
        run_dir = Path(root) / str(model_run)
        cfg = run_dir / "config.yaml"
        ckpt = run_dir / f"testNo{fold_id}" / f"epoch{int(epoch)}.model"
        metadata = run_dir / "run_metadata.yaml"

        if cfg.is_file() and ckpt.is_file():
            return {
                "checkpoint_path": str(ckpt.resolve()),
                "config_path": str(cfg.resolve()),
                "metadata_path": str(metadata.resolve()) if metadata.is_file() else None,
                "model_root": str(Path(root).resolve()),
                "tabestimator_root": str(Path(root).parent.resolve()),
            }

        errors.append(f"root={root} | config={cfg.is_file()} | checkpoint={ckpt.is_file()}")

    raise FileNotFoundError(
        "Could not resolve hidden-onset hand TabEstimator model.\n"
        f"model_run={model_run}\n"
        f"epoch={epoch}\n"
        f"test_num={test_num}\n"
        "Tried:\n" + "\n".join(errors)
    )


def _clean_state_dict_module_prefix(state_dict):
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if isinstance(state_dict, dict):
        return {str(k)[7:] if str(k).startswith("module.") else k: v for k, v in state_dict.items()}
    return state_dict


def _load_state_dict_allow_hidonset_aliases(model, state_dict):
    """
    Load strictly when possible. If a local network.py defines alias modules like
    frame_onset_head/global_onset_head, tolerate only those alias-key differences.
    """
    try:
        model.load_state_dict(state_dict, strict=True)
        return {"strict": True, "missing_keys": [], "unexpected_keys": []}
    except RuntimeError:
        result = model.load_state_dict(state_dict, strict=False)
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys)

        allowed_prefixes = (
            "frame_onset_head.",
            "global_onset_head.",
            "frame_onset_output_layer.",
            "global_onset_output_layer.",
        )

        def allowed(key):
            return str(key).startswith(allowed_prefixes)

        bad_missing = [k for k in missing if not allowed(k)]
        bad_unexpected = [k for k in unexpected if not allowed(k)]

        if bad_missing or bad_unexpected:
            raise RuntimeError(
                "Hidden-onset checkpoint does not match network.py.\n"
                f"Bad missing keys: {bad_missing[:30]}\n"
                f"Bad unexpected keys: {bad_unexpected[:30]}\n"
                f"All missing keys: {missing[:50]}\n"
                f"All unexpected keys: {unexpected[:50]}"
            )

        return {
            "strict": False,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        }


def load_tabestimator_hand_hidonset_model(
    model_run=None,
    epoch=None,
    test_num=6,
    model_root=None,
    checkpoint_path=None,
    config_path=None,
    device=None,
):
    """
    Load and cache the hidden-onset hand TabEstimator.

    This is the new architecture:
      - frame_tab_output_layer
      - note_tab_output_layer
      - frame_onset_output_layer
      - global_onset_output_layer
      - optional note_onset_conditioning

    The app should decode final transcription from note_tab_pred only.
    """
    paths = resolve_tabestimator_hand_hidonset_paths(
        model_run=model_run,
        epoch=epoch,
        test_num=test_num,
        model_root=model_root,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_key = f"{paths['checkpoint_path']}|{paths['config_path']}|{device}"
    if cache_key in TABESTIMATOR_HIDONSET_MODEL_CACHE:
        return TABESTIMATOR_HIDONSET_MODEL_CACHE[cache_key]

    with open(paths["config_path"], "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    metadata = _read_yaml_if_exists(paths.get("metadata_path"))

    input_feature_type = str(_get_config_required(config, "input_feature_type"))
    if input_feature_type == "cqt":
        n_bins = int(_get_config_required(config, "cqt_n_bins"))
    elif input_feature_type == "melspec":
        n_bins = 128
    else:
        raise ValueError(f"Unknown input_feature_type: {input_feature_type!r}")

    HidonsetTabEstimator = _load_hidonset_network_class(paths["tabestimator_root"])

    use_hand_position = _bool_from_config(
        _metadata_or_config(metadata, config, "use_hand_position", True),
        default=True,
    )
    hand_pos_dim = int(_metadata_or_config(metadata, config, "hand_pos_dim", 20))
    hand_position_fusion = str(_metadata_or_config(metadata, config, "hand_position_fusion", "hidden+prior"))
    hand_hidden_gate_init = float(_metadata_or_config(metadata, config, "hand_hidden_gate_init", 0.5))
    hand_prior_strength = float(_metadata_or_config(metadata, config, "hand_prior_strength", 0.35))
    hand_span = int(_metadata_or_config(metadata, config, "hand_span", 4))
    note_target_length = int(
        _metadata_or_config(
            metadata,
            config,
            "note_target_length",
            int(_metadata_or_config(metadata, config, "note_resolution", 16)) * 4,
        )
    )

    model = HidonsetTabEstimator(
        str(_get_config_required(config, "mode")),
        str(_get_config_required(config, "encoder_type")),
        _bool_from_config(_get_config_required(config, "use_custom_decimation_func")),
        _bool_from_config(_get_config_required(config, "use_conv_stack")),
        n_bins,
        int(_get_config_required(config, "hop_length")),
        int(_get_config_required(config, "down_sampling_rate")),
        encoder_heads=int(_get_config_required(config, "encoder_heads")),
        encoder_layers=int(_get_config_required(config, "encoder_layers")),
        use_hand_position=use_hand_position,
        hand_pos_dim=hand_pos_dim,
        hand_position_fusion=hand_position_fusion,
        hand_hidden_gate_init=hand_hidden_gate_init,
        hand_prior_strength=hand_prior_strength,
        hand_span=hand_span,
        note_target_length=note_target_length,
        onset_hidden_dim=int(_metadata_or_config(metadata, config, "onset_hidden_dim", 64)),
        onset_dropout=float(_metadata_or_config(metadata, config, "onset_dropout", 0.25)),
        onset_kernel_size=int(_metadata_or_config(metadata, config, "onset_kernel_size", 3)),
        onset_tcn_levels=int(_metadata_or_config(metadata, config, "onset_tcn_levels", 4)),
        onset_use_raw_features=_bool_from_config(_metadata_or_config(metadata, config, "onset_use_raw_features", True), default=True),
        onset_raw_proj_dim=int(_metadata_or_config(metadata, config, "onset_raw_proj_dim", 64)),
        onset_raw_dropout=float(_metadata_or_config(metadata, config, "onset_raw_dropout", 0.10)),
        onset_input_mode=str(_metadata_or_config(metadata, config, "onset_input_mode", "full")),
        use_hidden_onset_to_note=_bool_from_config(_metadata_or_config(metadata, config, "uses_hidden_onset_to_note", _metadata_or_config(metadata, config, "use_hidden_onset_to_note", True)), default=True),
        onset_note_fusion=str(_metadata_or_config(metadata, config, "onset_note_fusion", "gated_add")),
        detach_onset_features_for_note=_bool_from_config(_metadata_or_config(metadata, config, "detach_onset_features_for_note", False), default=False),
        note_hidden_hand_fusion=_bool_from_config(_metadata_or_config(metadata, config, "note_hidden_hand_fusion", False), default=False),
        note_rest_preserving_prior=_bool_from_config(_metadata_or_config(metadata, config, "note_rest_preserving_prior", True), default=True),
    )

    state_dict = _safe_torch_load(paths["checkpoint_path"], map_location="cpu")
    state_dict = _clean_state_dict_module_prefix(state_dict)
    load_report = _load_state_dict_allow_hidonset_aliases(model, state_dict)

    model.to(device)
    model.eval()

    bundle = {
        "model": model,
        "config": config,
        "metadata": metadata,
        "device": device,
        "checkpoint_path": paths["checkpoint_path"],
        "config_path": paths["config_path"],
        "metadata_path": paths.get("metadata_path"),
        "model_root": paths["model_root"],
        "tabestimator_root": paths["tabestimator_root"],
        "input_feature_type": input_feature_type,
        "use_hand_position": bool(use_hand_position),
        "hand_pos_dim": int(hand_pos_dim),
        "hand_span": int(hand_span),
        "hand_position_fusion": hand_position_fusion,
        "note_target_length": int(note_target_length),
        "onset_input_mode": str(_metadata_or_config(metadata, config, "onset_input_mode", "full")),
        "uses_hidden_onset_to_note": bool(_bool_from_config(_metadata_or_config(metadata, config, "uses_hidden_onset_to_note", _metadata_or_config(metadata, config, "use_hidden_onset_to_note", True)), default=True)),
        "uses_thresholded_onset_for_prediction": False,
        "load_report": load_report,
    }

    TABESTIMATOR_HIDONSET_MODEL_CACHE[cache_key] = bundle
    return bundle


def run_tabestimator_hand_hidonset_chunk_outputs(
    model,
    features: np.ndarray,
    bpm: float,
    frame_hand_pos: np.ndarray,
    note_hand_pos: np.ndarray,
    note_len: int,
    device: str,
    min_note_prob: Optional[float] = None,
    min_margin_vs_rest: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run one split/chunk through the hidden-onset model.

    Final transcription is from note_tab_pred. Onset logits are returned only as
    diagnostics; they are not thresholded or used to decode notes.
    """
    features = np.asarray(features, dtype=np.float32)
    frame_hand_pos = np.asarray(frame_hand_pos, dtype=np.float32)
    note_hand_pos = np.asarray(note_hand_pos, dtype=np.float32)

    if features.ndim != 2:
        raise ValueError(f"features must have shape (T, F), got {features.shape}")
    if frame_hand_pos.ndim != 2:
        raise ValueError(f"frame_hand_pos must have shape (T, H), got {frame_hand_pos.shape}")
    if note_hand_pos.ndim != 2:
        raise ValueError(f"note_hand_pos must have shape (N, H), got {note_hand_pos.shape}")
    if frame_hand_pos.shape[0] != features.shape[0]:
        raise ValueError(
            f"frame_hand_pos length must match features length: "
            f"{frame_hand_pos.shape[0]} != {features.shape[0]}"
        )
    if note_hand_pos.shape[0] != int(note_len):
        raise ValueError(
            f"note_hand_pos length must match note_len: "
            f"{note_hand_pos.shape[0]} != {int(note_len)}"
        )

    input_features = torch.from_numpy(features).unsqueeze(0).to(device)
    frame_len = torch.tensor([features.shape[0]], dtype=torch.long, device=device)
    note_len_tensor = torch.tensor([int(note_len)], dtype=torch.long, device=device)
    bpm_tensor = torch.tensor([float(bpm)], dtype=torch.float32, device=device)
    frame_hand_tensor = torch.from_numpy(frame_hand_pos).unsqueeze(0).to(device)
    note_hand_tensor = torch.from_numpy(note_hand_pos).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(
            input_features.float(),
            frame_len,
            note_len_tensor,
            bpm_tensor,
            frame_hand_pos=frame_hand_tensor.float(),
            note_hand_pos=note_hand_tensor.float(),
        )

    if not isinstance(outputs, (tuple, list)):
        raise RuntimeError("Hidden-onset model forward must return a tuple/list.")

    if len(outputs) >= 5:
        frame_pred, note_pred, frame_onset_logits, global_onset_logits, olens = outputs[:5]
    elif len(outputs) == 3:
        # Defensive fallback for accidental old hand model loading.
        frame_pred, note_pred, olens = outputs
        frame_onset_logits = None
        global_onset_logits = None
    else:
        raise RuntimeError(
            "Hidden-onset model forward must return either "
            "(frame_tab_pred, note_tab_pred, frame_onset_logits, global_onset_logits, olens) "
            "or legacy (frame_tab_pred, note_tab_pred, olens)."
        )

    frame_pred_np = torch.squeeze(frame_pred, 0).detach().cpu().numpy()
    note_pred_np = torch.squeeze(note_pred, 0).detach().cpu().numpy()

    if frame_pred_np.ndim != 3 or frame_pred_np.shape[1] != 6:
        raise RuntimeError(f"Unexpected frame_pred shape from hidonset model: {frame_pred_np.shape}")
    if note_pred_np.ndim != 3 or note_pred_np.shape[1] != 6:
        raise RuntimeError(f"Unexpected note_pred shape from hidonset model: {note_pred_np.shape}")

    frame_out_len = int(olens[0].item()) if olens is not None else int(frame_pred_np.shape[0])
    frame_pred_np = frame_pred_np[:frame_out_len]

    frame_classes = tabestimator_logits_to_classes(
        frame_pred_np,
        rest_class=TABESTIMATOR_REST_CLASS,
        min_note_prob=min_note_prob,
        min_margin_vs_rest=min_margin_vs_rest,
    )

    # The final app-level prediction comes from note_tab_pred.
    note_classes = tabestimator_logits_to_classes(
        note_pred_np,
        rest_class=TABESTIMATOR_REST_CLASS,
        min_note_prob=min_note_prob,
        min_margin_vs_rest=min_margin_vs_rest,
    )

    diagnostics: Dict[str, Any] = {}
    if frame_onset_logits is not None:
        frame_onset_probs = torch.sigmoid(torch.squeeze(frame_onset_logits, 0)).detach().cpu().numpy()
        diagnostics["frame_onset_prob_shape"] = [int(x) for x in frame_onset_probs.shape]
        diagnostics["frame_onset_prob_max"] = float(np.max(frame_onset_probs)) if frame_onset_probs.size else 0.0
        diagnostics["frame_onset_prob_mean"] = float(np.mean(frame_onset_probs)) if frame_onset_probs.size else 0.0

    if global_onset_logits is not None:
        global_onset_probs = torch.sigmoid(torch.squeeze(global_onset_logits, 0)).detach().cpu().numpy().reshape(-1)
        diagnostics["global_onset_prob_shape"] = [int(x) for x in global_onset_probs.shape]
        diagnostics["global_onset_prob_max"] = float(np.max(global_onset_probs)) if global_onset_probs.size else 0.0
        diagnostics["global_onset_prob_mean"] = float(np.mean(global_onset_probs)) if global_onset_probs.size else 0.0

    return {
        "frame_classes": frame_classes.astype(np.int64),
        "note_classes": note_classes.astype(np.int64),
        "olens": [int(frame_out_len)],
        "onset_diagnostics": diagnostics,
        "main_decoder": "note_tab_pred",
        "uses_thresholded_onset_for_prediction": False,
    }


def run_tabestimator_hand_hidonset_npz_inference(
    model,
    npz_path: str | os.PathLike,
    config: Dict[str, Any],
    device: str,
    hand_pos_dim: int,
    bpm: Optional[float] = None,
    feature_key: Optional[str] = None,
    normalize_hand_rows: bool = False,
    start_time: float = 0.0,
    start_step: int = 0,
    start_frame: int = 0,
    step_width: int = 4,
    frame_step_width: int = 2,
    min_note_prob: Optional[float] = None,
    min_margin_vs_rest: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run the hidden-onset hand TabEstimator on one runtime NPZ split.

    The returned `events` are decoded from note_tab_pred only. The onset heads are
    hidden/auxiliary streams and are not thresholded for final prediction.
    """
    npz_path = _resolve_existing_npz_path(npz_path)

    with np.load(npz_path, allow_pickle=True) as data:
        available_keys = list(data.files)
        key = _feature_key_from_config(config, feature_key=feature_key)

        if key not in data.files:
            raise KeyError(f"Feature key {key!r} not found in NPZ. Available keys: {available_keys}")
        if "frame_hand_pos" not in data.files:
            raise KeyError(f"Hand key 'frame_hand_pos' not found in NPZ. Available keys: {available_keys}")
        if "hand_pos" not in data.files:
            raise KeyError(f"Hand key 'hand_pos' not found in NPZ. Available keys: {available_keys}")

        features = np.asarray(data[key], dtype=np.float32)
        frame_hand_pos = np.asarray(data["frame_hand_pos"], dtype=np.float32)
        note_hand_pos = np.asarray(data["hand_pos"], dtype=np.float32)

        if bpm is None:
            used_bpm = float(np.asarray(data["tempo"]).reshape(-1)[0]) if "tempo" in data.files else 120.0
        else:
            used_bpm = float(bpm)

        if "len_in_notes" in data.files:
            note_len = int(np.asarray(data["len_in_notes"]).reshape(-1)[0])
        else:
            note_len = int(note_hand_pos.shape[0])

        hand_pos_index = np.asarray(data["hand_pos_index"], dtype=np.int64) if "hand_pos_index" in data.files else None
        frame_hand_pos_index = np.asarray(data["frame_hand_pos_index"], dtype=np.int64) if "frame_hand_pos_index" in data.files else None

    if features.ndim != 2:
        raise ValueError(f"Expected features shape (T, F), got {features.shape}")

    frame_count = int(features.shape[0])
    note_len = int(note_len)

    frame_hand_pos = _fit_2d_time_and_dim(
        frame_hand_pos,
        target_len=frame_count,
        target_dim=int(hand_pos_dim),
        normalize_rows=normalize_hand_rows,
    )

    note_hand_pos = _fit_2d_time_and_dim(
        note_hand_pos,
        target_len=note_len,
        target_dim=int(hand_pos_dim),
        normalize_rows=normalize_hand_rows,
    )

    outputs = run_tabestimator_hand_hidonset_chunk_outputs(
        model=model,
        features=features,
        bpm=used_bpm,
        frame_hand_pos=frame_hand_pos,
        note_hand_pos=note_hand_pos,
        note_len=note_len,
        device=device,
        min_note_prob=min_note_prob,
        min_margin_vs_rest=min_margin_vs_rest,
    )

    pred_classes = outputs["note_classes"]
    frame_pred_classes = outputs["frame_classes"]

    note_resolution = int(_get_config_required(config, "note_resolution"))
    sr = int(_get_config_required(config, "down_sampling_rate"))
    hop_length = int(_get_config_required(config, "hop_length"))

    events = tabestimator_classes_to_events(
        pred_classes,
        chunk_start_time=float(start_time),
        bpm=used_bpm,
        note_resolution=note_resolution,
        collapse_repeats=True,
    )

    for ev in events:
        ev["global_step"] = int(start_step) + int(ev["step"])
        ev["decoder"] = "note_tab_pred"
        ev["source"] = "hidonset_note_tab_pred"

    frame_events = tabestimator_frame_classes_to_events(
        frame_pred_classes,
        chunk_start_time=float(start_time),
        sr=sr,
        hop_length=hop_length,
        collapse_repeats=True,
        start_frame=int(start_frame),
    )

    ascii_tab = tabestimator_classes_to_ascii(
        pred_classes,
        step_width=int(step_width),
        reverse_strings=True,
        collapse_repeats=True,
    )

    frame_ascii_tab = tabestimator_classes_to_ascii(
        frame_pred_classes,
        step_width=int(frame_step_width),
        reverse_strings=True,
        collapse_repeats=True,
    )

    return {
        "npz_path": npz_path,
        "available_keys": available_keys,
        "feature_key": key,
        "bpm": float(used_bpm),
        "note_resolution": int(note_resolution),
        "frame_count": int(frame_count),
        "frame_pred_len": int(frame_pred_classes.shape[0]),
        "note_len": int(note_len),
        "start_time": float(start_time),
        "start_step": int(start_step),
        "start_frame": int(start_frame),

        "feature_shape": [int(x) for x in features.shape],
        "frame_hand_pos_shape": [int(x) for x in frame_hand_pos.shape],
        "note_hand_pos_shape": [int(x) for x in note_hand_pos.shape],

        "hand_pos_index": None if hand_pos_index is None else hand_pos_index[:note_len].astype(int).tolist(),
        "frame_hand_pos_index": None if frame_hand_pos_index is None else frame_hand_pos_index[:frame_count].astype(int).tolist(),

        "pred_classes_low_e_first": pred_classes.astype(int).tolist(),
        "frame_pred_classes_low_e_first": frame_pred_classes.astype(int).tolist(),

        "events": events,
        "note_events": events,
        "frame_events": frame_events,

        "vextab": events_to_vextab_text(events),
        "note_vextab": events_to_vextab_text(events),
        "ascii_tab": ascii_tab,
        "note_ascii_tab": ascii_tab,
        "frame_ascii_tab": frame_ascii_tab,

        "main_decoder": "note_tab_pred",
        "uses_thresholded_onset_for_prediction": False,
        "onset_logits_are_diagnostics_only": True,
        "onset_diagnostics": outputs.get("onset_diagnostics", {}),
    }



def clear_tabestimator_model_cache():
    TABESTIMATOR_MODEL_CACHE.clear()
    TABESTIMATOR_ONSET_MODEL_CACHE.clear()
    TABESTIMATOR_HIDONSET_MODEL_CACHE.clear()


# =============================================================================
# Audio feature extraction
# =============================================================================


def estimate_bpm_for_tabestimator(y: np.ndarray, sr: int) -> float:
    """
    Estimate tempo for runtime NPZ creation. You can override BPM from the endpoint.
    """
    try:
        if hasattr(librosa.feature, "tempo"):
            tempo = librosa.feature.tempo(y=y, sr=sr)
        else:
            tempo = librosa.beat.tempo(y=y, sr=sr)
        bpm = float(np.asarray(tempo).reshape(-1)[0])
    except Exception:
        bpm = 120.0

    if not np.isfinite(bpm) or bpm <= 0:
        bpm = 120.0

    return float(bpm)


def extract_tabestimator_features(y: np.ndarray, sr: int, config: Dict[str, Any]) -> np.ndarray:
    """
    Direct feature extraction for legacy chunked inference.

    For best training/inference parity, prefer the runtime NPZ path below:
        create_audio_only_tabestimator_npz -> add_actual_hand_positions_to_npz -> split_npz_for_tabestimator_inference
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)

    hop_length = int(_get_config_required(config, "hop_length"))
    input_feature_type = str(_get_config_required(config, "input_feature_type"))

    if input_feature_type == "cqt":
        cqt = librosa.cqt(
            y,
            sr=sr,
            hop_length=hop_length,
            n_bins=int(_get_config_required(config, "cqt_n_bins")),
            bins_per_octave=int(_get_config_required(config, "bins_per_octave")),
        )
        features = np.abs(cqt).T

    elif input_feature_type == "melspec":
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            hop_length=hop_length,
            n_mels=128,
            power=2.0,
        )
        features = mel.T

    else:
        raise ValueError(f"Unknown input_feature_type: {input_feature_type!r}")

    if features.ndim != 2 or features.shape[0] == 0:
        raise RuntimeError(f"Feature extraction produced invalid shape: {features.shape}")

    return features.astype(np.float32)


# =============================================================================
# Hand prior aggregation: CV finger frets -> TabEstimator base-position prior
# =============================================================================


def _normalize_max(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    if v.size == 0:
        return v
    vmax = float(np.max(v))
    if vmax > 0:
        v = v / vmax
    return v.astype(np.float32)


def _normalize_distribution(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    if v.size == 0:
        return v
    v = np.maximum(v, 0.0)
    s = float(np.sum(v))
    if s > 0:
        v = v / s
    return v.astype(np.float32)


def convert_hand_vector_to_model_dim(
    hand_vector: List[float] | np.ndarray,
    target_dim: int,
    fold_excess_to_last_bin: bool = True,
) -> np.ndarray:
    """
    Legacy converter from an app hand vector to the model dimension.

    The main inference path should prefer base-position conversion from finger_frets.
    This is only a fallback when no usable base position is available.
    """
    target_dim = int(target_dim)
    if target_dim <= 0:
        raise ValueError(f"target_dim must be positive, got {target_dim}")

    h = np.asarray(hand_vector, dtype=np.float32).reshape(-1)

    if len(h) == target_dim:
        out = h.copy()
    elif len(h) > target_dim:
        out = h[:target_dim].copy()
        if fold_excess_to_last_bin and len(h[target_dim:]) > 0:
            out[-1] = max(float(out[-1]), float(np.max(h[target_dim:])))
    else:
        out = np.zeros(target_dim, dtype=np.float32)
        out[: len(h)] = h

    return _normalize_max(out)


def soft_position_vector_from_base(
    base_position: Optional[float],
    target_dim: int,
    sigma: float = 1.0,
    floor: float = 1e-5,
) -> np.ndarray:
    """
    Convert a continuous base hand position into a soft distribution over hand-position bins.

    This mirrors create_phantom_handpos_npz.py hard_positions_to_soft(), except
    base_position may be fractional because it comes from CV.
    """
    target_dim = int(target_dim)
    if target_dim <= 0:
        raise ValueError(f"target_dim must be positive, got {target_dim}")

    base_position = _finite_float_or_none(base_position)
    if base_position is None:
        return np.zeros(target_dim, dtype=np.float32)

    sigma = float(sigma)
    if sigma <= 0:
        sigma = 1.0

    base_position = max(0.0, min(base_position, float(target_dim - 1)))

    grid = np.arange(target_dim, dtype=np.float32)
    dist = np.exp(-0.5 * ((grid - base_position) / sigma) ** 2)
    dist = dist + float(floor)
    return _normalize_distribution(dist)


def hand_base_position_from_entry(
    hb: Optional[Dict[str, Any]],
    hand_span: int = 4,
    target_dim: int = 20,
) -> Optional[float]:
    """
    Convert one CV hand bbox entry into the TabEstimator hand-position convention.

    CV gives center-ish information: hand_center_fret and finger_frets.
    The model was trained with a base position p, where p roughly covers:
        p, p+1, ..., p+hand_span-1

    Best case:
        base_position = min positive finger_fret

    Fallback:
        base_position = hand_center_fret - (hand_span - 1) / 2
    """
    if hb is None:
        return None

    target_dim = int(target_dim)
    hand_span = int(hand_span)

    cleaned_frets: List[float] = []
    for f in hb.get("finger_frets") or []:
        ff = _finite_float_or_none(f)
        if ff is not None:
            cleaned_frets.append(ff)

    positive_frets = [f for f in cleaned_frets if f > 0.25]

    if positive_frets:
        base_position = float(min(positive_frets))
    else:
        center = _finite_float_or_none(hb.get("hand_center_fret"))
        if center is None:
            pinky = _finite_float_or_none(hb.get("pinky_fret"))
            if pinky is None or pinky < 0:
                return None
            center = pinky
        base_position = float(center) - (float(hand_span) - 1.0) / 2.0

    return float(max(0.0, min(base_position, float(target_dim - 1))))


def get_hand_prior_at_time(
    hand_bboxes: List[Dict[str, Any]],
    t: float,
    pre: float = 0.05,
    post: float = 0.08,
    max_dt: float = 0.15,
    max_fret: int = 24,
) -> Tuple[List[float], Optional[float], Optional[Dict[str, Any]]]:
    """
    Aggregate actual CV hand priors around time t.

    Returns:
        hand_soft_vector, hand_center_fret, matched_hand_entry
    """
    if not hand_bboxes:
        return [0.0] * (int(max_fret) + 1), None, None

    t = float(t)

    windowed = [
        hb for hb in hand_bboxes
        if (t - float(pre)) <= float(hb.get("t", -1.0)) <= (t + float(post))
        and hb.get("hand_soft_vector") is not None
    ]

    if windowed:
        vecs = []
        for hb in windowed:
            v = np.asarray(hb.get("hand_soft_vector", []), dtype=np.float32).reshape(-1)
            if v.size > 0:
                vecs.append(v)

        if vecs:
            max_len = max(v.size for v in vecs)
            padded = np.zeros((len(vecs), max_len), dtype=np.float32)
            for i, v in enumerate(vecs):
                padded[i, : v.size] = v
            H = padded.mean(axis=0)
        else:
            H = np.zeros(int(max_fret) + 1, dtype=np.float32)

        mus = []
        for hb in windowed:
            mu = _finite_float_or_none(hb.get("hand_center_fret"))
            if mu is not None:
                mus.append(mu)
        mu = float(np.mean(mus)) if mus else None

        nearest = min(windowed, key=lambda hb: abs(float(hb.get("t", 0.0)) - t))
        return _normalize_max(H).tolist(), mu, nearest

    valid = [hb for hb in hand_bboxes if hb.get("hand_soft_vector") is not None]
    if not valid:
        return [0.0] * (int(max_fret) + 1), None, None

    nearest = min(valid, key=lambda hb: abs(float(hb.get("t", 0.0)) - t))
    dt = abs(float(nearest.get("t", 0.0)) - t)

    if dt > float(max_dt):
        return [0.0] * (int(max_fret) + 1), None, None

    H = np.asarray(nearest.get("hand_soft_vector", []), dtype=np.float32).reshape(-1)
    if H.size == 0:
        H = np.zeros(int(max_fret) + 1, dtype=np.float32)

    mu = _finite_float_or_none(nearest.get("hand_center_fret"))
    return _normalize_max(H).tolist(), mu, nearest


def hand_base_position_at_time(
    hand_bboxes: List[Dict[str, Any]],
    t: float,
    target_dim: int,
    hand_span: int = 4,
    pre: float = 0.05,
    post: float = 0.08,
    max_dt: float = 0.15,
) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    """
    Aggregate CV hand entries around time t and return a model-convention base position.
    """
    if not hand_bboxes:
        return None, None

    t = float(t)

    windowed = [
        hb for hb in hand_bboxes
        if (t - float(pre)) <= float(hb.get("t", -1.0)) <= (t + float(post))
    ]

    candidates: List[Tuple[float, Dict[str, Any]]] = []
    for hb in windowed:
        base = hand_base_position_from_entry(hb, hand_span=hand_span, target_dim=target_dim)
        if base is not None:
            candidates.append((base, hb))

    if candidates:
        base_position = float(np.median([b for b, _ in candidates]))
        matched = min([hb for _, hb in candidates], key=lambda x: abs(float(x.get("t", 0.0)) - t))
        return base_position, matched

    nearest = None
    nearest_dt = float("inf")
    nearest_base = None

    for hb in hand_bboxes:
        base = hand_base_position_from_entry(hb, hand_span=hand_span, target_dim=target_dim)
        if base is None:
            continue

        dt = abs(float(hb.get("t", 0.0)) - t)
        if dt < nearest_dt:
            nearest_dt = dt
            nearest = hb
            nearest_base = base

    if nearest is None or nearest_dt > float(max_dt):
        return None, None

    return float(nearest_base), nearest


def hand_prior_at_time_for_tabestimator(
    hand_bboxes: List[Dict[str, Any]],
    t: float,
    target_dim: int,
    hand_span: int = 4,
    sigma: float = 1.0,
    pre: float = 0.05,
    post: float = 0.08,
    max_dt: float = 0.15,
) -> Tuple[np.ndarray, Optional[float], Optional[Dict[str, Any]]]:
    """
    Return a TabEstimator-compatible hand-position prior at time t.

    Returns:
        H_model, base_position, matched_hand_entry
    """
    base_position, matched = hand_base_position_at_time(
        hand_bboxes=hand_bboxes,
        t=float(t),
        target_dim=int(target_dim),
        hand_span=int(hand_span),
        pre=pre,
        post=post,
        max_dt=max_dt,
    )

    H = soft_position_vector_from_base(
        base_position,
        target_dim=int(target_dim),
        sigma=float(sigma),
    )

    return H, base_position, matched


def _actual_hand_distribution_at_time(
    hand_bboxes: List[Dict[str, Any]],
    t: float,
    n_positions: int,
    max_dt: float = 0.35,
    sigma: float = 1.0,
    hand_span: int = 4,
    debug: bool = False,
) -> Tuple[np.ndarray, int, Optional[float], Optional[Dict[str, Any]]]:
    """
    Convert actual CV hand data at time t into a TabEstimator hand-position distribution.

    Important convention:
      - The model was trained with phantom hand_pos as BASE POSITION.
      - Prefer min positive finger_frets.
      - Fallback to center_fret - half hand span.
      - Last fallback to hand_soft_vector.
    """
    n_positions = int(n_positions)

    dbg: Dict[str, Any] = {
        "t": float(t),
        "matched": False,
        "source": "none",
        "center_fret": None,
        "base_position": None,
        "hard_index": -1,
        "sum": 0.0,
        "peak": 0.0,
    }

    H, mu, matched = get_hand_prior_at_time(
        hand_bboxes=hand_bboxes,
        t=float(t),
        pre=0.05,
        post=0.08,
        max_dt=float(max_dt),
        max_fret=24,
    )

    if matched is not None:
        matched_t = float(matched.get("t", -1.0))
        dbg["matched"] = True
        dbg["matched_t"] = matched_t
        dbg["dt"] = abs(matched_t - float(t))

    base_position = None

    if matched is not None:
        positive_finger_frets = []
        for fret in matched.get("finger_frets") or []:
            ff = _finite_float_or_none(fret)
            if ff is not None and ff > 0.25:
                positive_finger_frets.append(ff)

        if positive_finger_frets:
            base_position = float(np.min(positive_finger_frets))
            dbg["source"] = "min_positive_finger_fret"
            dbg["finger_frets"] = [float(x) for x in positive_finger_frets]

    if base_position is None and mu is not None:
        center_fret = _finite_float_or_none(mu)
        if center_fret is not None:
            base_position = center_fret - ((float(hand_span) - 1.0) / 2.0)
            dbg["source"] = "center_minus_half_span"
            dbg["center_fret"] = float(center_fret)

    if base_position is None:
        soft = convert_hand_vector_to_model_dim(
            H,
            target_dim=n_positions,
            fold_excess_to_last_bin=True,
        )
        soft = _normalize_distribution(soft)

        if soft.size == 0 or float(np.sum(soft)) <= 0:
            dbg["source"] = "no_valid_hand"
            return np.zeros(n_positions, dtype=np.float32), -1, None, dbg if debug else None

        hard_index = int(np.argmax(soft))
        dbg["source"] = "hand_soft_vector_fallback"
        dbg["base_position"] = float(hard_index)
        dbg["hard_index"] = hard_index
        dbg["sum"] = float(np.sum(soft))
        dbg["peak"] = float(np.max(soft))
        return soft.astype(np.float32), hard_index, float(hard_index), dbg if debug else None

    base_position = max(0.0, min(float(base_position), float(n_positions - 1)))
    soft = soft_position_vector_from_base(base_position, target_dim=n_positions, sigma=float(sigma))

    if soft.size == 0 or float(np.sum(soft)) <= 0:
        dbg["source"] = "empty_gaussian"
        return np.zeros(n_positions, dtype=np.float32), -1, float(base_position), dbg if debug else None

    hard_index = int(np.argmax(soft))
    dbg["base_position"] = float(base_position)
    dbg["hard_index"] = hard_index
    dbg["sum"] = float(np.sum(soft))
    dbg["peak"] = float(np.max(soft))

    return soft.astype(np.float32), hard_index, float(base_position), dbg if debug else None


def build_frame_hand_sequence_for_tabestimator(
    hand_bboxes: List[Dict[str, Any]],
    frame_count: int,
    chunk_start_sec: float,
    sr: int,
    hop_length: int,
    target_dim: int,
    hand_span: int = 4,
    sigma: float = 1.0,
    pre: float = 0.05,
    post: float = 0.08,
    max_dt: float = 0.15,
) -> Tuple[np.ndarray, List[Optional[float]]]:
    """
    Legacy helper for direct chunked inference. Runtime NPZ path is preferred.
    """
    frame_count = int(frame_count)
    target_dim = int(target_dim)

    seq = np.zeros((frame_count, target_dim), dtype=np.float32)
    base_positions: List[Optional[float]] = []

    for i in range(frame_count):
        t = float(chunk_start_sec) + (i * int(hop_length)) / float(sr)
        H, base_position, _ = hand_prior_at_time_for_tabestimator(
            hand_bboxes=hand_bboxes,
            t=t,
            target_dim=target_dim,
            hand_span=hand_span,
            sigma=sigma,
            pre=pre,
            post=post,
            max_dt=max_dt,
        )
        seq[i] = H
        base_positions.append(None if base_position is None else float(base_position))

    return seq.astype(np.float32), base_positions


def build_note_hand_sequence_for_tabestimator(
    hand_bboxes: List[Dict[str, Any]],
    note_count: int,
    chunk_start_sec: float,
    bpm: float,
    note_resolution: int,
    target_dim: int,
    hand_span: int = 4,
    sigma: float = 1.0,
    pre: float = 0.05,
    post: float = 0.08,
    max_dt: float = 0.15,
) -> Tuple[np.ndarray, List[Optional[float]]]:
    """
    Legacy helper for direct chunked inference. Runtime NPZ path is preferred.
    """
    note_count = int(note_count)
    target_dim = int(target_dim)

    seq = np.zeros((note_count, target_dim), dtype=np.float32)
    base_positions: List[Optional[float]] = []

    seconds_per_note_step = _note_step_duration_seconds(float(bpm), int(note_resolution))

    for i in range(note_count):
        t = float(chunk_start_sec) + (i + 0.5) * seconds_per_note_step
        H, base_position, _ = hand_prior_at_time_for_tabestimator(
            hand_bboxes=hand_bboxes,
            t=t,
            target_dim=target_dim,
            hand_span=hand_span,
            sigma=sigma,
            pre=pre,
            post=post,
            max_dt=max_dt,
        )
        seq[i] = H
        base_positions.append(None if base_position is None else float(base_position))

    return seq.astype(np.float32), base_positions


# =============================================================================
# Audio chunking: legacy direct inference
# =============================================================================


def _note_step_duration_seconds(bpm: float, note_resolution: int) -> float:
    """
    Match midi_to_numpy.py:

        note_dur = 60.0 / tempo / note_resolution * 4.0

    For note_resolution=16 and bpm=120:
        note step = 1/16 note = 0.125 sec
    """
    bpm = float(bpm)
    note_resolution = int(note_resolution)

    if bpm <= 0:
        raise ValueError(f"bpm must be positive, got {bpm}")
    if note_resolution <= 0:
        raise ValueError(f"note_resolution must be positive, got {note_resolution}")

    return 60.0 / bpm / float(note_resolution) * 4.0


def iter_tabestimator_audio_chunks(
    y: np.ndarray,
    sr: int,
    bpm: float,
    note_resolution: int,
    window_note_len: int,
    hop_note_len: Optional[int] = None,
) -> Iterable[Dict[str, Any]]:
    """
    Yield fixed note-grid chunks. The note head usually expects 64 note steps.
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if len(y) == 0:
        return

    if hop_note_len is None:
        hop_note_len = window_note_len

    seconds_per_note_step = _note_step_duration_seconds(float(bpm), int(note_resolution))
    window_samples = max(1, int(round(int(window_note_len) * seconds_per_note_step * int(sr))))
    hop_samples = max(1, int(round(int(hop_note_len) * seconds_per_note_step * int(sr))))

    start = 0
    chunk_index = 0

    while start < len(y):
        end = start + window_samples
        chunk = y[start:end]

        if len(chunk) < window_samples:
            chunk = np.pad(chunk, (0, window_samples - len(chunk)), mode="constant")

        yield {
            "chunk_index": int(chunk_index),
            "start_sample": int(start),
            "end_sample": int(min(end, len(y))),
            "start_time": float(start / float(sr)),
            "end_time": float(min(end, len(y)) / float(sr)),
            "audio": chunk.astype(np.float32),
            "note_len": int(window_note_len),
        }

        start += hop_samples
        chunk_index += 1


# =============================================================================
# Model inference / decoding
# =============================================================================


def tabestimator_logits_to_classes(
    note_pred: np.ndarray,
    rest_class: int = TABESTIMATOR_REST_CLASS,
    min_note_prob: Optional[float] = None,
    min_margin_vs_rest: Optional[float] = None,
) -> np.ndarray:
    """
    Convert note_pred (T, 6, 21) to hard classes.

    Default behavior is standard TabEstimator inference:
        argmax over the class axis.

    Optional thresholds can suppress weak non-rest notes for UI/debug use, but
    keep them disabled when you want standard checkpoint behavior.
    """
    x = np.asarray(note_pred, dtype=np.float32)

    if x.ndim != 3 or x.shape[1] != 6:
        raise ValueError(f"Expected note_pred shape (T, 6, C), got {x.shape}")

    cls = np.argmax(x, axis=2).astype(np.int64)

    if min_note_prob is None and min_margin_vs_rest is None:
        return cls

    # Compute probabilities only when thresholding is requested.
    sums = x.sum(axis=2, keepdims=True)
    if np.any(x < 0) or not np.allclose(float(sums.mean()), 1.0, atol=0.1):
        x = x - np.max(x, axis=2, keepdims=True)
        x = np.exp(x)
        x = x / np.maximum(np.sum(x, axis=2, keepdims=True), 1e-12)

    best_prob = np.max(x, axis=2)
    rest_prob = x[:, :, int(rest_class)]

    weak_note = cls != int(rest_class)

    if min_note_prob is not None:
        weak_note &= best_prob < float(min_note_prob)

    if min_margin_vs_rest is not None:
        weak_note |= (cls != int(rest_class)) & ((best_prob - rest_prob) < float(min_margin_vs_rest))

    cls[weak_note] = int(rest_class)
    return cls.astype(np.int64)

def run_tabestimator_hand_chunk_outputs(
    model,
    features: np.ndarray,
    bpm: float,
    frame_hand_pos: np.ndarray,
    note_hand_pos: np.ndarray,
    note_len: int,
    device: str,
    min_note_prob: Optional[float] = None,
    min_margin_vs_rest: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run one split/chunk and return both frame-level and note-level predictions.

    Returns:
        {
            "frame_classes": (T_frame, 6),
            "note_classes":  (T_note, 6),
            "olens": [...]
        }
    """
    features = np.asarray(features, dtype=np.float32)
    frame_hand_pos = np.asarray(frame_hand_pos, dtype=np.float32)
    note_hand_pos = np.asarray(note_hand_pos, dtype=np.float32)

    if features.ndim != 2:
        raise ValueError(f"features must have shape (T, F), got {features.shape}")
    if frame_hand_pos.ndim != 2:
        raise ValueError(f"frame_hand_pos must have shape (T, H), got {frame_hand_pos.shape}")
    if note_hand_pos.ndim != 2:
        raise ValueError(f"note_hand_pos must have shape (N, H), got {note_hand_pos.shape}")
    if frame_hand_pos.shape[0] != features.shape[0]:
        raise ValueError(
            f"frame_hand_pos length must match features length: "
            f"{frame_hand_pos.shape[0]} != {features.shape[0]}"
        )
    if note_hand_pos.shape[0] != int(note_len):
        raise ValueError(
            f"note_hand_pos length must match note_len: "
            f"{note_hand_pos.shape[0]} != {int(note_len)}"
        )

    input_features = torch.from_numpy(features).unsqueeze(0).to(device)
    frame_len = torch.tensor([features.shape[0]], dtype=torch.long, device=device)
    note_len_tensor = torch.tensor([int(note_len)], dtype=torch.long, device=device)
    bpm_tensor = torch.tensor([float(bpm)], dtype=torch.float32, device=device)
    frame_hand_tensor = torch.from_numpy(frame_hand_pos).unsqueeze(0).to(device)
    note_hand_tensor = torch.from_numpy(note_hand_pos).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        frame_pred, note_pred, olens = model(
            input_features.float(),
            frame_len,
            note_len_tensor,
            bpm_tensor,
            frame_hand_pos=frame_hand_tensor.float(),
            note_hand_pos=note_hand_tensor.float(),
        )

    frame_pred = torch.squeeze(frame_pred, 0).detach().cpu().numpy()
    note_pred = torch.squeeze(note_pred, 0).detach().cpu().numpy()

    if frame_pred.ndim != 3 or frame_pred.shape[1] != 6:
        raise RuntimeError(f"Unexpected frame_pred shape from model: {frame_pred.shape}")
    if note_pred.ndim != 3 or note_pred.shape[1] != 6:
        raise RuntimeError(f"Unexpected note_pred shape from model: {note_pred.shape}")

    frame_out_len = int(olens[0].item()) if olens is not None else int(frame_pred.shape[0])
    frame_pred = frame_pred[:frame_out_len]

    frame_classes = tabestimator_logits_to_classes(
        frame_pred,
        rest_class=TABESTIMATOR_REST_CLASS,
        min_note_prob=min_note_prob,
        min_margin_vs_rest=min_margin_vs_rest,
    )

    note_classes = tabestimator_logits_to_classes(
        note_pred,
        rest_class=TABESTIMATOR_REST_CLASS,
        min_note_prob=min_note_prob,
        min_margin_vs_rest=min_margin_vs_rest,
    )

    return {
        "frame_classes": frame_classes.astype(np.int64),
        "note_classes": note_classes.astype(np.int64),
        "olens": [int(frame_out_len)],
    }
    
def run_tabestimator_hand_chunk_full(
    model,
    features: np.ndarray,
    bpm: float,
    frame_hand_pos: np.ndarray,
    note_hand_pos: np.ndarray,
    note_len: int,
    device: str,
    min_note_prob: Optional[float] = None,
    min_margin_vs_rest: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Run one chunk/split through the model.

    Returns both:
      - note_classes:  shape (T_note, 6)
      - frame_classes: shape (T_frame, 6)
    """
    features = np.asarray(features, dtype=np.float32)
    frame_hand_pos = np.asarray(frame_hand_pos, dtype=np.float32)
    note_hand_pos = np.asarray(note_hand_pos, dtype=np.float32)

    if features.ndim != 2:
        raise ValueError(f"features must have shape (T, F), got {features.shape}")
    if frame_hand_pos.ndim != 2:
        raise ValueError(f"frame_hand_pos must have shape (T, H), got {frame_hand_pos.shape}")
    if note_hand_pos.ndim != 2:
        raise ValueError(f"note_hand_pos must have shape (N, H), got {note_hand_pos.shape}")
    if frame_hand_pos.shape[0] != features.shape[0]:
        raise ValueError(
            f"frame_hand_pos length must match features length: "
            f"{frame_hand_pos.shape[0]} != {features.shape[0]}"
        )
    if note_hand_pos.shape[0] != int(note_len):
        raise ValueError(
            f"note_hand_pos length must match note_len: "
            f"{note_hand_pos.shape[0]} != {int(note_len)}"
        )

    input_features = torch.from_numpy(features).unsqueeze(0).to(device)
    frame_len = torch.tensor([features.shape[0]], dtype=torch.long, device=device)
    note_len_tensor = torch.tensor([int(note_len)], dtype=torch.long, device=device)
    bpm_tensor = torch.tensor([float(bpm)], dtype=torch.float32, device=device)
    frame_hand_tensor = torch.from_numpy(frame_hand_pos).unsqueeze(0).to(device)
    note_hand_tensor = torch.from_numpy(note_hand_pos).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        frame_pred, note_pred, _ = model(
            input_features.float(),
            frame_len,
            note_len_tensor,
            bpm_tensor,
            frame_hand_pos=frame_hand_tensor.float(),
            note_hand_pos=note_hand_tensor.float(),
        )

    frame_pred = torch.squeeze(frame_pred, 0).detach().cpu().numpy()
    note_pred = torch.squeeze(note_pred, 0).detach().cpu().numpy()

    if note_pred.ndim != 3 or note_pred.shape[1] != 6:
        raise RuntimeError(f"Unexpected note_pred shape from model: {note_pred.shape}")

    if frame_pred.ndim != 3 or frame_pred.shape[1] != 6:
        raise RuntimeError(f"Unexpected frame_pred shape from model: {frame_pred.shape}")

    note_classes = tabestimator_logits_to_classes(
        note_pred,
        rest_class=TABESTIMATOR_REST_CLASS,
        min_note_prob=min_note_prob,
        min_margin_vs_rest=min_margin_vs_rest,
    )

    frame_classes = tabestimator_logits_to_classes(
        frame_pred,
        rest_class=TABESTIMATOR_REST_CLASS,
        min_note_prob=min_note_prob,
        min_margin_vs_rest=min_margin_vs_rest,
    )

    return {
        "note_classes": note_classes.astype(np.int64),
        "frame_classes": frame_classes.astype(np.int64),
    }    
    
def run_tabestimator_hand_chunk(
    model,
    features: np.ndarray,
    bpm: float,
    frame_hand_pos: np.ndarray,
    note_hand_pos: np.ndarray,
    note_len: int,
    device: str,
    min_note_prob: Optional[float] = None,
    min_margin_vs_rest: Optional[float] = None,
) -> np.ndarray:
    """
    Backward-compatible wrapper.

    Returns only note-level tab classes, as before.
    """
    outputs = run_tabestimator_hand_chunk_outputs(
        model=model,
        features=features,
        bpm=bpm,
        frame_hand_pos=frame_hand_pos,
        note_hand_pos=note_hand_pos,
        note_len=note_len,
        device=device,
        min_note_prob=min_note_prob,
        min_margin_vs_rest=min_margin_vs_rest,
    )

    return outputs["note_classes"]

# =============================================================================
# Output formatting
# =============================================================================


def tabestimator_classes_to_ascii(
    tab_classes: np.ndarray,
    step_width: int = 4,
    reverse_strings: bool = True,
    collapse_repeats: bool = False,
    rest_class: int = TABESTIMATOR_REST_CLASS,
) -> str:
    """
    Convert tab classes to a simple ASCII tablature block.

    tab_classes shape:
        (T, 6), low-E-first

    reverse_strings=True prints standard tab order:
        high e on top, low E on bottom
    """
    tab_classes = np.asarray(tab_classes)
    if tab_classes.ndim != 2 or tab_classes.shape[1] != 6:
        raise ValueError(f"Expected tab_classes shape (T, 6), got {tab_classes.shape}")

    T, _ = tab_classes.shape
    printable = tab_classes.copy()

    if collapse_repeats:
        prev = np.full((6,), rest_class, dtype=np.int64)
        for t in range(T):
            current = printable[t].copy()
            for s in range(6):
                if current[s] != rest_class and current[s] == prev[s]:
                    printable[t, s] = rest_class
            prev = current

    string_order = list(range(6))
    if reverse_strings:
        string_order = string_order[::-1]

    lines = []
    for s in string_order:
        line = ""
        for t in range(T):
            fret = int(printable[t, s])
            if fret == rest_class:
                token = "-" * int(step_width)
            else:
                txt = str(fret)
                token = "-" * max(0, int(step_width) - len(txt)) + txt
            line += token
        lines.append(line)

    return "\n".join(lines)

def tabestimator_frame_classes_to_events(
    tab_classes: np.ndarray,
    chunk_start_time: float,
    sr: int,
    hop_length: int,
    rest_class: int = TABESTIMATOR_REST_CLASS,
    collapse_repeats: bool = True,
    start_frame: int = 0,
) -> List[Dict[str, Any]]:
    """
    Convert frame-level class grid to event list.

    tab_classes shape:
        (T_frame, 6), low-E-first
    """
    tab_classes = np.asarray(tab_classes)

    if tab_classes.ndim != 2 or tab_classes.shape[1] != 6:
        raise ValueError(f"Expected tab_classes shape (T, 6), got {tab_classes.shape}")

    seconds_per_frame = float(hop_length) / float(sr)
    events: List[Dict[str, Any]] = []

    prev = np.full((6,), rest_class, dtype=np.int64)

    for frame_idx in range(tab_classes.shape[0]):
        current = tab_classes[frame_idx]
        t = float(chunk_start_time) + frame_idx * seconds_per_frame

        for string_idx in range(6):
            fret = int(current[string_idx])
            if fret == rest_class:
                continue
            if collapse_repeats and fret == int(prev[string_idx]):
                continue

            events.append({
                "time": float(t),
                "frame": int(frame_idx),
                "global_frame": int(start_frame) + int(frame_idx),
                "string_index_low_e_first": int(string_idx),
                "string_number_low_e_first": int(string_idx + 1),
                "fret": int(fret),
            })

        prev = current.copy()

    return events

def tabestimator_classes_to_events(
    tab_classes: np.ndarray,
    chunk_start_time: float,
    bpm: float,
    note_resolution: int,
    rest_class: int = TABESTIMATOR_REST_CLASS,
    collapse_repeats: bool = True,
) -> List[Dict[str, Any]]:
    """
    Convert class grid to event list.

    Backend convention:
        string_index_low_e_first = 0..5
        string_number_low_e_first = 1..6
    """
    tab_classes = np.asarray(tab_classes)
    if tab_classes.ndim != 2 or tab_classes.shape[1] != 6:
        raise ValueError(f"Expected tab_classes shape (T, 6), got {tab_classes.shape}")

    seconds_per_step = _note_step_duration_seconds(float(bpm), int(note_resolution))
    events: List[Dict[str, Any]] = []

    prev = np.full((6,), rest_class, dtype=np.int64)

    for step_idx in range(tab_classes.shape[0]):
        current = tab_classes[step_idx]
        t = float(chunk_start_time) + step_idx * seconds_per_step

        for string_idx in range(6):
            fret = int(current[string_idx])
            if fret == rest_class:
                continue
            if collapse_repeats and fret == int(prev[string_idx]):
                continue

            events.append({
                "time": float(t),
                "step": int(step_idx),
                "string_index_low_e_first": int(string_idx),
                "string_number_low_e_first": int(string_idx + 1),
                "fret": int(fret),
            })

        prev = current.copy()

    return events


def tabestimator_classes_to_vextab_notes(
    tab_classes: np.ndarray,
    rest_class: int = TABESTIMATOR_REST_CLASS,
    collapse_repeats: bool = True,
) -> str:
    events = tabestimator_classes_to_events(
        tab_classes,
        chunk_start_time=0.0,
        bpm=120.0,
        note_resolution=4,
        rest_class=rest_class,
        collapse_repeats=collapse_repeats,
    )

    notes = []
    for ev in events:
        string_idx = int(ev["string_index_low_e_first"])
        vex_string = 6 - string_idx
        notes.append(f"{int(ev['fret'])}/{vex_string}")

    return " ".join(notes) if notes else "=:|"


def events_to_vextab_text(events: List[Dict[str, Any]], width: int = 1240) -> str:
    """
    Convert decoded events to simple VexTab text.

    If events contain chord_group_id, notes with the same group id are rendered
    as a VexTab chord. Otherwise, events with the same global_step are grouped.
    This makes the string-onset decoder arpeggio-safe while still allowing
    near-simultaneous string onsets to display as chords.
    """
    grouped: Dict[Any, List[Tuple[int, int, str]]] = {}
    group_order: Dict[Any, Tuple[float, int]] = {}

    for idx, ev in enumerate(events or []):
        fret = ev.get("fret")
        string_idx = ev.get("string_index_low_e_first")
        if string_idx is None and ev.get("string_number_low_e_first") is not None:
            string_idx = int(ev["string_number_low_e_first"]) - 1

        if fret is None or string_idx is None:
            continue

        vex_string = 6 - int(string_idx)
        if not (1 <= vex_string <= 6 and int(fret) >= 0):
            continue

        if ev.get("chord_group_id") is not None:
            group_key = ("chord", int(ev["chord_group_id"]))
        elif ev.get("global_step") is not None:
            group_key = ("step", int(ev["global_step"]))
        else:
            group_key = ("event", int(idx))

        grouped.setdefault(group_key, [])
        grouped[group_key].append((int(string_idx), int(fret), f"{int(fret)}/{vex_string}"))

        order_time = float(ev.get("time", idx))
        order_step = int(ev.get("global_step", ev.get("step", idx)))
        if group_key not in group_order:
            group_order[group_key] = (order_time, order_step)

    notes = []
    for group_key in sorted(grouped.keys(), key=lambda k: group_order.get(k, (0.0, 0))):
        # Sort by physical string low-E -> high-e for stable output, and dedupe.
        items = sorted(set(grouped[group_key]), key=lambda item: (item[0], item[1]))
        note_tokens = [item[2] for item in items]
        if len(note_tokens) == 1:
            notes.append(note_tokens[0])
        elif len(note_tokens) > 1:
            notes.append("(" + ".".join(note_tokens) + ")")

    note_text = " ".join(notes) if notes else "=:|"

    return f"""
options width={int(width)}
tabstave notation=true
notes {note_text}
"""


# =============================================================================
# NPZ-based runtime inference helpers
# =============================================================================

def remove_short_tab_runs(
    classes: np.ndarray,
    rest_class: int = TABESTIMATOR_REST_CLASS,
    min_run: int = 2,
) -> np.ndarray:
    """
    Remove very short isolated non-rest runs per string.

    classes shape:
        (T, 6)

    Example with min_run=3:
        ----5----  ->  ---------
        ----555--  ->  ----555--
    """
    x = np.asarray(classes, dtype=np.int64).copy()

    if x.ndim != 2 or x.shape[1] != 6:
        raise ValueError(f"Expected classes shape (T, 6), got {x.shape}")

    min_run = int(min_run)
    if min_run <= 1:
        return x

    T = x.shape[0]

    for string_idx in range(6):
        y = x[:, string_idx]
        i = 0

        while i < T:
            fret = int(y[i])

            if fret == int(rest_class):
                i += 1
                continue

            start = i
            while i < T and int(y[i]) == fret:
                i += 1
            end = i

            run_len = end - start

            if run_len < min_run:
                y[start:end] = int(rest_class)

    return x.astype(np.int64)


def copy_events_with_chord_grouping(
    events: List[Dict[str, Any]],
    window_steps: int = 0,
    window_seconds: Optional[float] = None,
    step_key: str = "global_step",
) -> List[Dict[str, Any]]:
    """
    Return flat events, but force notes in the same chord group to share the same
    display grouping fields.

    This keeps the frontend simple: events that should render as one chord get
    the same `global_step`.

    Rules:
      - events are sorted by time/step
      - only different strings are merged into the same chord
      - neighboring steps/frames can be treated as one chord
      - original timing is preserved in original_time/original_global_step
    """
    if not events:
        return []

    window_steps = int(window_steps)
    window_seconds = None if window_seconds is None else float(window_seconds)

    ordered = sorted(
        [dict(ev) for ev in events],
        key=lambda ev: (
            float(ev.get("time", 0.0)),
            int(ev.get(step_key, ev.get("step", 0))),
            int(ev.get("string_index_low_e_first", 0)),
        ),
    )

    groups: List[List[Dict[str, Any]]] = []

    for ev in ordered:
        ev_time = float(ev.get("time", 0.0))
        ev_step = int(ev.get(step_key, ev.get("step", 0)))
        ev_string = int(
            ev.get(
                "string_index_low_e_first",
                int(ev.get("string_number_low_e_first", 1)) - 1,
            )
        )

        placed = False

        if groups:
            group = groups[-1]
            anchor = group[0]
            anchor_time = float(anchor.get("time", 0.0))
            anchor_step = int(anchor.get(step_key, anchor.get("step", 0)))

            used_strings = {
                int(
                    g.get(
                        "string_index_low_e_first",
                        int(g.get("string_number_low_e_first", 1)) - 1,
                    )
                )
                for g in group
            }

            close_by_step = abs(ev_step - anchor_step) <= window_steps
            close_by_time = (
                window_seconds is not None
                and abs(ev_time - anchor_time) <= window_seconds
            )

            if ev_string not in used_strings and (close_by_step or close_by_time):
                group.append(ev)
                placed = True

        if not placed:
            groups.append([ev])

    out: List[Dict[str, Any]] = []

    for chord_id, group in enumerate(groups):
        group = sorted(
            group,
            key=lambda ev: int(
                ev.get(
                    "string_index_low_e_first",
                    int(ev.get("string_number_low_e_first", 1)) - 1,
                )
            ),
        )

        anchor = min(
            group,
            key=lambda ev: (
                float(ev.get("time", 0.0)),
                int(ev.get(step_key, ev.get("step", 0))),
            ),
        )

        anchor_time = float(anchor.get("time", 0.0))
        anchor_step = int(anchor.get(step_key, anchor.get("step", 0)))

        for ev in group:
            new_ev = dict(ev)

            new_ev["chord_group_id"] = int(chord_id)
            new_ev["chord_size"] = int(len(group))
            new_ev["chord_group_size"] = int(len(group))
            new_ev["display_time"] = float(anchor_time)
            new_ev["display_step"] = int(anchor_step)
            new_ev["original_time"] = float(ev.get("time", anchor_time))
            new_ev["original_step"] = int(ev.get("step", anchor_step))

            if "global_step" in ev:
                new_ev["original_global_step"] = int(ev["global_step"])

            # Important:
            # The frontend already groups VexTab notes by global_step.
            # So we make all notes in this chord share the anchor global_step.
            new_ev["time"] = anchor_time
            new_ev["step"] = anchor_step

            if "global_step" in ev:
                new_ev["global_step"] = anchor_step

            out.append(new_ev)

    return out


# def tabestimator_frame_classes_to_events(
#     frame_classes: np.ndarray,
#     chunk_start_time: float,
#     sr: int,
#     hop_length: int,
#     rest_class: int = TABESTIMATOR_REST_CLASS,
#     collapse_repeats: bool = True,
# ) -> List[Dict[str, Any]]:
#     """
#     Convert frame-level class grid to events.

#     frame_classes shape:
#         (T_frame, 6), low-E-first
#     """
#     frame_classes = np.asarray(frame_classes, dtype=np.int64)

#     if frame_classes.ndim != 2 or frame_classes.shape[1] != 6:
#         raise ValueError(f"Expected frame_classes shape (T, 6), got {frame_classes.shape}")

#     events: List[Dict[str, Any]] = []
#     seconds_per_frame = float(hop_length) / float(sr)

#     prev = np.full((6,), int(rest_class), dtype=np.int64)

#     for frame_idx in range(frame_classes.shape[0]):
#         current = frame_classes[frame_idx]
#         t = float(chunk_start_time) + frame_idx * seconds_per_frame

#         for string_idx in range(6):
#             fret = int(current[string_idx])

#             if fret == int(rest_class):
#                 continue

#             if collapse_repeats and fret == int(prev[string_idx]):
#                 continue

#             events.append(
#                 {
#                     "time": float(t),
#                     "frame": int(frame_idx),
#                     "step": int(frame_idx),
#                     "string_index_low_e_first": int(string_idx),
#                     "string_number_low_e_first": int(string_idx + 1),
#                     "fret": int(fret),
#                 }
#             )

#         prev = current.copy()

#     return events

def _librosa_resample_compat(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)

    if int(orig_sr) == int(target_sr):
        return y.astype(np.float32)

    try:
        return librosa.resample(y, orig_sr=int(orig_sr), target_sr=int(target_sr)).astype(np.float32)
    except TypeError:
        return librosa.resample(y, int(orig_sr), int(target_sr)).astype(np.float32)


def _load_audio_mono_for_npz(audio_filename: str | os.PathLike) -> Tuple[int, np.ndarray]:
    y, sr_original = librosa.load(str(audio_filename), sr=None, mono=True)
    y = np.asarray(y, dtype=np.float32).reshape(-1)

    if y.size == 0:
        raise RuntimeError(f"Audio file is empty: {audio_filename}")

    return int(sr_original), y


def _process_cqt_for_npz(audio: np.ndarray, sr_original: int, config: Dict[str, Any]) -> np.ndarray:
    """
    Match midi_to_numpy.py process_cqt(). Returns shape (frame_len, cqt_n_bins).
    """
    down_sampling_rate = int(_get_config_required(config, "down_sampling_rate"))
    bins_per_octave = int(_get_config_required(config, "bins_per_octave"))
    n_bins = int(_get_config_required(config, "cqt_n_bins"))
    hop_length = int(_get_config_required(config, "hop_length"))

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio = librosa.util.normalize(audio)
    audio = _librosa_resample_compat(audio, sr_original, down_sampling_rate)

    cqt = np.abs(
        librosa.cqt(
            audio,
            hop_length=hop_length,
            sr=down_sampling_rate,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
        )
    )

    return cqt.T.astype(np.float32)


def _process_mel_spec_for_npz(audio: np.ndarray, sr_original: int, config: Dict[str, Any]) -> np.ndarray:
    """
    Match midi_to_numpy.py process_mel_spec(). Returns shape (frame_len, 128).
    """
    down_sampling_rate = int(_get_config_required(config, "down_sampling_rate"))
    hop_length = int(_get_config_required(config, "hop_length"))

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio = librosa.util.normalize(audio)
    audio = _librosa_resample_compat(audio, sr_original, down_sampling_rate)

    mel_spec = np.abs(
        librosa.feature.melspectrogram(
            y=audio,
            sr=down_sampling_rate,
            n_fft=2048,
            hop_length=hop_length,
        )
    )

    return mel_spec.T.astype(np.float32)


def _pad_or_trim_2d(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")

    target_len = int(target_len)
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")

    if x.shape[0] >= target_len:
        return x[:target_len].astype(np.float32)

    pad = np.zeros((target_len - x.shape[0], x.shape[1]), dtype=np.float32)
    return np.vstack([x.astype(np.float32), pad])


def _save_npz_atomic(npz_path_no_ext: str | os.PathLike, **payload) -> str:
    npz_path_no_ext = str(npz_path_no_ext)
    final_path = npz_path_no_ext + ".npz"
    tmp_path = npz_path_no_ext + ".tmp.npz"

    os.makedirs(os.path.dirname(final_path), exist_ok=True)

    try:
        np.savez_compressed(tmp_path, **payload)
        os.replace(tmp_path, final_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    return final_path


def _round_up_to_bar_multiple(note_steps: int, note_resolution: int) -> int:
    note_steps = int(note_steps)
    split_len = int(note_resolution) * 4

    if split_len <= 0:
        raise ValueError(f"Invalid split length from note_resolution={note_resolution}")

    if note_steps <= 0:
        return split_len

    return int(math.ceil(note_steps / float(split_len)) * split_len)


def create_audio_only_tabestimator_npz(
    audio_filename: str | os.PathLike,
    npz_path_no_ext: str | os.PathLike,
    config: Dict[str, Any],
    tempo: float,
) -> str:
    """
    Create a TabEstimator-style NPZ from audio only, without MIDI targets.

    This mirrors the feature/length logic from midi_to_numpy.py, but writes empty
    target arrays because this NPZ is for inference.
    """
    note_resolution = int(_get_config_required(config, "note_resolution"))
    down_sampling_rate = int(_get_config_required(config, "down_sampling_rate"))
    hop_length = int(_get_config_required(config, "hop_length"))

    sr_original, audio = _load_audio_mono_for_npz(audio_filename)

    cqt = _process_cqt_for_npz(audio, sr_original, config)
    log_cqt = librosa.amplitude_to_db(np.abs(cqt)).astype(np.float32)
    mel_spec = _process_mel_spec_for_npz(audio, sr_original, config)

    note_dur = _note_step_duration_seconds(float(tempo), note_resolution)
    audio_duration = float(len(audio)) / float(sr_original)

    raw_note_steps = int(round(audio_duration / note_dur))
    len_in_notes = _round_up_to_bar_multiple(raw_note_steps, note_resolution)
    feature_len = int(len_in_notes * note_dur * (float(down_sampling_rate) / float(hop_length)))

    if len_in_notes <= 0:
        raise RuntimeError(f"Invalid len_in_notes={len_in_notes} for {audio_filename}")
    if feature_len <= 0:
        raise RuntimeError(f"Invalid feature_len={feature_len} for {audio_filename}")

    cqt = _pad_or_trim_2d(cqt, feature_len)
    log_cqt = _pad_or_trim_2d(log_cqt, feature_len)
    mel_spec = _pad_or_trim_2d(mel_spec, feature_len)

    tab = np.zeros((len_in_notes, 6, 21), dtype=np.float32)
    tab[:, :, TABESTIMATOR_REST_CLASS] = 1.0

    tab_onset = np.zeros((len_in_notes, 6, 21), dtype=np.float32)
    tab_onset[:, :, TABESTIMATOR_REST_CLASS] = 1.0

    frame_tab = np.zeros((feature_len, 6, 21), dtype=np.float32)
    frame_tab[:, :, TABESTIMATOR_REST_CLASS] = 1.0

    frame_tab_onset = np.zeros((feature_len, 6, 21), dtype=np.float32)
    frame_tab_onset[:, :, TABESTIMATOR_REST_CLASS] = 1.0

    F0 = np.zeros((len_in_notes, 44), dtype=np.float32)
    F0_onset = np.zeros((len_in_notes, 44), dtype=np.float32)
    frame_F0 = np.zeros((feature_len, 44), dtype=np.float32)
    frame_F0_onset = np.zeros((feature_len, 44), dtype=np.float32)

    return _save_npz_atomic(
        npz_path_no_ext,
        cqt=cqt.astype(np.float32),
        log_cqt=log_cqt.astype(np.float32),
        mel_spec=mel_spec.astype(np.float32),
        tab=tab,
        tab_onset=tab_onset,
        frame_tab=frame_tab,
        frame_tab_onset=frame_tab_onset,
        F0=F0,
        F0_onset=F0_onset,
        frame_F0=frame_F0,
        frame_F0_onset=frame_F0_onset,
        tempo=np.asarray(float(tempo), dtype=np.float32),
        len_in_notes=np.asarray(int(len_in_notes), dtype=np.int64),
    )


def fill_missing_hand_positions_with_nearest(
    hand_pos: np.ndarray,
    hand_pos_index: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Replace missing all-zero hand-position rows with the nearest valid row.
    """
    hand_pos = np.asarray(hand_pos, dtype=np.float32).copy()
    hand_pos_index = np.asarray(hand_pos_index, dtype=np.int64).copy()

    if hand_pos.ndim != 2:
        raise ValueError(f"Expected hand_pos shape (T, H), got {hand_pos.shape}")

    row_sums = hand_pos.sum(axis=1)
    valid = (hand_pos_index >= 0) & (row_sums > 0)

    valid_indices = np.flatnonzero(valid)
    missing_indices = np.flatnonzero(~valid)

    if len(valid_indices) == 0:
        return hand_pos, hand_pos_index, 0

    filled = 0
    for i in missing_indices:
        nearest_j = valid_indices[np.argmin(np.abs(valid_indices - i))]
        hand_pos[i] = hand_pos[nearest_j]
        hand_pos_index[i] = hand_pos_index[nearest_j]
        filled += 1

    return hand_pos, hand_pos_index, filled


def add_actual_hand_positions_to_npz(
    input_npz_path: str | os.PathLike,
    output_npz_path: str | os.PathLike,
    hand_bboxes: List[Dict[str, Any]],
    config: Dict[str, Any],
    tempo: float,
    n_positions: int = 20,
    hand_span: int = 4,
    max_dt: float = 0.35,
    sigma: float = 1.0,
    overwrite: bool = True,
    debug: bool = True,
    fill_missing: bool = True,
) -> str:
    """
    Copy an audio-only TabEstimator NPZ and add actual hand-position arrays.
    """
    input_npz_path = str(input_npz_path)
    output_npz_path = str(output_npz_path)

    if os.path.exists(output_npz_path) and not overwrite:
        return output_npz_path

    if not os.path.isfile(input_npz_path):
        raise FileNotFoundError(f"Input NPZ not found: {input_npz_path}")

    note_resolution = int(_get_config_required(config, "note_resolution"))
    down_sampling_rate = int(_get_config_required(config, "down_sampling_rate"))
    hop_length = int(_get_config_required(config, "hop_length"))

    with np.load(input_npz_path, allow_pickle=True) as data:
        output = {key: data[key] for key in data.files}

    if "cqt" not in output:
        raise KeyError(f"{input_npz_path} does not contain key 'cqt'")
    if "len_in_notes" not in output:
        raise KeyError(f"{input_npz_path} does not contain key 'len_in_notes'")

    feature_len = int(output["cqt"].shape[0])
    len_in_notes = int(np.asarray(output["len_in_notes"]).reshape(-1)[0])
    n_positions = int(n_positions)
    note_dur = _note_step_duration_seconds(float(tempo), note_resolution)

    hand_pos = np.zeros((len_in_notes, n_positions), dtype=np.float32)
    hand_pos_index = np.full((len_in_notes,), -1, dtype=np.int64)
    frame_hand_pos = np.zeros((feature_len, n_positions), dtype=np.float32)
    frame_hand_pos_index = np.full((feature_len,), -1, dtype=np.int64)

    hand_times = []
    valid_mu_count = 0
    valid_finger_count = 0
    valid_vector_count = 0

    for hb in hand_bboxes or []:
        tt = _finite_float_or_none(hb.get("t", 0.0))
        if tt is not None:
            hand_times.append(tt)

        if _finite_float_or_none(hb.get("hand_center_fret")) is not None:
            valid_mu_count += 1

        if hb.get("finger_frets"):
            valid_finger_count += 1

        try:
            hv = np.asarray(hb.get("hand_soft_vector", []), dtype=np.float32)
            if hv.size and float(np.max(hv)) > 0:
                valid_vector_count += 1
        except Exception:
            pass

    debug_summary: Dict[str, Any] = {
        "input_npz_path": input_npz_path,
        "output_npz_path": output_npz_path,
        "tempo": float(tempo),
        "note_resolution": int(note_resolution),
        "note_dur": float(note_dur),
        "down_sampling_rate": int(down_sampling_rate),
        "hop_length": int(hop_length),
        "feature_len": int(feature_len),
        "len_in_notes": int(len_in_notes),
        "n_positions": int(n_positions),
        "hand_span": int(hand_span),
        "sigma": float(sigma),
        "max_dt": float(max_dt),
        "fill_missing": bool(fill_missing),
        "hand_bbox_count": int(len(hand_bboxes or [])),
        "hand_time_min": None if not hand_times else float(min(hand_times)),
        "hand_time_max": None if not hand_times else float(max(hand_times)),
        "valid_mu_count": int(valid_mu_count),
        "valid_finger_count": int(valid_finger_count),
        "valid_vector_count": int(valid_vector_count),
        "note_reason_counts": {},
        "frame_reason_counts": {},
        "note_valid_count": 0,
        "frame_valid_count": 0,
        "sample_note_debug": [],
        "sample_frame_debug": [],
    }

    if debug:
        print("[tabest handpos] ----------------------------------------", flush=True)
        print(f"[tabest handpos] input_npz={input_npz_path}", flush=True)
        print(f"[tabest handpos] output_npz={output_npz_path}", flush=True)
        print(f"[tabest handpos] hand_bbox_count={len(hand_bboxes or [])}", flush=True)
        print(f"[tabest handpos] valid_mu_count={valid_mu_count}", flush=True)
        print(f"[tabest handpos] valid_finger_count={valid_finger_count}", flush=True)
        print(f"[tabest handpos] valid_vector_count={valid_vector_count}", flush=True)
        print(f"[tabest handpos] hand_time_range={debug_summary['hand_time_min']}..{debug_summary['hand_time_max']}", flush=True)
        print(f"[tabest handpos] feature_len={feature_len}, len_in_notes={len_in_notes}", flush=True)
        print(f"[tabest handpos] note_dur={note_dur}", flush=True)
        print(f"[tabest handpos] max_dt={max_dt}, sigma={sigma}, hand_span={hand_span}", flush=True)

    for i in range(len_in_notes):
        t = (i + 0.5) * note_dur
        soft, hard_idx, base_position, dbg = _actual_hand_distribution_at_time(
            hand_bboxes=hand_bboxes,
            t=t,
            n_positions=n_positions,
            max_dt=max_dt,
            sigma=sigma,
            hand_span=hand_span,
            debug=True,
        )

        hand_pos[i] = soft
        hand_pos_index[i] = int(hard_idx)

        source = str((dbg or {}).get("source", "unknown"))
        debug_summary["note_reason_counts"][source] = debug_summary["note_reason_counts"].get(source, 0) + 1

        if hard_idx >= 0:
            debug_summary["note_valid_count"] += 1

        if debug and (i < 12 or i % 64 == 0):
            debug_summary["sample_note_debug"].append(dbg)

    for i in range(feature_len):
        t = (i * hop_length) / float(down_sampling_rate)
        soft, hard_idx, base_position, dbg = _actual_hand_distribution_at_time(
            hand_bboxes=hand_bboxes,
            t=t,
            n_positions=n_positions,
            max_dt=max_dt,
            sigma=sigma,
            hand_span=hand_span,
            debug=True,
        )

        frame_hand_pos[i] = soft
        frame_hand_pos_index[i] = int(hard_idx)

        source = str((dbg or {}).get("source", "unknown"))
        debug_summary["frame_reason_counts"][source] = debug_summary["frame_reason_counts"].get(source, 0) + 1

        if hard_idx >= 0:
            debug_summary["frame_valid_count"] += 1

        if debug and (i < 12 or i % 250 == 0):
            debug_summary["sample_frame_debug"].append(dbg)

    raw_note_missing = int(np.sum((hand_pos_index < 0) | (hand_pos.sum(axis=1) <= 0)))
    raw_frame_missing = int(np.sum((frame_hand_pos_index < 0) | (frame_hand_pos.sum(axis=1) <= 0)))

    if fill_missing:
        hand_pos, hand_pos_index, note_filled_count = fill_missing_hand_positions_with_nearest(hand_pos, hand_pos_index)
        frame_hand_pos, frame_hand_pos_index, frame_filled_count = fill_missing_hand_positions_with_nearest(frame_hand_pos, frame_hand_pos_index)
    else:
        note_filled_count = 0
        frame_filled_count = 0

    debug_summary["raw_note_missing_count"] = raw_note_missing
    debug_summary["raw_frame_missing_count"] = raw_frame_missing
    debug_summary["note_filled_nearest_count"] = int(note_filled_count)
    debug_summary["frame_filled_nearest_count"] = int(frame_filled_count)

    metadata = {
        "description": "actual soft hand positions inferred from MediaPipe hand landmarks and fretboard detection",
        "n_positions": int(n_positions),
        "hand_span": int(hand_span),
        "sigma": float(sigma),
        "max_dt": float(max_dt),
        "fill_missing": bool(fill_missing),
        "position_convention": "0..n_positions-1 are base-fret hand positions",
        "source": "offline_hand_bboxes.json",
        "tempo": float(tempo),
        "note_resolution": int(note_resolution),
        "rest_class": int(TABESTIMATOR_REST_CLASS),
    }

    output["hand_pos"] = hand_pos.astype(np.float32)
    output["hand_pos_index"] = hand_pos_index.astype(np.int64)
    output["frame_hand_pos"] = frame_hand_pos.astype(np.float32)
    output["frame_hand_pos_index"] = frame_hand_pos_index.astype(np.int64)
    output["hand_pos_positions"] = np.arange(n_positions, dtype=np.int64)
    output["hand_pos_metadata"] = np.asarray(json.dumps(metadata))
    output["hand_pos_debug_summary"] = np.asarray(json.dumps(debug_summary, indent=2))

    os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
    tmp_path = output_npz_path + ".tmp.npz"

    try:
        np.savez_compressed(tmp_path, **output)
        os.replace(tmp_path, output_npz_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    if debug:
        print("[tabest handpos] note_reason_counts:", debug_summary["note_reason_counts"], flush=True)
        print("[tabest handpos] frame_reason_counts:", debug_summary["frame_reason_counts"], flush=True)
        print(f"[tabest handpos] note_valid={debug_summary['note_valid_count']}/{len_in_notes}", flush=True)
        print(f"[tabest handpos] frame_valid={debug_summary['frame_valid_count']}/{feature_len}", flush=True)
        print(f"[tabest handpos] filled missing note hand positions: {note_filled_count}/{raw_note_missing}", flush=True)
        print(f"[tabest handpos] filled missing frame hand positions: {frame_filled_count}/{raw_frame_missing}", flush=True)

        if debug_summary["sample_note_debug"]:
            print("[tabest handpos] sample_note_debug:", flush=True)
            for row in debug_summary["sample_note_debug"][:8]:
                print("  ", row, flush=True)

        if debug_summary["sample_frame_debug"]:
            print("[tabest handpos] sample_frame_debug:", flush=True)
            for row in debug_summary["sample_frame_debug"][:8]:
                print("  ", row, flush=True)

        print("[tabest handpos] saved:", output_npz_path, flush=True)
        print("[tabest handpos] ----------------------------------------", flush=True)

    return output_npz_path


def _slice_array_for_split(arr: np.ndarray, start: int, end: int, target_len: int) -> np.ndarray:
    arr = np.asarray(arr)
    sliced = arr[start:end]

    if sliced.shape[0] >= target_len:
        return sliced[:target_len]

    pad_shape = (target_len - sliced.shape[0],) + arr.shape[1:]
    pad = np.zeros(pad_shape, dtype=arr.dtype)
    return np.concatenate([sliced, pad], axis=0)


def split_npz_for_tabestimator_inference(
    input_npz_path: str | os.PathLike,
    output_split_dir: str | os.PathLike,
    note_resolution: int,
    overwrite: bool = True,
) -> List[str]:
    """
    Split a full runtime NPZ into TabEstimator-style 4/4-bar split NPZ files.
    """
    input_npz_path = str(input_npz_path)
    output_split_dir = str(output_split_dir)

    if not os.path.isfile(input_npz_path):
        raise FileNotFoundError(f"Input NPZ not found: {input_npz_path}")

    os.makedirs(output_split_dir, exist_ok=True)

    with np.load(input_npz_path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}

    if "cqt" not in payload:
        raise KeyError(f"{input_npz_path} does not contain key 'cqt'")
    if "len_in_notes" not in payload:
        raise KeyError(f"{input_npz_path} does not contain key 'len_in_notes'")

    feature_len = int(payload["cqt"].shape[0])
    note_len = int(np.asarray(payload["len_in_notes"]).reshape(-1)[0])

    split_note_len = int(note_resolution) * 4
    if split_note_len <= 0:
        raise ValueError(f"Invalid note_resolution={note_resolution}")

    n_splits = int(math.ceil(note_len / float(split_note_len)))
    if n_splits <= 0:
        return []

    split_feature_len = int(round(float(feature_len) / float(n_splits)))
    if split_feature_len <= 0:
        raise RuntimeError(
            f"Invalid split_feature_len={split_feature_len}, feature_len={feature_len}, n_splits={n_splits}"
        )

    base_name = Path(input_npz_path).stem
    written_paths: List[str] = []

    note_level_keys = {"tab", "tab_onset", "F0", "F0_onset", "hand_pos", "hand_pos_index"}
    frame_level_keys = {
        "cqt",
        "log_cqt",
        "mel_spec",
        "frame_tab",
        "frame_tab_onset",
        "frame_F0",
        "frame_F0_onset",
        "frame_hand_pos",
        "frame_hand_pos_index",
    }
    passthrough_keys = {"tempo", "hand_pos_positions", "hand_pos_metadata", "hand_pos_debug_summary"}

    for split_idx in range(n_splits):
        note_start = split_note_len * split_idx
        note_end = split_note_len * (split_idx + 1)

        frame_start = split_feature_len * split_idx
        frame_end = feature_len if split_idx == n_splits - 1 else split_feature_len * (split_idx + 1)

        out: Dict[str, Any] = {}

        for key, value in payload.items():
            if key in note_level_keys:
                out[key] = _slice_array_for_split(value, note_start, note_end, split_note_len).astype(value.dtype, copy=False)
            elif key in frame_level_keys:
                out[key] = _slice_array_for_split(value, frame_start, frame_end, split_feature_len).astype(value.dtype, copy=False)
            elif key in passthrough_keys:
                out[key] = value
            elif key == "len_in_notes":
                out[key] = np.asarray(split_note_len, dtype=np.int64)
            else:
                arr = np.asarray(value)
                if arr.ndim == 0:
                    out[key] = value

        out_path = os.path.join(output_split_dir, f"{base_name}_0{split_idx}.npz")

        if os.path.exists(out_path) and not overwrite:
            written_paths.append(out_path)
            continue

        tmp_path = out_path + ".tmp.npz"
        try:
            np.savez_compressed(tmp_path, **out)
            os.replace(tmp_path, out_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        written_paths.append(out_path)

    return written_paths


# =============================================================================
# Debug / testing helpers for standard training-style NPZ files
# =============================================================================


def _resolve_existing_npz_path(npz_path: str | os.PathLike) -> str:
    p = Path(npz_path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    p = p.resolve()

    if not p.is_file():
        raise FileNotFoundError(f"NPZ file not found: {p}")

    return str(p)


def _feature_key_from_config(config: Dict[str, Any], feature_key: Optional[str] = None) -> str:
    if feature_key:
        return str(feature_key)

    input_feature_type = str(_get_config_required(config, "input_feature_type"))

    if input_feature_type == "cqt":
        return "cqt"
    if input_feature_type == "melspec":
        return "mel_spec"

    raise ValueError(f"Unknown input_feature_type: {input_feature_type!r}")


def _pad_or_trim_3d(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {x.shape}")

    target_len = int(target_len)
    if x.shape[0] >= target_len:
        return x[:target_len]

    pad_shape = (target_len - x.shape[0],) + x.shape[1:]
    pad = np.zeros(pad_shape, dtype=x.dtype)
    return np.concatenate([x, pad], axis=0)


def _pad_or_trim_1d(x: np.ndarray, target_len: int, pad_value: int = -1) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {x.shape}")

    target_len = int(target_len)
    if x.shape[0] >= target_len:
        return x[:target_len]

    pad = np.full((target_len - x.shape[0],), pad_value, dtype=x.dtype)
    return np.concatenate([x, pad], axis=0)


def _fit_2d_time_and_dim(
    x: np.ndarray,
    target_len: int,
    target_dim: int,
    normalize_rows: bool = False,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")

    x = _pad_or_trim_2d(x, int(target_len))

    if x.shape[1] == int(target_dim):
        out = x.astype(np.float32)
    elif x.shape[1] > int(target_dim):
        out = x[:, : int(target_dim)].astype(np.float32)
        excess = x[:, int(target_dim):]
        if excess.size:
            out[:, -1] = np.maximum(out[:, -1], np.max(excess, axis=1))
    else:
        out = np.zeros((x.shape[0], int(target_dim)), dtype=np.float32)
        out[:, : x.shape[1]] = x

    if normalize_rows:
        sums = out.sum(axis=1, keepdims=True)
        valid = sums[:, 0] > 0
        out[valid] = out[valid] / sums[valid]

    return out.astype(np.float32)


def tab_onehot_to_classes(tab: np.ndarray, rest_class: int = TABESTIMATOR_REST_CLASS) -> np.ndarray:
    tab = np.asarray(tab)

    if tab.ndim != 3 or tab.shape[1] != 6:
        raise ValueError(f"Expected tab shape (T, 6, 21), got {tab.shape}")
    if tab.shape[2] <= rest_class:
        raise ValueError(f"Expected last tab dimension to include rest class {rest_class}, got {tab.shape}")

    return np.argmax(tab, axis=2).astype(np.int64)


def evaluate_tab_class_grids(
    pred_classes: np.ndarray,
    target_classes: np.ndarray,
    rest_class: int = TABESTIMATOR_REST_CLASS,
) -> Dict[str, Any]:
    pred_classes = np.asarray(pred_classes, dtype=np.int64)
    target_classes = np.asarray(target_classes, dtype=np.int64)

    if pred_classes.shape != target_classes.shape:
        raise ValueError(
            f"pred_classes and target_classes must have same shape, got "
            f"{pred_classes.shape} vs {target_classes.shape}"
        )

    total_cells = int(np.prod(target_classes.shape))
    exact_cells = int(np.sum(pred_classes == target_classes))

    target_active = target_classes != int(rest_class)
    pred_active = pred_classes != int(rest_class)

    target_active_count = int(np.sum(target_active))
    pred_active_count = int(np.sum(pred_active))
    exact_active_matches = int(np.sum((pred_classes == target_classes) & target_active))

    missed = int(np.sum(target_active & ~pred_active))
    extra = int(np.sum(~target_active & pred_active))
    substitutions = int(np.sum(target_active & pred_active & (pred_classes != target_classes)))

    precision = exact_active_matches / pred_active_count if pred_active_count > 0 else 0.0
    recall = exact_active_matches / target_active_count if target_active_count > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0

    return {
        "shape": [int(x) for x in target_classes.shape],
        "total_cells": total_cells,
        "exact_cells": exact_cells,
        "all_cell_accuracy": exact_cells / total_cells if total_cells > 0 else 0.0,
        "target_active_count": target_active_count,
        "pred_active_count": pred_active_count,
        "exact_active_matches": exact_active_matches,
        "missed": missed,
        "extra": extra,
        "substitutions": substitutions,
        "precision_active": float(precision),
        "recall_active": float(recall),
        "f1_active": float(f1),
    }


def run_tabestimator_hand_npz_inference(
    model,
    npz_path: str | os.PathLike,
    config: Dict[str, Any],
    device: str,
    hand_pos_dim: int,
    bpm: Optional[float] = None,
    feature_key: Optional[str] = None,
    normalize_hand_rows: bool = False,
    start_time: float = 0.0,
    start_step: int = 0,
    start_frame: int = 0,
    step_width: int = 4,
    frame_step_width: int = 2,
    min_note_prob: Optional[float] = None,
    min_margin_vs_rest: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run TabEstimator on one runtime NPZ split.

    Returns both:
      - note-level transcription from note_pred
      - frame-level transcription from frame_pred
    """
    npz_path = _resolve_existing_npz_path(npz_path)

    with np.load(npz_path, allow_pickle=True) as data:
        available_keys = list(data.files)
        key = _feature_key_from_config(config, feature_key=feature_key)

        if key not in data.files:
            raise KeyError(f"Feature key {key!r} not found in NPZ. Available keys: {available_keys}")
        if "frame_hand_pos" not in data.files:
            raise KeyError(f"Hand key 'frame_hand_pos' not found in NPZ. Available keys: {available_keys}")
        if "hand_pos" not in data.files:
            raise KeyError(f"Hand key 'hand_pos' not found in NPZ. Available keys: {available_keys}")

        features = np.asarray(data[key], dtype=np.float32)
        frame_hand_pos = np.asarray(data["frame_hand_pos"], dtype=np.float32)
        note_hand_pos = np.asarray(data["hand_pos"], dtype=np.float32)

        if bpm is None:
            used_bpm = float(np.asarray(data["tempo"]).reshape(-1)[0]) if "tempo" in data.files else 120.0
        else:
            used_bpm = float(bpm)

        if "len_in_notes" in data.files:
            note_len = int(np.asarray(data["len_in_notes"]).reshape(-1)[0])
        else:
            note_len = int(note_hand_pos.shape[0])

        hand_pos_index = np.asarray(data["hand_pos_index"], dtype=np.int64) if "hand_pos_index" in data.files else None
        frame_hand_pos_index = np.asarray(data["frame_hand_pos_index"], dtype=np.int64) if "frame_hand_pos_index" in data.files else None

    if features.ndim != 2:
        raise ValueError(f"Expected features shape (T, F), got {features.shape}")

    frame_count = int(features.shape[0])
    note_len = int(note_len)

    frame_hand_pos = _fit_2d_time_and_dim(
        frame_hand_pos,
        target_len=frame_count,
        target_dim=int(hand_pos_dim),
        normalize_rows=normalize_hand_rows,
    )

    note_hand_pos = _fit_2d_time_and_dim(
        note_hand_pos,
        target_len=note_len,
        target_dim=int(hand_pos_dim),
        normalize_rows=normalize_hand_rows,
    )

    outputs = run_tabestimator_hand_chunk_outputs(
        model=model,
        features=features,
        bpm=used_bpm,
        frame_hand_pos=frame_hand_pos,
        note_hand_pos=note_hand_pos,
        note_len=note_len,
        device=device,
        min_note_prob=min_note_prob,
        min_margin_vs_rest=min_margin_vs_rest,
    )

    pred_classes = outputs["note_classes"]
    frame_pred_classes = outputs["frame_classes"]

    note_resolution = int(_get_config_required(config, "note_resolution"))
    sr = int(_get_config_required(config, "down_sampling_rate"))
    hop_length = int(_get_config_required(config, "hop_length"))

    events = tabestimator_classes_to_events(
        pred_classes,
        chunk_start_time=float(start_time),
        bpm=used_bpm,
        note_resolution=note_resolution,
        collapse_repeats=True,
    )

    for ev in events:
        ev["global_step"] = int(start_step) + int(ev["step"])

    frame_events = tabestimator_frame_classes_to_events(
        frame_pred_classes,
        chunk_start_time=float(start_time),
        sr=sr,
        hop_length=hop_length,
        collapse_repeats=True,
        start_frame=int(start_frame),
    )

    ascii_tab = tabestimator_classes_to_ascii(
        pred_classes,
        step_width=int(step_width),
        reverse_strings=True,
        collapse_repeats=True,
    )

    frame_ascii_tab = tabestimator_classes_to_ascii(
        frame_pred_classes,
        step_width=int(frame_step_width),
        reverse_strings=True,
        collapse_repeats=True,
    )

    return {
        "npz_path": npz_path,
        "available_keys": available_keys,
        "feature_key": key,
        "bpm": float(used_bpm),
        "note_resolution": int(note_resolution),
        "frame_count": int(frame_count),
        "frame_pred_len": int(frame_pred_classes.shape[0]),
        "note_len": int(note_len),
        "start_time": float(start_time),
        "start_step": int(start_step),
        "start_frame": int(start_frame),

        "feature_shape": [int(x) for x in features.shape],
        "frame_hand_pos_shape": [int(x) for x in frame_hand_pos.shape],
        "note_hand_pos_shape": [int(x) for x in note_hand_pos.shape],

        "hand_pos_index": None if hand_pos_index is None else hand_pos_index[:note_len].astype(int).tolist(),
        "frame_hand_pos_index": None if frame_hand_pos_index is None else frame_hand_pos_index[:frame_count].astype(int).tolist(),

        "pred_classes_low_e_first": pred_classes.astype(int).tolist(),
        "frame_pred_classes_low_e_first": frame_pred_classes.astype(int).tolist(),

        "events": events,
        "frame_events": frame_events,

        "vextab": events_to_vextab_text(events),
        "ascii_tab": ascii_tab,
        "frame_ascii_tab": frame_ascii_tab,
    }

def run_tabestimator_hand_npz(
    model,
    npz_path: str | os.PathLike,
    config: Dict[str, Any],
    device: str,
    hand_pos_dim: int,
    bpm: Optional[float] = None,
    feature_key: Optional[str] = None,
    normalize_hand_rows: bool = False,
    min_note_prob: Optional[float] = None,
    min_margin_vs_rest: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run TabEstimator on one standard training-style NPZ and compare prediction to target tab.
    """
    npz_path = _resolve_existing_npz_path(npz_path)

    with np.load(npz_path, allow_pickle=True) as data:
        available_keys = list(data.files)
        key = _feature_key_from_config(config, feature_key=feature_key)

        if key not in data.files:
            raise KeyError(f"Feature key {key!r} not found in NPZ. Available keys: {available_keys}")
        if "tab" not in data.files:
            raise KeyError(f"Target key 'tab' not found in NPZ. Available keys: {available_keys}")
        if "frame_hand_pos" not in data.files:
            raise KeyError(f"Hand key 'frame_hand_pos' not found in NPZ. Available keys: {available_keys}")
        if "hand_pos" not in data.files:
            raise KeyError(f"Hand key 'hand_pos' not found in NPZ. Available keys: {available_keys}")

        features = np.asarray(data[key], dtype=np.float32)
        target_tab = np.asarray(data["tab"], dtype=np.float32)
        frame_hand_pos = np.asarray(data["frame_hand_pos"], dtype=np.float32)
        note_hand_pos = np.asarray(data["hand_pos"], dtype=np.float32)

        if bpm is None:
            used_bpm = float(np.asarray(data["tempo"]).reshape(-1)[0]) if "tempo" in data.files else 120.0
        else:
            used_bpm = float(bpm)

        if "len_in_notes" in data.files:
            note_len = int(np.asarray(data["len_in_notes"]).reshape(-1)[0])
        else:
            note_len = int(target_tab.shape[0])

    if features.ndim != 2:
        raise ValueError(f"Expected features shape (T, F), got {features.shape}")
    if target_tab.ndim != 3 or target_tab.shape[1] != 6:
        raise ValueError(f"Expected target tab shape (T, 6, 21), got {target_tab.shape}")

    frame_count = int(features.shape[0])
    note_len = int(note_len)

    target_tab = _pad_or_trim_3d(target_tab, note_len)
    target_classes = tab_onehot_to_classes(target_tab)

    frame_hand_pos = _fit_2d_time_and_dim(
        frame_hand_pos,
        target_len=frame_count,
        target_dim=int(hand_pos_dim),
        normalize_rows=normalize_hand_rows,
    )
    note_hand_pos = _fit_2d_time_and_dim(
        note_hand_pos,
        target_len=note_len,
        target_dim=int(hand_pos_dim),
        normalize_rows=normalize_hand_rows,
    )

    pred_classes = run_tabestimator_hand_chunk(
        model=model,
        features=features,
        bpm=used_bpm,
        frame_hand_pos=frame_hand_pos,
        note_hand_pos=note_hand_pos,
        note_len=note_len,
        device=device,
        min_note_prob=min_note_prob,
        min_margin_vs_rest=min_margin_vs_rest,
    )

    min_len = min(int(pred_classes.shape[0]), int(target_classes.shape[0]))
    pred_eval_classes = pred_classes[:min_len]
    target_eval_classes = target_classes[:min_len]

    note_resolution = int(_get_config_required(config, "note_resolution"))

    pred_events = tabestimator_classes_to_events(
        pred_eval_classes,
        chunk_start_time=0.0,
        bpm=used_bpm,
        note_resolution=note_resolution,
        collapse_repeats=True,
    )
    target_events = tabestimator_classes_to_events(
        target_eval_classes,
        chunk_start_time=0.0,
        bpm=used_bpm,
        note_resolution=note_resolution,
        collapse_repeats=True,
    )

    evaluation = evaluate_tab_class_grids(
        pred_eval_classes,
        target_eval_classes,
        rest_class=TABESTIMATOR_REST_CLASS,
    )

    ascii_pred = tabestimator_classes_to_ascii(
        pred_eval_classes,
        step_width=4,
        reverse_strings=True,
        collapse_repeats=True,
    )
    ascii_target = tabestimator_classes_to_ascii(
        target_eval_classes,
        step_width=4,
        reverse_strings=True,
        collapse_repeats=True,
    )

    return {
        "npz_path": npz_path,
        "available_keys": available_keys,
        "feature_key": key,
        "bpm": float(used_bpm),
        "note_resolution": int(note_resolution),
        "frame_count": int(frame_count),
        "note_len": int(note_len),
        "eval_note_len": int(min_len),
        "feature_shape": [int(x) for x in features.shape],
        "frame_hand_pos_shape": [int(x) for x in frame_hand_pos.shape],
        "note_hand_pos_shape": [int(x) for x in note_hand_pos.shape],
        "target_tab_shape": [int(x) for x in target_tab.shape],
        "pred_classes_shape": [int(x) for x in pred_classes.shape],
        "evaluation": evaluation,
        "pred_classes_low_e_first": pred_eval_classes.astype(int).tolist(),
        "target_classes_low_e_first": target_eval_classes.astype(int).tolist(),
        "pred_events": pred_events,
        "target_events": target_events,
        "pred_vextab": events_to_vextab_text(pred_events),
        "target_vextab": events_to_vextab_text(target_events),
        "pred_ascii_tab": ascii_pred,
        "target_ascii_tab": ascii_target,
    }
