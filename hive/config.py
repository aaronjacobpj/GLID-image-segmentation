"""
Configuration settings for model training.
KISS principle: Keep It Simple, Structured.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from os import environ


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    data_dir: Path = Path(environ.get("DATA_DIR") or "data")
    image_size: int = 512
    batch_size: int = 16
    num_workers: int = 4
    train_split: float = 0.8
    val_split: float = 0.1
    random_seed: int = 42


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    input_channels: int = 3
    output_channels: int = 1
    encoder_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    scheduler: Optional[str] = "cosine"
    warmup_epochs: int = 2
    patience: int = 10
    save_epoch: bool = False
    device: str = "auto"  # 'cuda', 'mps', 'cpu', or 'auto'


@dataclass
class LossConfig:
    """Loss function configuration."""
    loss_type: str = "dice_bce"  # 'dice_bce', 'dice', 'bce'
    weight_bce: float = 0.4
    weight_dice: float = 0.6
    smooth: float = 1e-5
    pos_weight: float = 10.0


@dataclass
class Config:
    """Complete configuration object."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    # Paths
    output_dir: Path = Path(environ.get("OUTPUT_DIR") or "outputs")
    checkpoint_dir: Optional[Path] = None
    logs_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.checkpoint_dir is None:
            checkpoint_dir_env = environ.get("CHECKPOINT_DIR")
            self.checkpoint_dir = (
                Path(checkpoint_dir_env)
                if checkpoint_dir_env is not None
                else self.output_dir / "checkpoints"
            )
        if self.logs_dir is None:
            logs_dir_env = environ.get("LOGS_DIR")
            self.logs_dir = (
                Path(logs_dir_env)
                if logs_dir_env is not None
                else self.output_dir / "logs"
            )

    def setup_directories(self) -> None:
        """Create necessary directories."""
        for path in [self.output_dir, self.checkpoint_dir, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Run-fingerprinting helpers
# These functions encode the identity of a hyperparameter-tuning run so that
# completed runs can be skipped when resuming after a crash or power loss.
# ---------------------------------------------------------------------------

def hash_run_config(
    model_name: str,
    learning_rate: float,
    optimizer: str,
    batch_size: int,
    epochs: int,
    subset_size: Optional[int],
    seed: int,
) -> str:
    """Return a deterministic SHA-256 fingerprint of a tuning run's config.

    The hash is derived from every parameter that affects training so that
    two identical configurations always produce the same digest.  This lets
    ``tune_hyperparameters`` skip runs that already completed (even after a
    crash or power loss).

    Args:
        model_name:    Model architecture string (e.g. 'unet').
        learning_rate: Learning rate float.
        optimizer:     Optimizer name string.
        batch_size:    Mini-batch size.
        epochs:        Number of training epochs.
        subset_size:   Optional dataset size cap (None means full dataset).
        seed:          Random seed used for the run.

    Returns:
        Hex-encoded SHA-256 digest (64 characters).
    """
    config_dict: Dict[str, Any] = {
        "model": model_name.lower(),
        "lr": learning_rate,
        "optimizer": optimizer.lower(),
        "batch_size": batch_size,
        "epochs": epochs,
        "subset_size": subset_size,
        "seed": seed,
    }
    # json.dumps with sort_keys guarantees a stable byte sequence.
    payload = json.dumps(config_dict, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _completed_hashes_path(logs_dir: Path) -> Path:
    """Return the path of the file that stores completed run hashes."""
    return logs_dir / "hyperparameter_tuning_completed.txt"


def _load_completed_hashes(logs_dir: Path) -> Set[str]:
    """Load the set of hashes for already-completed tuning runs.

    Returns an empty set if the file does not yet exist.
    """
    path = _completed_hashes_path(logs_dir)
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _mark_run_completed(logs_dir: Path, run_hash: str) -> None:
    """Append *run_hash* to the completed-runs file.

    Appending (rather than rewriting) is crash-safe: a partial write leaves
    previous entries intact.
    """
    path = _completed_hashes_path(logs_dir)
    with open(path, "a", encoding="utf-8") as f:
        f.write(run_hash + "\n")
