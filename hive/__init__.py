"""
hive - GLID Training Package

A clean, simple training pipeline for glacial lake segmentation.
KISS principle: Keep It Simple, Structured.
"""

from .config import Config, DataConfig, ModelConfig, TrainingConfig, LossConfig, hash_run_config
from .data import GLIDDataset, create_dataloaders
from .deeplab import DeepLabV3Plus
from .transformer import SwinModel
from .unet import UNet, DoubleConv
from .utils import (
    setup_device,
    setup_logger,
    DiceBCELoss,
    IoUMetric,
    PixelAccuracy,
    PrecisionMetric,
    RecallMetric,
    F1ScoreMetric,
    DiceScoreMetric,
    DiceMetric,
    MeanIoUMetric,
    MeanDiceScoreMetric,
    MeanDiceMetric,
    get_pred_mask,
    set_seed,
)
from .train import Trainer, create_optimizer, create_scheduler, tune_hyperparameters

__version__ = "1.0.0"
__author__ = "GLID Team"
__description__ = "Simple, optimized training pipeline for glacial lake segmentation"

__all__ = [
    "Config",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "LossConfig",
    "hash_run_config",
    "GLIDDataset",
    "create_dataloaders",
    "UNet",
    "DoubleConv",
    "DeepLabV3Plus",
    "SwinModel",
    "setup_device",
    "setup_logger",
    "DiceBCELoss",
    "IoUMetric",
    "PixelAccuracy",
    "PrecisionMetric",
    "RecallMetric",
    "F1ScoreMetric",
    "DiceScoreMetric",
    "DiceMetric",
    "MeanIoUMetric",
    "MeanDiceScoreMetric",
    "MeanDiceMetric",
    "get_pred_mask",
    "set_seed",
    "Trainer",
    "create_optimizer",
    "create_scheduler",
    "tune_hyperparameters",
]
