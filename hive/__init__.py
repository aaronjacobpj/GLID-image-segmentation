"""
hive - GLID Training Package

A clean, simple training pipeline for glacial lake segmentation.
KISS principle: Keep It Simple, Structured.
"""

from .config import Config, DataConfig, ModelConfig, TrainingConfig, LossConfig
from .data import GLIDDataset, create_dataloaders
from .models import UNet, SimpleCNN, DoubleConv
from .utils import (
    setup_device,
    setup_logger,
    DiceBCELoss,
    IoUMetric,
    PixelAccuracy,
    get_pred_mask,
    set_seed,
)
from .train import Trainer, create_optimizer, create_scheduler

__version__ = "1.0.0"
__author__ = "GLID Team"
__description__ = "Simple, optimized training pipeline for glacial lake segmentation"

__all__ = [
    "Config",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "LossConfig",
    "GLIDDataset",
    "create_dataloaders",
    "UNet",
    "SimpleCNN",
    "DoubleConv",
    "setup_device",
    "setup_logger",
    "DiceBCELoss",
    "IoUMetric",
    "PixelAccuracy",
    "get_pred_mask",
    "set_seed",
    "Trainer",
    "create_optimizer",
    "create_scheduler",
]
