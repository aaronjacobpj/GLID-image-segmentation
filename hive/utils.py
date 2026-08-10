"""
Training utilities: metrics, losses, device setup, logging.
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Tuple


def setup_device(device_str: str = "auto") -> str:
    """
    Setup device for training.
    
    Args:
        device_str: 'cuda', 'mps', 'cpu', or 'auto'
    
    Returns:
        Device string
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_str


def setup_logger(log_dir: Path, name: str = "training") -> logging.Logger:
    """Setup logger to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_dir / f"{name}.log")
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


class DiceBCELoss(nn.Module):
    """Combined Dice and Binary Cross Entropy loss for binary segmentation."""
    
    def __init__(
        self,
        weight_bce: float = 0.5,
        weight_dice: float = 0.5,
        smooth: float = 1e-5,
        pos_weight: float = 10.0,
        device: str = "cpu",
    ):
        """
        Initialize loss.
        
        Args:
            weight_bce: Weight for BCE component
            weight_dice: Weight for Dice component
            smooth: Smoothing factor for Dice
            pos_weight: Positive weight for BCE
            device: Device to move tensors to
        """
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.smooth = smooth
        self.device = device
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            logits: Model output (B, 1, H, W) without sigmoid
            targets: Ground truth binary mask (B, 1, H, W)
        
        Returns:
            Combined loss
        """
        # BCE loss
        bce_loss = self.bce(logits, targets)
        
        # Dice loss
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        dice_loss = 1.0 - dice_score
        
        # Combined
        return self.weight_bce * bce_loss + self.weight_dice * dice_loss


class IoUMetric:
    """Intersection over Union metric."""
    
    def __init__(self, smooth: float = 1e-6, threshold: float = 0.5):
        self.smooth = smooth
        self.threshold = threshold
    
    def __call__(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """
        Compute IoU.
        
        Args:
            logits: Model output (B, 1, H, W) or probabilities
            targets: Ground truth (B, 1, H, W)
        
        Returns:
            IoU score
        """
        if logits.shape[1] == 1:
            preds = (torch.sigmoid(logits) > self.threshold).float()
        else:
            preds = (logits > self.threshold).float()
        
        intersection = (preds * targets).sum().item()
        union = preds.sum().item() + targets.sum().item() - intersection
        
        return (intersection + self.smooth) / (union + self.smooth)


class PixelAccuracy:
    """Pixel accuracy metric."""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def __call__(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """
        Compute pixel accuracy.
        
        Args:
            logits: Model output
            targets: Ground truth
        
        Returns:
            Accuracy
        """
        preds = (torch.sigmoid(logits) > self.threshold).float()
        correct = (preds == targets).sum().item()
        total = targets.numel()
        return correct / total


def get_pred_mask(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Convert logits to binary predictions."""
    return (torch.sigmoid(logits) > threshold).float()


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
