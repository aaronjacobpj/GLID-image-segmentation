"""
Main training pipeline for GLID glacial lake segmentation.
KISS principle: Simple, clean, and focused.
"""

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, PolynomialLR
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional
import argparse

from .config import Config
from .data import create_dataloaders
from .models import UNet, SimpleCNN
from .utils import (
    setup_device,
    setup_logger,
    DiceBCELoss,
    IoUMetric,
    PixelAccuracy,
    get_pred_mask,
    set_seed,
)


class Trainer:
    """Simple training loop manager."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        config: Config,
        logger,
    ):
        """Initialize trainer."""
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.logger = logger
        
        # Metrics
        self.iou_metric = IoUMetric()
        self.accuracy_metric = PixelAccuracy()
        
        # History
        self.history = pd.DataFrame(
            columns=["epoch", "train_loss", "val_loss", "train_iou", "val_iou"]
        )
        
        # Best checkpoint
        self.best_val_iou = -float("inf")
        self.checkpoint_dir = config.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_iou = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            logits = self.model(images)
            loss = self.loss_fn(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            iou = self.iou_metric(logits.detach(), labels.detach())
            total_iou += iou
            num_batches += 1
            
            pbar.set_postfix({"loss": loss.item():.4f}, refresh=False)
        
        avg_loss = total_loss / num_batches
        avg_iou = total_iou / num_batches
        
        return avg_loss, avg_iou
    
    def validate(self) -> Tuple[float, float]:
        """Validate on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        total_iou = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)
                
                # Metrics
                total_loss += loss.item()
                iou = self.iou_metric(logits, labels)
                total_iou += iou
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_iou = total_iou / num_batches
        
        return avg_loss, avg_iou
    
    def train(self, epochs: int, scheduler=None, patience: Optional[int] = None):
        """
        Train model.
        
        Args:
            epochs: Number of epochs
            scheduler: Learning rate scheduler
            patience: Early stopping patience
        """
        self.logger.info(f"Starting training for {epochs} epochs")
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_iou = self.train_epoch()
            val_loss, val_iou = self.validate()
            
            # Learning rate scheduler
            if scheduler is not None:
                scheduler.step()
            
            # Log
            row = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_iou": train_iou,
                "val_iou": val_iou,
            }
            self.history = pd.concat([self.history, pd.DataFrame([row])], ignore_index=True)
            
            # Save best checkpoint
            if val_iou > self.best_val_iou:
                self.best_val_iou = val_iou
                patience_counter = 0
                self._save_checkpoint(epoch)
            else:
                patience_counter += 1
            
            # Log message
            msg = (
                f"Epoch [{epoch+1:3d}/{epochs}] | "
                f"Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} | "
                f"Train IoU: {train_iou:.4f} Val IoU: {val_iou:.4f}"
            )
            self.logger.info(msg)
            print(msg)
            
            # Early stopping
            if patience is not None and patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Save history
        history_path = self.config.logs_dir / "training_history.csv"
        self.history.to_csv(history_path, index=False)
        self.logger.info(f"Training complete. History saved to {history_path}")
    
    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"best_model_epoch{epoch+1}.pth"
        torch.save(self.model.state_dict(), checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")


def create_optimizer(model: nn.Module, config: Config) -> torch.optim.Optimizer:
    """Create optimizer."""
    if config.training.optimizer.lower() == "adam":
        return Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
    elif config.training.optimizer.lower() == "sgd":
        return SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer}")


def create_scheduler(optimizer: torch.optim.Optimizer, config: Config):
    """Create learning rate scheduler."""
    if config.training.scheduler is None:
        return None
    
    if config.training.scheduler.lower() == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=config.training.epochs,
        )
    elif config.training.scheduler.lower() == "polynomial":
        return PolynomialLR(
            optimizer,
            total_iters=config.training.epochs,
        )
    else:
        return None


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train GLID segmentation model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="project/data",
        help="Path to data directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=["unet", "simplecnn"],
        help="Model architecture",
    )
    args = parser.parse_args()
    
    # Setup
    set_seed(42)
    config = Config()
    config.data.data_dir = Path(args.data_dir)
    config.data.batch_size = args.batch_size
    config.training.epochs = args.epochs
    config.training.learning_rate = args.lr
    config.setup_directories()
    
    # Logger
    logger = setup_logger(config.logs_dir)
    logger.info(f"Config: {config}")
    
    # Device
    device = setup_device(config.training.device)
    logger.info(f"Using device: {device}")
    
    # Data
    logger.info(f"Loading data from {config.data.data_dir}")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=str(config.data.data_dir),
        batch_size=config.data.batch_size,
        image_size=config.data.image_size,
        num_workers=config.data.num_workers,
    )
    logger.info(
        f"Data: train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}, "
        f"test={len(test_loader.dataset)}"
    )
    
    # Model
    if args.model.lower() == "unet":
        model = UNet(
            in_channels=config.model.input_channels,
            out_channels=config.model.output_channels,
            encoder_channels=config.model.encoder_channels,
        )
    else:
        model = SimpleCNN(
            in_channels=config.model.input_channels,
            out_channels=config.model.output_channels,
        )
    logger.info(f"Model: {args.model}")
    
    # Loss, optimizer, scheduler
    loss_fn = DiceBCELoss(
        weight_bce=config.loss.weight_bce,
        weight_dice=config.loss.weight_dice,
        pos_weight=config.loss.pos_weight,
        device=device,
    )
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    logger.info(f"Optimizer: {config.training.optimizer}, LR: {config.training.learning_rate}")
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        config=config,
        logger=logger,
    )
    
    # Train
    trainer.train(
        epochs=config.training.epochs,
        scheduler=scheduler,
        patience=config.training.patience,
    )
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
