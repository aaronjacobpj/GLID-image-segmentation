"""
Main training pipeline for GLID glacial lake segmentation.
KISS principle: Simple, clean, and focused.
"""

import logging
import csv
from copy import deepcopy
from itertools import product
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, PolynomialLR, _LRScheduler
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional, List, Dict, Sequence, Any
import argparse

from .config import (
    Config,
    hash_run_config,
    _completed_hashes_path,
    _load_completed_hashes,
    _mark_run_completed,
)
from .data import create_dataloaders
from .deeplab import DeepLabV3Plus
from .transformer import SwinModel
from .unet import UNet
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
    MeanIoUMetric,
    MeanDiceScoreMetric,
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
        logger: logging.Logger,
        run_name: str = "training",
        save_epoch: bool = False,
    ) -> None:
        """Initialize trainer."""
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.logger = logger
        self.run_name = run_name
        self.save_epoch = save_epoch
        
        # Metrics
        self.iou_metric = IoUMetric()
        self.mean_iou_metric = MeanIoUMetric()
        self.accuracy_metric = PixelAccuracy()
        self.precision_metric = PrecisionMetric()
        self.recall_metric = RecallMetric()
        self.f1_metric = F1ScoreMetric()
        self.dice_metric = DiceScoreMetric()
        self.mean_dice_metric = MeanDiceScoreMetric()
        
        # History (include accuracy, precision, recall, f1, dice, mean iou, mean dice)
        self.history = pd.DataFrame(
            columns=[
                "epoch",
                "train_loss",
                "val_loss",
                "train_iou",
                "val_iou",
                "train_mean_iou",
                "val_mean_iou",
                "train_accuracy",
                "val_accuracy",
                "train_precision",
                "val_precision",
                "train_recall",
                "val_recall",
                "train_f1",
                "val_f1",
                "train_dice",
                "val_dice",
                "train_mean_dice",
                "val_mean_dice",
            ]
        )
        
        # Best checkpoint
        self.best_val_iou = -float("inf")
        self.checkpoint_dir = self.config.checkpoint_dir
        assert self.checkpoint_dir is not None, "Checkpoint directory is None"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Prepare metrics CSV for live logging (include run_name)
        assert self.config.logs_dir is not None, "Logs directory is None"
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.config.logs_dir / f"{self.run_name}_metrics.csv"
        if not self.metrics_path.exists():
            with open(self.metrics_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "epoch",
                        "train_loss",
                        "val_loss",
                        "train_iou",
                        "val_iou",
                        "train_mean_iou",
                        "val_mean_iou",
                        "train_accuracy",
                        "val_accuracy",
                        "train_precision",
                        "val_precision",
                        "train_recall",
                        "val_recall",
                        "train_f1",
                        "val_f1",
                        "train_dice",
                        "val_dice",
                        "train_mean_dice",
                        "val_mean_dice",
                    ],
                )
                writer.writeheader()
    
    def train_epoch(self) -> Tuple[float, float, float, float, float, float, float, float, float]:
        """Train one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_iou = 0.0
        total_mean_iou = 0.0
        total_acc = 0.0
        total_prec = 0.0
        total_rec = 0.0
        total_f1 = 0.0
        total_dice = 0.0
        total_mean_dice = 0.0
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
            detached_logits = logits.detach()
            detached_labels = labels.detach()
            total_iou += self.iou_metric(detached_logits, detached_labels)
            total_mean_iou += self.mean_iou_metric(detached_logits, detached_labels)
            total_acc += self.accuracy_metric(detached_logits, detached_labels)
            total_prec += self.precision_metric(detached_logits, detached_labels)
            total_rec += self.recall_metric(detached_logits, detached_labels)
            total_f1 += self.f1_metric(detached_logits, detached_labels)
            total_dice += self.dice_metric(detached_logits, detached_labels)
            total_mean_dice += self.mean_dice_metric(detached_logits, detached_labels)
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"}, refresh=False)
            
        avg_loss = total_loss / num_batches
        avg_iou = total_iou / num_batches
        avg_mean_iou = total_mean_iou / num_batches
        avg_acc = total_acc / num_batches
        avg_prec = total_prec / num_batches
        avg_rec = total_rec / num_batches
        avg_f1 = total_f1 / num_batches
        avg_dice = total_dice / num_batches
        avg_mean_dice = total_mean_dice / num_batches

        return (
            avg_loss,
            avg_iou,
            avg_mean_iou,
            avg_acc,
            avg_prec,
            avg_rec,
            avg_f1,
            avg_dice,
            avg_mean_dice,
        )
    
    def validate(self) -> Tuple[float, float, float, float, float, float, float, float, float]:
        """Validate on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        total_iou = 0.0
        total_mean_iou = 0.0
        total_acc = 0.0
        total_prec = 0.0
        total_rec = 0.0
        total_f1 = 0.0
        total_dice = 0.0
        total_mean_dice = 0.0
        num_batches = 0
        
        with torch.inference_mode():
            pbar = tqdm(self.val_loader, desc="Validation")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)
                
                # Metrics
                total_loss += loss.item()
                total_iou += self.iou_metric(logits, labels)
                total_mean_iou += self.mean_iou_metric(logits, labels)
                total_acc += self.accuracy_metric(logits, labels)
                total_prec += self.precision_metric(logits, labels)
                total_rec += self.recall_metric(logits, labels)
                total_f1 += self.f1_metric(logits, labels)
                total_dice += self.dice_metric(logits, labels)
                total_mean_dice += self.mean_dice_metric(logits, labels)
                num_batches += 1
        
            avg_loss = total_loss / num_batches
            avg_iou = total_iou / num_batches
            avg_mean_iou = total_mean_iou / num_batches
            avg_acc = total_acc / num_batches
            avg_prec = total_prec / num_batches
            avg_rec = total_rec / num_batches
            avg_f1 = total_f1 / num_batches
            avg_dice = total_dice / num_batches
            avg_mean_dice = total_mean_dice / num_batches

            return (
                avg_loss,
                avg_iou,
                avg_mean_iou,
                avg_acc,
                avg_prec,
                avg_rec,
                avg_f1,
                avg_dice,
                avg_mean_dice,
            )
    
    def train(self, epochs: int, scheduler: Optional[_LRScheduler] = None, patience: Optional[int] = None) -> None:
        """
            set_seed(args.seed)
        
        Args:
            epochs: Number of epochs
            scheduler: Learning rate scheduler
            patience: Early stopping patience
        """
        self.logger.info(f"Starting training for {epochs} epochs")
        patience_counter = 0
        
        for epoch in range(epochs):
            (
                train_loss,
                train_iou,
                train_mean_iou,
                train_acc,
                train_prec,
                train_rec,
                train_f1,
                train_dice,
                train_mean_dice,
            ) = self.train_epoch()
            (
                val_loss,
                val_iou,
                val_mean_iou,
                val_acc,
                val_prec,
                val_rec,
                val_f1,
                val_dice,
                val_mean_dice,
            ) = self.validate()
            
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
                "train_mean_iou": train_mean_iou,
                "val_mean_iou": val_mean_iou,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "train_precision": train_prec,
                "val_precision": val_prec,
                "train_recall": train_rec,
                "val_recall": val_rec,
                "train_f1": train_f1,
                "val_f1": val_f1,
                "train_dice": train_dice,
                "val_dice": val_dice,
                "train_mean_dice": train_mean_dice,
                "val_mean_dice": val_mean_dice,
            }
            self.history = pd.concat([self.history, pd.DataFrame([row])], ignore_index=True)
            # Append metrics to live CSV
            try:
                with open(self.metrics_path, "a", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "epoch",
                            "train_loss",
                            "val_loss",
                            "train_iou",
                            "val_iou",
                            "train_mean_iou",
                            "val_mean_iou",
                            "train_accuracy",
                            "val_accuracy",
                            "train_precision",
                            "val_precision",
                            "train_recall",
                            "val_recall",
                            "train_f1",
                            "val_f1",
                            "train_dice",
                            "val_dice",
                            "train_mean_dice",
                            "val_mean_dice",
                        ],
                    )
                    writer.writerow(row)
            except Exception as e:
                self.logger.warning(f"Failed to write metrics CSV row: {e}")
            
            # Save best checkpoint
            if val_iou > self.best_val_iou:
                self.best_val_iou = val_iou
                patience_counter = 0
                self._save_checkpoint(epoch if self.save_epoch else None)
            else:
                patience_counter += 1
            
            # Log message
            msg = (
                f"Epoch [{epoch+1:3d}/{epochs}] | "
                f"Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} | "
                f"Train IoU: {train_iou:.4f} Val IoU: {val_iou:.4f} | "
                f"Train mIoU: {train_mean_iou:.4f} Val mIoU: {val_mean_iou:.4f} | "
                f"Train Dice: {train_dice:.4f} Val Dice: {val_dice:.4f} | "
                f"Train mDice: {train_mean_dice:.4f} Val mDice: {val_mean_dice:.4f} | "
                f"Train Prec: {train_prec:.4f} Val Prec: {val_prec:.4f} | "
                f"Train Rec: {train_rec:.4f} Val Rec: {val_rec:.4f} | "
                f"Train F1: {train_f1:.4f} Val F1: {val_f1:.4f}"
            )
            self.logger.info(msg)
            print(msg)
            
            # Early stopping
            if patience is not None and patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break

        assert self.config.logs_dir is not None, "Logs directory is None"

        # Save history (include run_name)
        history_path = self.config.logs_dir / f"{self.run_name}_training_history.csv"
        self.history.to_csv(history_path, index=False)
        self.logger.info(f"Training complete. History saved to {history_path}")
    
    def _save_checkpoint(self, epoch: Optional[int] = None):
        """Save model checkpoint."""

        assert self.config.checkpoint_dir is not None, "Checkpoint directory is None"
        
        # Handle case where epoch is None (e.g. for early stopping)
        state = f"_epoch{epoch+1}" if epoch is not None else ""
        checkpoint_path = self.config.checkpoint_dir / f"{self.run_name}_best_model{state}.pth"
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
    elif config.training.optimizer.lower() == "adamw":
        return AdamW(
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


def create_scheduler(optimizer: torch.optim.Optimizer, config: Config) -> Optional[_LRScheduler]:
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


def build_run_name(
    model_name: str,
    learning_rate: float,
    optimizer: str,
    batch_size: int,
    seed: Optional[int] = None,
    suffix: Optional[str] = None,
) -> str:
    """Build a stable basename for logs, checkpoints, and metrics.

    Includes model, optimizer, lr, batch size, and optional seed/suffix.
    """
    lr_str = f"{learning_rate:.0e}".replace("+0", "").replace("-0", "-")
    parts = [model_name, optimizer, f"lr{lr_str}", f"bs{batch_size}"]
    if seed is not None:
        parts.append(f"s{seed}")
    if suffix:
        parts.append(str(suffix))
    return "_".join(parts)


def build_model(model_name: str, config: Config) -> nn.Module:
    """Construct a model by name."""
    model_name = model_name.lower()
    if model_name == "unet":
        return UNet(
            in_channels=config.model.input_channels,
            out_channels=config.model.output_channels,
            encoder_channels=config.model.encoder_channels,
        )
    if model_name == "deeplab":
        return DeepLabV3Plus(num_classes=config.model.output_channels)
    if model_name == "swin":
        return SwinModel(
            channels=256,
            out_size=config.data.image_size,
            classes=config.model.output_channels,
            pretrained=False,
        )
    raise ValueError(f"Unknown model architecture: {model_name}")


def tune_hyperparameters(
    config: Config,
    data_dir: str,
    train_dir: Optional[str],
    val_dir: Optional[str],
    test_dir: Optional[str],
    subset_size: Optional[int],
    epochs: int,
    model_names: Sequence[str],
    learning_rates: Sequence[float],
    optimizers: Sequence[str],
    batch_sizes: Sequence[int],
    patience: Optional[int] = None,
    seed: int = 42,
    save_epoch: bool = False,
) -> None:
    """Run a grid search over model and training hyperparameters."""
    config.setup_directories()
    assert config.logs_dir is not None, "Logs directory is None"
    assert config.checkpoint_dir is not None, "Checkpoint directory is None"

    summary_path = config.logs_dir / "hyperparameter_tuning_summary.csv"

    # Load hashes of runs that already completed so we can skip them on resume.
    completed_hashes = _load_completed_hashes(config.logs_dir)
    if completed_hashes:
        logging.getLogger(__name__).info(
            f"Resuming tuning — {len(completed_hashes)} run(s) already completed."
        )

    if not summary_path.exists():
        with open(summary_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "run_name",
                    "model",
                    "optimizer",
                    "learning_rate",
                    "batch_size",
                    "epochs",
                    "subset_size",
                    "best_val_iou",
                    "best_val_loss",
                    "train_loss",
                    "val_loss",
                    "train_iou",
                    "val_iou",
                    "train_mean_iou",
                    "val_mean_iou",
                    "train_accuracy",
                    "val_accuracy",
                    "train_precision",
                    "val_precision",
                    "train_recall",
                    "val_recall",
                    "train_f1",
                    "val_f1",
                    "train_dice",
                    "val_dice",
                    "train_mean_dice",
                    "val_mean_dice",
                ],
            )
            writer.writeheader()

    for model_name, lr, optimizer_name, batch_size in product(
        model_names,
        learning_rates,
        optimizers,
        batch_sizes,
    ):
        # ------------------------------------------------------------------
        # Skip this configuration if it was already completed.
        # ------------------------------------------------------------------
        run_hash = hash_run_config(
            model_name=model_name,
            learning_rate=lr,
            optimizer=optimizer_name,
            batch_size=batch_size,
            epochs=epochs,
            subset_size=subset_size,
            seed=seed,
        )
        if run_hash in completed_hashes:
            logging.getLogger(__name__).info(
                f"Skipping already-completed run: model={model_name}, "
                f"optimizer={optimizer_name}, lr={lr}, batch_size={batch_size} "
                f"[hash={run_hash[:12]}...]"
            )
            continue

        # Set up config for this run
        config_copy = deepcopy(config)
        config_copy.data.batch_size = batch_size
        config_copy.training.epochs = epochs
        config_copy.training.learning_rate = lr
        config_copy.training.optimizer = optimizer_name
        config_copy.setup_directories()
        assert config_copy.logs_dir is not None, "Logs directory is None"
        assert config_copy.checkpoint_dir is not None, "Checkpoint directory is None"

        # Apply seed for reproducibility per run
        set_seed(seed)

        run_name = build_run_name(model_name, lr, optimizer_name, batch_size, seed=seed)
        logger = setup_logger(config_copy.logs_dir, name=run_name)
        logger.info(
            f"Hyperparameter tuning job: model={model_name}, optimizer={optimizer_name}, "
            f"lr={lr}, batch_size={batch_size}, epochs={epochs}, subset_size={subset_size}"
        )

        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=str(Path(data_dir)) if train_dir is None else None,
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            batch_size=config_copy.data.batch_size,
            image_size=config_copy.data.image_size,
            num_workers=config_copy.data.num_workers,
            subset_size=subset_size,
        )

        model = build_model(model_name, config_copy)
        loss_fn = DiceBCELoss(
            weight_bce=config_copy.loss.weight_bce,
            weight_dice=config_copy.loss.weight_dice,
            pos_weight=config_copy.loss.pos_weight,
            device=setup_device(config_copy.training.device),
        )
        optimizer = create_optimizer(model, config_copy)
        scheduler = create_scheduler(optimizer, config_copy)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=setup_device(config_copy.training.device),
            config=config_copy,
            logger=logger,
            run_name=run_name,
            save_epoch=save_epoch,
        )
        trainer.train(epochs=epochs, scheduler=scheduler, patience=patience)

        final_row = trainer.history.iloc[-1].to_dict() if not trainer.history.empty else {}
        row: Dict[str, Any] = {
            "run_name": run_name,
            "model": model_name,
            "optimizer": optimizer_name,
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "subset_size": subset_size,
            "best_val_iou": trainer.best_val_iou,
            "best_val_loss": final_row.get("val_loss", None),
            "train_loss": final_row.get("train_loss", None),
            "val_loss": final_row.get("val_loss", None),
            "train_iou": final_row.get("train_iou", None),
            "val_iou": final_row.get("val_iou", None),
            "train_mean_iou": final_row.get("train_mean_iou", None),
            "val_mean_iou": final_row.get("val_mean_iou", None),
            "train_accuracy": final_row.get("train_accuracy", None),
            "val_accuracy": final_row.get("val_accuracy", None),
            "train_precision": final_row.get("train_precision", None),
            "val_precision": final_row.get("val_precision", None),
            "train_recall": final_row.get("train_recall", None),
            "val_recall": final_row.get("val_recall", None),
            "train_f1": final_row.get("train_f1", None),
            "val_f1": final_row.get("val_f1", None),
            "train_dice": final_row.get("train_dice", None),
            "val_dice": final_row.get("val_dice", None),
            "train_mean_dice": final_row.get("train_mean_dice", None),
            "val_mean_dice": final_row.get("val_mean_dice", None),
        }
        with open(summary_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)

        # Mark this configuration as done so it is skipped on resume.
        _mark_run_completed(config.logs_dir, run_hash)
        completed_hashes.add(run_hash)
        logger.info(
            f"Completed tuning run {run_name} [hash={run_hash[:12]}...]. "
            f"Summary saved to {summary_path}"
        )


def main(argv: Optional[List[str]] = None) -> None:
    """Main training pipeline."""
    config = Config()
    parser = argparse.ArgumentParser(description="Train GLID segmentation model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(config.data.data_dir),
        help="Path to data directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.data.batch_size,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.training.epochs,
        help="Number of epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config.training.learning_rate,
        help="Learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.data.random_seed,
        help="Random seed",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=config.training.optimizer,
        choices=["adam", "adamw", "sgd"],
        help="Optimizer",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=["unet", "deeplab", "swin"],
        help="Model architecture",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=None,
        help="Path to training data folder containing images/ and labels/",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default=None,
        help="Path to validation data folder containing images/ and labels/",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help="Path to test data folder containing images/ and labels/",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Use only the first N samples from each dataset",
    )
    parser.add_argument(
        "--save-epoch",
        action="store_true",
        default=config.training.save_epoch,
        help="Save checkpoint filenames with epoch number instead of overwriting",
    )
    args = parser.parse_args(argv)
    
    # Setup
    set_seed(args.seed)
    config = Config()
    config.data.data_dir = Path(args.data_dir)
    config.data.batch_size = args.batch_size
    config.training.epochs = args.epochs
    config.training.learning_rate = args.lr
    config.training.optimizer = args.optimizer
    config.training.save_epoch = args.save_epoch
    config.setup_directories()
    
    # Construct run name (model + learning rate + optimizer + batch + seed) and Logger
    model_name = args.model.lower()
    run_name = build_run_name(
        model_name,
        config.training.learning_rate,
        config.training.optimizer,
        config.data.batch_size,
        seed=args.seed,
    )
    assert config.logs_dir is not None, "Logs directory is None"
    logger = setup_logger(config.logs_dir, name=run_name)
    logger.info(f"Config: {config}")
    
    # Device
    device = setup_device(config.training.device)
    logger.info(f"Using device: {device}")
    
    # Data
    logger.info(f"Loading data from {config.data.data_dir}")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=str(config.data.data_dir) if args.train_dir is None else None,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        batch_size=config.data.batch_size,
        image_size=config.data.image_size,
        num_workers=config.data.num_workers,
        subset_size=args.subset_size,
    )
    logger.info(
        f"Data: train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}, "
        f"test={len(test_loader.dataset)}"
    )
    
    # Model
    # model_name already defined above
    if model_name == "unet":
        model = UNet(
            in_channels=config.model.input_channels,
            out_channels=config.model.output_channels,
            encoder_channels=config.model.encoder_channels,
        )
    elif model_name == "deeplab":
        model = DeepLabV3Plus(num_classes=config.model.output_channels)
    elif model_name == "swin":
        model = SwinModel(
            channels=256,
            out_size=config.data.image_size,
            classes=config.model.output_channels,
            pretrained=False,
        )
    else:
        raise ValueError(f"Unknown model architecture: {args.model}")
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
        run_name=run_name,
        save_epoch=args.save_epoch,
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
