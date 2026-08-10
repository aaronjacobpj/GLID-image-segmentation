"""
Configuration settings for model training.
KISS principle: Keep It Simple, Structured.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    data_dir: Path = Path("project/data")
    image_size: int = 224
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
    encoder_channels: list = None
    
    def __post_init__(self):
        if self.encoder_channels is None:
            self.encoder_channels = [64, 128, 256, 512]


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    scheduler: Optional[str] = "cosine"
    warmup_epochs: int = 2
    patience: int = 10
    device: str = "auto"  # 'cuda', 'mps', 'cpu', or 'auto'


@dataclass
class LossConfig:
    """Loss function configuration."""
    loss_type: str = "dice_bce"  # 'dice_bce', 'dice', 'bce'
    weight_bce: float = 0.5
    weight_dice: float = 0.5
    smooth: float = 1e-5
    pos_weight: float = 10.0


@dataclass
class Config:
    """Complete configuration object."""
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    loss: LossConfig = None
    
    # Paths
    output_dir: Path = Path("hive/outputs")
    checkpoint_dir: Path = Path("hive/checkpoints")
    logs_dir: Path = Path("hive/logs")
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.loss is None:
            self.loss = LossConfig()
    
    def setup_directories(self):
        """Create necessary directories."""
        for path in [self.output_dir, self.checkpoint_dir, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)
