# Hive - GLID Training Package

A clean, simple, and optimized training pipeline for glacial lake segmentation using deep learning.

**KISS Principle**: Keep It Simple, Structured.

## Overview

The `hive` package provides production-ready code for training semantic segmentation models on the GLID (Glacial Lakes Inventory Dataset) for mapping glacial lakes from satellite imagery.

## Package Structure

```
hive/
├── __init__.py       # Package exports
├── config.py         # Configuration management (dataclass-based)
├── data.py           # Dataset loading and preprocessing
├── models.py         # Model architectures (UNet, SimpleCNN)
├── utils.py          # Training utilities (losses, metrics, logger)
├── train.py          # Main training pipeline
├── inference.py      # Model inference script
└── README.md         # Documentation
```

## Installation

### Option 1: Install from repository
```bash
cd GLID-image-segmentation
pip install -e .
```

### Option 2: Use directly from hive folder
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Data

Download GLID dataset from [Zenodo](https://zenodo.org/records/14838695) and organize:

```
project/data/
├── images/      # RGB satellite tiles (*.png, *.jpg, *.tiff)
└── labels/      # Binary masks (*.png) - same filenames as images
```

### 2. Train a Model

**Using command line:**
```bash
python -m hive.train --data-dir project/data --epochs 50 --batch-size 16
```

**Using Python API:**
```python
import hive
from pathlib import Path

# Configure
config = hive.Config()
config.data.data_dir = Path("project/data")
config.training.epochs = 50
config.setup_directories()

# Create components
device = hive.setup_device()
logger = hive.setup_logger(config.logs_dir)
train_loader, val_loader, _ = hive.create_dataloaders(
    data_dir=str(config.data.data_dir),
    batch_size=config.data.batch_size,
)

# Create model and trainer
model = hive.UNet()
loss_fn = hive.DiceBCELoss(device=device)
optimizer = hive.create_optimizer(model, config)
scheduler = hive.create_scheduler(optimizer, config)

trainer = hive.Trainer(
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
```

### 3. Run Inference

```bash
python -m hive.inference --model hive/checkpoints/best_model.pth --image test_image.png --output prediction.png
```

## Command Line Options

```bash
python -m hive.train [OPTIONS]

Options:
  --data-dir PATH      Path to data directory (default: project/data)
  --batch-size N       Batch size (default: 16)
  --epochs N           Number of epochs (default: 50)
  --lr FLOAT           Learning rate (default: 0.001)
  --model NAME         Model: unet or simplecnn (default: unet)
```

## Configuration

Modify settings via `hive.Config`:

```python
from hive import Config, DataConfig, TrainingConfig

config = Config(
    data=DataConfig(
        batch_size=32,
        image_size=224,
        num_workers=4,
    ),
    training=TrainingConfig(
        epochs=100,
        learning_rate=1e-4,
        optimizer="adam",
        scheduler="cosine",
        patience=10,
    ),
)
```

## Models

### UNet (Default)
- Best for production
- Proven architecture for segmentation
- Encoder-decoder with skip connections
- Parameters: ~7.8M

### SimpleCNN
- Lightweight baseline
- Faster training
- Lower memory requirements
- Parameters: ~0.4M

## Loss Function

**DiceBCELoss** (default):
- Combines Dice Loss (overlap metric) with Binary Cross Entropy
- Handles class imbalance via `pos_weight` parameter
- Better for imbalanced segmentation tasks

```python
loss = hive.DiceBCELoss(
    weight_bce=0.5,      # Weight for BCE component
    weight_dice=0.5,     # Weight for Dice component
    pos_weight=10.0,     # Positive class weight for imbalance
    device="cuda",
)
```

## Metrics

- **IoU (Intersection over Union)**: Primary metric for segmentation
- **Pixel Accuracy**: Percentage of correctly classified pixels

## Output Files

```
hive/
├── checkpoints/
│   └── best_model_epochN.pth    # Best model weights
├── logs/
│   ├── training.log              # Training logs
│   └── training_history.csv      # Metrics per epoch
└── outputs/
    └── ...                        # Additional outputs
```

## Tips for Best Results

### Data Preparation
- ✓ Ensure image-label pairs have matching filenames
- ✓ Use proper formats: PNG, JPG, or TIFF
- ✓ Labels should be binary: 0=background, 1=lake
- ✓ Minimum 100 image pairs for training

### Training
- ✓ Start with default learning rate (1e-3)
- ✓ Use batch_size 16 or 32 for most GPUs
- ✓ Monitor `training_history.csv` for convergence
- ✓ Early stopping prevents overfitting (default patience=10)

### Model Selection
- ✓ **UNet**: Best accuracy, moderate speed
- ✓ **SimpleCNN**: Faster experiments, lower memory

## Troubleshooting

### Out of Memory
```bash
python -m hive.train --batch-size 8
```

### Slow Training
```bash
python -m hive.train --model simplecnn --batch-size 32
```

### Not Converging
```bash
python -m hive.train --lr 5e-4 --epochs 100 --patience 15
```

### Data Not Found
- Check: `project/data/images/` and `project/data/labels/` exist
- Verify image-label filename matching
- Ensure correct file extensions

## Architecture Decisions

### Why KISS Principle?
1. **Maintainability**: Code is easy to understand and modify
2. **Reproducibility**: No hidden complexity
3. **Performance**: Direct PyTorch, minimal overhead
4. **Flexibility**: Easy to customize for custom datasets

### Why These Components?
- **Dataclass Config**: Type-safe, declarative configuration
- **DiceBCELoss**: Better for imbalanced segmentation data
- **Adam + Cosine**: Proven combination for convergence
- **Gradient Clipping**: Prevents exploding gradients
- **Early Stopping**: Prevents overfitting

## Citation

If you use this package or the GLID dataset, please cite:

```bibtex
@dataset{glid2024,
  title={GLID: Glacial Lakes Inventory Dataset},
  url={https://zenodo.org/records/14838695},
  year={2024}
}
```

## License

See main repository LICENSE.

## Contributing

Contributions welcome! Please follow the KISS principle.

## Support

For issues, questions, or suggestions, open an issue in the main repository.
