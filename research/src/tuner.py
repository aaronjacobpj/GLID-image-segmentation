"""Paper tuner: runs hyperparameter grid search using the `hive` package.

Usage:
    # Run directly:
    python research/src/tuner.py --data-dir project/data --epochs 3
    # Or as a module:
    python -m research.src.tuner --data-dir project/data --output-dir outputs/paper --epochs 3

This script uses the HyperGrid dataclass in `research/src/config.py` for default values
and accepts CLI arguments to override any of the parameters.
"""
import sys
from pathlib import Path
import argparse

# Ensure project root is in sys.path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .config import HyperGrid
except (ImportError, ValueError):
    from research.src.config import HyperGrid

import hive


def main(argv=None):
    grid = HyperGrid()

    parser = argparse.ArgumentParser(
        description="Run hyperparameter tuning using hive.tune_hyperparameters"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(hive.Config().data.data_dir),
        help="Path to dataset root (images/ and labels/)",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=None,
        help="Path to separate train dataset folder",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default=None,
        help="Path to separate validation dataset folder",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help="Path to separate test dataset folder",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output dir to override hive.Config().output_dir",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="*",
        default=grid.models,
        choices=["unet", "deeplab", "swin"],
        help="Model architectures to tune",
    )
    parser.add_argument(
        "--lr",
        type=float,
        nargs="*",
        default=grid.lrs,
        help="Learning rate values to try",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        nargs="*",
        default=grid.optimizers,
        choices=["adam", "sgd", "adamw"],
        help="Optimizers to try",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="*",
        default=grid.batch_sizes,
        help="Batch sizes to try",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=grid.epochs,
        help="Number of epochs per tuning run",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=grid.subset_size,
        help="Use only the first N samples from each dataset",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=grid.patience,
        help="Early stopping patience for each tuning run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=grid.seed,
        help="Random seed for tuning runs",
    )
    parser.add_argument(
        "--save-epoch",
        action="store_true",
        default=grid.save_epoch,
        help="Save checkpoint filenames with epoch number instead of overwriting",
    )
    args = parser.parse_args(argv)

    # Build hive config
    config = hive.Config()
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    config.setup_directories()

    # Run tuning
    hive.tune_hyperparameters(
        config=config,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        subset_size=args.subset_size,
        epochs=args.epochs,
        model_names=args.model,
        learning_rates=args.lr,
        optimizers=args.optimizer,
        batch_sizes=args.batch_size,
        patience=args.patience,
        seed=args.seed,
        save_epoch=args.save_epoch,
    )


if __name__ == "__main__":
    main()

