"""Paper tuner: runs hyperparameter grid search using the `hive` package.

Usage:
    python -m paper.src.tuner --data-dir project/data --output-dir outputs/paper --epochs 3

This script uses the HyperGrid dataclass in `paper.src.configs` by default but accepts
simple CLI overrides for `data_dir` and `output_dir`.
"""
from pathlib import Path
import argparse
from .configs import HyperGrid
import hive


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning using hive.tune_hyperparameters")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset root (images/ and labels/)")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output dir to override hive.Config().output_dir")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from grid")
    parser.add_argument("--seed", type=int, default=None, help="Override seed from grid")
    args = parser.parse_args(argv)

    grid = HyperGrid()
    # allow simple CLI overrides
    if args.epochs is not None:
        grid.epochs = args.epochs
    if args.seed is not None:
        grid.seed = args.seed

    # Build hive config
    config = hive.Config()
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    config.setup_directories()

    # Run tuning
    hive.tune_hyperparameters(
        config=config,
        data_dir=args.data_dir,
        train_dir=None,
        val_dir=None,
        test_dir=None,
        subset_size=grid.subset_size,
        epochs=grid.epochs,
        model_names=grid.models,
        learning_rates=grid.lrs,
        optimizers=grid.optimizers,
        batch_sizes=grid.batch_sizes,
        patience=grid.patience,
        seed=grid.seed,
    )


if __name__ == "__main__":
    main()
