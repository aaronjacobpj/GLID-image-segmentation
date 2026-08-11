import argparse
from pathlib import Path

from .config import Config
from . import train as train_module
from . import inference as inference_module


def run_train(args: argparse.Namespace) -> None:
    """Convert CLI args and execute the training module."""
    argv = [
        "--data-dir",
        args.data_dir,
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--optimizer",
        args.optimizer,
        "--model",
        args.model,
    ]

    if args.train_dir:
        argv.extend(["--train-dir", args.train_dir])
    if args.val_dir:
        argv.extend(["--val-dir", args.val_dir])
    if args.test_dir:
        argv.extend(["--test-dir", args.test_dir])
    if args.subset_size is not None:
        argv.extend(["--subset-size", str(args.subset_size)])

    train_module.main(argv)


def run_inference(args: argparse.Namespace) -> None:
    """Convert CLI args and execute the inference module."""
    argv = [
        "--model",
        args.model_path,
        "--image",
        args.image,
        "--output",
        args.output,
        "--threshold",
        str(args.threshold),
        "--model-type",
        args.model_type,
    ]
    inference_module.main(argv)


def main() -> None:
    """Entry point for the hive CLI."""
    config = Config()
    parser = argparse.ArgumentParser(
        description="hive CLI to run training and inference modules"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run training only")
    train_parser.add_argument(
        "--data-dir",
        type=str,
        default=str(config.data.data_dir),
        help="Path to dataset root",
    )
    train_parser.add_argument("--train-dir", type=str, default=None, help="Path to separate train dataset folder")
    train_parser.add_argument("--val-dir", type=str, default=None, help="Path to separate validation dataset folder")
    train_parser.add_argument("--test-dir", type=str, default=None, help="Path to separate test dataset folder")
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=config.data.batch_size, 
        help="Batch size",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=config.training.epochs,
        help="Number of epochs",
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=config.training.learning_rate,
        help="Learning rate",
    )
    train_parser.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=["unet", "deeplab", "swin"],
        help="Model architecture",
    )
    train_parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Use only the first N samples from each dataset",
    )
    train_parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Optimizer",
    )

    tune_parser = subparsers.add_parser("tune", help="Run hyperparameter tuning")
    tune_parser.add_argument(
        "--data-dir",
        type=str,
        default=str(config.data.data_dir),
        help="Path to dataset root",
    )
    tune_parser.add_argument("--train-dir", type=str, default=None, help="Path to separate train dataset folder")
    tune_parser.add_argument("--val-dir", type=str, default=None, help="Path to separate validation dataset folder")
    tune_parser.add_argument("--test-dir", type=str, default=None, help="Path to separate test dataset folder")
    tune_parser.add_argument(
        "--epochs",
        type=int,
        default=config.training.epochs,
        help="Number of epochs per tuning run",
    )
    tune_parser.add_argument(
        "--model",
        type=str,
        nargs="*",
        default=["unet", "deeplab", "swin"],
        choices=["unet", "deeplab", "swin"],
        help="Model architectures to tune",
    )
    tune_parser.add_argument(
        "--lr",
        type=float,
        nargs="*",
        default=[0.01, 0.001, 0.0005, 0.0001],
        help="Learning rate values to try",
    )
    tune_parser.add_argument(
        "--optimizer",
        type=str,
        nargs="*",
        default=["adam", "sgd"],
        choices=["adam", "sgd"],
        help="Optimizers to try",
    )
    tune_parser.add_argument(
        "--batch-size",
        type=int,
        nargs="*",
        default=[config.data.batch_size],
        help="Batch sizes to try",
    )
    tune_parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Use only the first N samples from each dataset",
    )
    tune_parser.add_argument(
        "--patience",
        type=int,
        default=config.training.patience,
        help="Early stopping patience for each tuning run",
    )

    infer_parser = subparsers.add_parser("infer", help="Run inference only")
    infer_parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    infer_parser.add_argument("--image", type=str, required=True, help="Path to input image")
    infer_parser.add_argument("--output", type=str, default="prediction.png", help="Output path")
    infer_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold",
    )
    infer_parser.add_argument(
        "--model-type",
        type=str,
        default="unet",
        choices=["unet", "deeplab", "swin"],
        help="Model type for inference",
    )

    all_parser = subparsers.add_parser("all", help="Run training then inference")
    all_parser.add_argument(
        "--data-dir",
        type=str,
        default=str(config.data.data_dir),
        help="Path to dataset root",
    )
    all_parser.add_argument("--train-dir", type=str, default=None, help="Path to separate train dataset folder")
    all_parser.add_argument("--val-dir", type=str, default=None, help="Path to separate validation dataset folder")
    all_parser.add_argument("--test-dir", type=str, default=None, help="Path to separate test dataset folder")
    all_parser.add_argument(
        "--batch-size",
        type=int,
        default=config.data.batch_size,
        help="Batch size",
    )
    all_parser.add_argument(
        "--epochs",
        type=int,
        default=config.training.epochs,
        help="Number of epochs",
    )
    all_parser.add_argument(
        "--lr",
        type=float,
        default=config.training.learning_rate,
        help="Learning rate",
    )
    all_parser.add_argument(
        "--optimizer",
        type=str,
        default=config.training.optimizer,
        choices=["adam", "sgd"],
        help="Optimizer",
    )
    all_parser.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=["unet", "deeplab", "swin"],
        help="Model architecture",
    )
    all_parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Use only the first N samples from each dataset",
    )
    all_parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint for inference")
    all_parser.add_argument("--image", type=str, required=True, help="Path to input image for inference")
    all_parser.add_argument("--output", type=str, default="prediction.png", help="Output path for inference")
    all_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for inference",
    )
    all_parser.add_argument(
        "--model-type",
        type=str,
        default="unet",
        choices=["unet", "deeplab", "swin"],
        help="Model type for inference",
    )

    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "tune":
        from .train import tune_hyperparameters

        tune_hyperparameters(
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
        )
    elif args.command == "infer":
        run_inference(args)
    elif args.command == "all":
        run_train(args)
        run_inference(args)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
