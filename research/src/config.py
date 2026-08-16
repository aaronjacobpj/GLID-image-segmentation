from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class HyperGrid:
    """Dataclass describing a hyperparameter grid for tuning.

    - `models`: list of model names supported by `hive` (unet, deeplab, swin)
    - `lrs`: learning rates to try
    - `optimizers`: optimizer names to try (adam, sgd)
    - `batch_sizes`: batch sizes to try
    - `epochs`: number of epochs per run
    - `subset_size`: optional dataset subset size
    - `patience`: early stopping patience
    - `seed`: random seed to use for reproducibility
    """

    models: List[str] = field(default_factory=lambda: ["unet", "deeplab", "swin"])
    lrs: List[float] = field(default_factory=lambda: [1e-3, 5e-4, 1e-4])
    optimizers: List[str] = field(default_factory=lambda: ["adam", "sgd"])
    batch_sizes: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    epochs: int = 100
    subset_size: Optional[int] = 2000
    patience: Optional[int] = 5
    seed: int = 42
    save_epoch: bool = False
