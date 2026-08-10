"""
Data loading utilities for GLID dataset.
KISS principle: Keep it simple and focused.
"""

import os
from pathlib import Path
from typing import Tuple, List
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF


class GLIDDataset(Dataset):
    """Simple GLID dataset loader with augmentation."""
    
    VALID_EXTENSIONS = {".jpeg", ".jpg", ".tiff", ".png"}
    
    def __init__(
        self,
        folder: str,
        image_size: int = 224,
        normalize_mean: float = 0.5,
        normalize_std: float = 0.5,
        augment: bool = False,
        augment_prob: float = 0.5,
    ):
        """
        Initialize dataset.
        
        Args:
            folder: Path to data folder (must have 'images' and 'labels' subdirs)
            image_size: Target image size
            normalize_mean: Normalization mean
            normalize_std: Normalization std
            augment: Enable data augmentation
            augment_prob: Probability of horizontal flip
        """
        self.folder = Path(folder)
        self.image_size = image_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.augment = augment
        self.augment_prob = augment_prob
        
        # Load file paths
        self.image_paths, self.label_paths = self._load_files()
        
        if not self.image_paths:
            raise ValueError(f"No valid image files found in {self.folder / 'images'}")
    
    def _load_files(self) -> Tuple[List[str], List[str]]:
        """Load matching image and label file paths."""
        images_dir = self.folder / "images"
        labels_dir = self.folder / "labels"
        
        image_paths = []
        label_paths = []
        
        for filename in sorted(os.listdir(images_dir)):
            if Path(filename).suffix.lower() not in self.VALID_EXTENSIONS:
                continue
            
            image_path = images_dir / filename
            label_path = labels_dir / filename
            
            if label_path.exists():
                image_paths.append(str(image_path))
                label_paths.append(str(label_path))
        
        return image_paths, label_paths
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and return image and label pair."""
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = Image.open(self.label_paths[idx]).convert("L")
        
        # Apply augmentation
        if self.augment and torch.rand(1).item() > self.augment_prob:
            image = TF.hflip(image)
            label = TF.hflip(label)
        
        # Transform
        image = self._transform_image(image)
        label = self._transform_label(label)
        
        return image, label
    
    def _transform_image(self, image: Image.Image) -> torch.Tensor:
        """Transform image: resize, convert, normalize."""
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[self.normalize_mean],
                std=[self.normalize_std]
            ),
        ])
        return transform(image)
    
    def _transform_label(self, label: Image.Image) -> torch.Tensor:
        """Transform label: resize and convert."""
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        return transform(label)


def create_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    image_size: int = 224,
    train_split: float = 0.8,
    num_workers: int = 4,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size
        image_size: Image size
        train_split: Fraction for training
        num_workers: Number of data loading workers
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Create dataset
    dataset = GLIDDataset(
        folder=data_dir,
        image_size=image_size,
        augment=True,
        augment_prob=0.5,
    )
    
    # Split data
    train_size = int(len(dataset) * train_split)
    val_size = int(len(dataset) * (1 - train_split) / 2)
    test_size = len(dataset) - train_size - val_size
    
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
