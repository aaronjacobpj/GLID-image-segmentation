"""
Inference script for using trained models.
KISS principle: Simple inference pipeline.
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
from typing import Tuple, Optional, List

from .deeplab import DeepLabV3Plus
from .transformer import SwinModel
from .unet import UNet
from .utils import setup_device


class Inference:
    """Simple inference wrapper."""
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "unet",
        device: str = "auto",
        image_size: int = 224,
    ) -> None:
        """
        Initialize inference.
        
        Args:
            model_path: Path to trained model checkpoint
            model_type: Model architecture type
            device: Device to use
            image_size: Input image size
        """
        self.device = setup_device(device)
        self.image_size = image_size
        self.model: torch.nn.Module
        
        # Load model
        model_type = model_type.lower()
        if model_type == "unet":
            self.model = UNet()
        elif model_type == "deeplab":
            self.model = DeepLabV3Plus(num_classes=1)
        elif model_type == "swin":
            self.model = SwinModel(classes=1, out_size=self.image_size, pretrained=False)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(
        self,
        image_path: str,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict segmentation mask.
        
        Args:
            image_path: Path to input image
            threshold: Confidence threshold
        
        Returns:
            Tuple of (probabilities, binary_mask)
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        
        # Convert to tensor
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(image_tensor)
        
        # Predictions
        probs = torch.sigmoid(logits)
        binary_mask = (probs > threshold).float()
        
        return probs.squeeze().cpu(), binary_mask.squeeze().cpu()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run inference on images")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="prediction.png", help="Output path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument(
        "--model-type",
        type=str,
        default="unet",
        choices=["unet", "deeplab", "swin"],
        help="Model type",
    )
    args = parser.parse_args(argv)
    
    # Inference
    inference = Inference(args.model, args.model_type)
    probs, mask = inference.predict(args.image, args.threshold)
    
    # Save prediction
    pred_image = Image.fromarray((mask.numpy() * 255).astype("uint8"))
    pred_image.save(args.output)
    
    print(f"Prediction saved to {args.output}")
    print(f"Mean confidence: {probs.mean():.4f}")


if __name__ == "__main__":
    main()
