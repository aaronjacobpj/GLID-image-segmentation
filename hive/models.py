"""
Model architectures for semantic segmentation.
KISS principle: Clean, simple implementations.
"""

import torch
import torch.nn as nn
from typing import List


class DoubleConv(nn.Module):
    """Double convolution block: Conv2d -> BN -> ReLU -> Conv2d -> BN -> ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Simple U-Net for binary semantic segmentation.
    Encoder-Decoder architecture with skip connections.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        encoder_channels: List[int] = None,
    ):
        """
        Initialize U-Net.
        
        Args:
            in_channels: Number of input channels (e.g., 3 for RGB)
            out_channels: Number of output channels (e.g., 1 for binary segmentation)
            encoder_channels: Channel sizes for encoder levels
        """
        super().__init__()
        
        if encoder_channels is None:
            encoder_channels = [64, 128, 256, 512]
        
        # Encoder (downsampling)
        self.down1 = DoubleConv(in_channels, encoder_channels[0])
        self.pool1 = nn.MaxPool2d(2)
        
        self.down2 = DoubleConv(encoder_channels[0], encoder_channels[1])
        self.pool2 = nn.MaxPool2d(2)
        
        self.down3 = DoubleConv(encoder_channels[1], encoder_channels[2])
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(encoder_channels[2], encoder_channels[3])
        
        # Decoder (upsampling)
        self.up1 = nn.ConvTranspose2d(encoder_channels[3], encoder_channels[2], kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(encoder_channels[2] * 2, encoder_channels[2])
        
        self.up2 = nn.ConvTranspose2d(encoder_channels[2], encoder_channels[1], kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(encoder_channels[1] * 2, encoder_channels[1])
        
        self.up3 = nn.ConvTranspose2d(encoder_channels[1], encoder_channels[0], kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(encoder_channels[0] * 2, encoder_channels[0])
        
        # Output layer
        self.final = nn.Conv2d(encoder_channels[0], out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connections."""
        # Encoder
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        
        # Bottleneck
        bn = self.bottleneck(p3)
        
        # Decoder with skip connections
        up1 = self.up1(bn)
        up1 = torch.cat([up1, d3], dim=1)
        up1 = self.conv_up1(up1)
        
        up2 = self.up2(up1)
        up2 = torch.cat([up2, d2], dim=1)
        up2 = self.conv_up2(up2)
        
        up3 = self.up3(up2)
        up3 = torch.cat([up3, d1], dim=1)
        up3 = self.conv_up3(up3)
        
        # Output
        out = self.final(up3)
        return out


class SimpleCNN(nn.Module):
    """Lightweight CNN baseline for segmentation."""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x
