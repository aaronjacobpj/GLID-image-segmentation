import math
import torch
import sklearn
import torchvision
import torch.nn as nn
from pathlib import Path
from sklearn import metrics
from torchvision import transforms, models
from torch.nn import functional as F
import torchvision.transforms.functional as TF


#  Double Convolution Block
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


#  U-Net Model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        #  Encoder
        self.down1 = DoubleConv(3, 64)  # RGB → 64
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)  # concat → 256+256

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)

        #  Final Output Layer
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        #  Encoder
        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        #  Bottleneck
        bn = self.bottleneck(p3)

        #  Decoder
        up1 = self.up1(bn)
        up1 = torch.cat([up1, d3], dim=1)
        up1 = self.conv1(up1)

        up2 = self.up2(up1)
        up2 = torch.cat([up2, d2], dim=1)
        up2 = self.conv2(up2)

        up3 = self.up3(up2)
        up3 = torch.cat([up3, d1], dim=1)
        up3 = self.conv3(up3)

        #  Output
        out = self.final(up3)

        return out
