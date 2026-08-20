import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
from typing import Tuple


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256, dropout: float = 0.5):
        super().__init__()

        def block(in_c, out_c, k, p, d):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, k, padding=p, dilation=d, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.conv1 = block(in_channels, out_channels, 1, 0, 1)

        # Output stride = 8 -> atrous rates doubled per the DeepLabV3 paper
        self.conv12 = block(in_channels, out_channels, 3, 12, 12)
        self.conv24 = block(in_channels, out_channels, 3, 24, 24)
        self.conv36 = block(in_channels, out_channels, 3, 36, 36)

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.out = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        size = x.shape[2:]

        x1 = self.conv1(x)
        x2 = self.conv12(x)
        x3 = self.conv24(x)
        x4 = self.conv36(x)

        x5 = self.pool(x)
        x5 = F.interpolate(x5, size=size, mode="bilinear", align_corners=False)

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)

        return self.out(x)


class Backbone(nn.Module):

    def __init__(self):
        super().__init__()

        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT,
            replace_stride_with_dilation=[False, True, True],
        )

        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):

        x = self.layer0(x)

        low = self.layer1(x)

        x = self.layer2(low)
        x = self.layer3(x)
        high = self.layer4(x)

        return low, high


class Decoder(nn.Module):
    def __init__(self, num_classes: int, low_level_channels: int = 256, dropout: float = 0.1):
        super().__init__()

        self.low_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        low = self.low_conv(low)

        high = F.interpolate(
            high, size=low.shape[2:], mode="bilinear", align_corners=False
        )

        x = torch.cat([low, high], dim=1)
        x = self.conv(x)

        return self.classifier(x)


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.backbone = Backbone()
        self.aspp = ASPP(2048, 256)
        self.decoder = Decoder(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]

        low, high = self.backbone(x)

        high = self.aspp(high)

        x = self.decoder(low, high)

        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

        return x


