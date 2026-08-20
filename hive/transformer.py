from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.models.feature_extraction import create_feature_extractor


class FPN(nn.Module):
    """
    Feature Pyramid Network.

    Input:
        C2: [B, 96,  H/4,  W/4]
        C3: [B, 192, H/8,  W/8]
        C4: [B, 384, H/16, W/16]
        C5: [B, 768, H/32, W/32]

    Output:
        P2-P5: [B, out_channels, corresponding spatial size]
    """

    def __init__(self, in_channels: List[int], out_channels: int = 256) -> None:
        super().__init__()

        # 1x1 lateral convolutions to reduce channel dimensions
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1)
            for c in in_channels
        ])

        # 3x3 output convolutions to smooth the pyramid levels
        self.output_convs = nn.ModuleList([
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            )
            for _ in in_channels
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass through FPN.
        
        Args:
            features: List of feature maps [C2, C3, C4, C5]
        
        Returns:
            List of pyramid feature maps [P2, P3, P4, P5]
        """

        # Apply lateral convolutions to all feature levels
        laterals = [
            conv(x)
            for conv, x in zip(self.lateral_convs, features)
        ]

        # Top-down pathway: upsample and add features from coarser levels
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode="nearest"
            )

        # Apply output convolutions to smooth the results
        outputs = [
            conv(x)
            for conv, x in zip(self.output_convs, laterals)
        ]

        return outputs


class SwinFPN(nn.Module):
    """
    Swin Transformer backbone + FPN.

    Produces:
        P2: 1/4 resolution
        P3: 1/8 resolution
        P4: 1/16 resolution
        P5: 1/32 resolution
    """

    def __init__(
        self,
        out_channels: int = 256,
        pretrained: bool = False,
    ) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Swin Transformer
        # ---------------------------------------------------------
        if pretrained:
            weights = Swin_T_Weights.DEFAULT
        else:
            weights = None

        swin = swin_t(weights=weights)

        # Swin-T feature stages:
        #
        # features.0 -> patch embedding
        # features.1 -> stage 1       -> 96 channels
        # features.2 -> patch merging
        # features.3 -> stage 2       -> 192 channels
        # features.4 -> patch merging
        # features.5 -> stage 3       -> 384 channels
        # features.6 -> patch merging
        # features.7 -> stage 4       -> 768 channels

        self.backbone = create_feature_extractor(
            swin,
            return_nodes={
                "features.1": "C2",
                "features.3": "C3",
                "features.5": "C4",
                "features.7": "C5",
            }
        )

        # ---------------------------------------------------------
        # FPN
        # ---------------------------------------------------------
        self.fpn = FPN(
            in_channels=[96, 192, 384, 768],
            out_channels=out_channels
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            dictionary containing P2-P5
        """

        features = self.backbone(x)

        # Swin outputs are [B, H, W, C]
        # Convert to [B, C, H, W] for convolution/FPN.
        features = [
            features["C2"].permute(0, 3, 1, 2),
            features["C3"].permute(0, 3, 1, 2),
            features["C4"].permute(0, 3, 1, 2),
            features["C5"].permute(0, 3, 1, 2),
        ]

        pyramid = self.fpn(features)

        return {
            "P2": pyramid[0],
            "P3": pyramid[1],
            "P4": pyramid[2],
            "P5": pyramid[3],
        }


class SwinModel(nn.Module):

    def __init__(
        self,
        channels: int = 256,
        classes: int = 1,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.fpn = SwinFPN(out_channels=channels,
                pretrained=pretrained)

        self.con2d = nn.Sequential(
            nn.Conv2d(
                channels * 4,
                channels,
                kernel_size=7,
                padding=3,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=5,
                padding=2,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv2d(
            channels,
            classes,
            kernel_size=3,
            padding=1
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass through Swin segmentation model."""

        input_size = X.shape[2:]

        pre = self.fpn(X)

        P2, P3, P4, P5 = pre["P2"], pre["P3"], pre["P4"], pre["P5"]

        P3 = F.interpolate(
            P3,
           size=P2.shape[-2:],
           mode="bilinear",
           align_corners=False
       )

        P4 = F.interpolate(
            P4,
            size=P2.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        P5 = F.interpolate(
            P5,
            size=P2.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # Multi-scale feature fusion
        x = torch.cat(
            [P2, P3, P4, P5],
            dim=1
        )
        
        out = self.classifier(self.con2d(x))

        return F.interpolate(
            out,
            size=input_size,
            mode="bilinear",
            align_corners=False
        )                              

