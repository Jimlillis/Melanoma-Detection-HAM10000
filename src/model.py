"""
EfficientNet-B0 baseline for binary classification (melanoma vs non-melanoma).
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_efficientnet_b0(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Build EfficientNet-B0 with a linear classifier head for `num_classes` outputs.
    Uses ImageNet weights when pretrained=True.
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
