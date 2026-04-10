import os
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import resnet50


class CIFAR10ResNet50(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.model = resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def load_resnet50_checkpoint(checkpoint_path: Optional[str], device: torch.device) -> nn.Module:
    model = CIFAR10ResNet50()
    model.to(device)
    model.eval()

    if checkpoint_path is None:
        print("[WARN] No checkpoint provided. Using randomly initialized model.")
        return model

    if not os.path.exists(checkpoint_path):
        print(f"[WARN] Checkpoint not found: {checkpoint_path}")
        print("[WARN] Using randomly initialized model instead.")
        return model

    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v

    model.load_state_dict(cleaned, strict=False)
    print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
    return model


def get_cifar10_class_names():
    return [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]