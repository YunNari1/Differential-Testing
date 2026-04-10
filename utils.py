import os
import random
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import datasets, transforms


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_cifar10_testloader(
    data_root: str = "./data",
    batch_size: int = 16,
    num_workers: int = 2,
):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return dataset, loader


def tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]

    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_np)


def save_result_image(save_path: str, image_tensor: torch.Tensor, title_lines):
    img = tensor_to_pil(image_tensor).convert("RGB")
    width, height = img.size

    canvas = Image.new("RGB", (width, height + 70), color=(255, 255, 255))
    canvas.paste(img, (0, 70))

    draw = ImageDraw.Draw(canvas)
    y = 10
    for line in title_lines:
        draw.text((10, y), line, fill=(0, 0, 0))
        y += 18

    canvas.save(save_path)