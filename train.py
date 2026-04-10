# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import CIFAR10ResNet50

def get_dataloader(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return loader

def train(model, loader, device, lr=0.001, epochs=3, save_path="model.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = get_dataloader()

    # Model 1
    model1 = CIFAR10ResNet50().to(device)
    train(model1, loader, device, lr=0.001, save_path="model1.pth")

    # Model 2 (different lr)
    model2 = CIFAR10ResNet50().to(device)
    train(model2, loader, device, lr=0.0005, save_path="model2.pth")