import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlShutdown,
)
import json

# Initialize NVML
nvmlInit()
gpu_handle = nvmlDeviceGetHandleByIndex(0)  # Assuming single GPU

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load MNIST dataset
def load_data(batch_size=64):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_dataset = datasets.MNIST(
        root="/home/linuxu/datasets/data",
        train=True,
        download=False,
        transform=transform,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


# Simple Model: Single fully connected layer
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)  # Flatten 28x28 input to 10 output classes

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = self.fc(x)
        return x


# Collect GPU memory metrics
def collect_gpu_memory_metrics(epoch):
    mem_info = nvmlDeviceGetMemoryInfo(gpu_handle)
    metrics = {
        "epoch": epoch,
        "memory_total_MB": mem_info.total // 1024**2,  # Convert bytes to MB
        "memory_used_MB": mem_info.used // 1024**2,  # Convert bytes to MB
    }
    print(f"Epoch {epoch} GPU Memory Metrics: {metrics}")
    return metrics


# Train the model and log GPU memory usage
def train_and_log_memory():
    train_loader = load_data()
    model = SimpleModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    gpu_metrics = []
    for epoch in range(1, 4):  # Train for 3 epochs
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

        print(f"Epoch {epoch} Training Loss: {running_loss / len(train_loader)}")

        # Collect GPU memory metrics after each epoch
        gpu_metrics.append(collect_gpu_memory_metrics(epoch))

    # Save GPU memory metrics to a JSON file
    logs_filename = "/home/linuxu/models-logs/mnist/nvml-logs/batch64_epochs3.json"
    with open(logs_filename, "w") as f:
        json.dump(gpu_metrics, f, indent=4)

    print("Training complete. GPU memory metrics saved to gpu_memory_metrics.json")


if __name__ == "__main__":
    try:
        train_and_log_memory()
    finally:
        nvmlShutdown()
