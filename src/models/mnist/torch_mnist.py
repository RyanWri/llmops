import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.profiler import (
    profile,
    ProfilerActivity,
    tensorboard_trace_handler,
)


def setup() -> dict:
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    root_dir = "/home/linuxu"
    dataset_dir = f"{root_dir}/datasets/data"
    logs_dir = os.path.join(root_dir, "models-logs", "mnist", "logs")
    return dict(
        device=device, root_dir=root_dir, dataset_dir=dataset_dir, logs_dir=logs_dir
    )


# Load MNIST dataset
def load_data(batch_size: int, dataset_dir: str):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_dataset = datasets.MNIST(
        root=dataset_dir, train=True, download=False, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# Train and profile the model
def train_and_profile_model():
    config = setup()
    train_loader = load_data(batch_size=64, dataset_dir=config["dataset_dir"])
    device = config["device"]
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Profiler log directory
    log_dir = config["logs_dir"]

    # Start profiling
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=tensorboard_trace_handler(log_dir),  # Save logs for TensorBoard
    ) as prof:
        model.train()
        for epoch in range(1):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                if batch_idx == 20:  # Limit profiling to 20 iterations for demo
                    break

    # Print profiling results
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
        )
    )
    print(f"Profiling logs saved to {log_dir}")


if __name__ == "__main__":
    train_and_profile_model()
