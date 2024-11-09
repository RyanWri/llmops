# GPU Mem Genius

**GPU Mem Genius** is an advanced reinforcement learning project aimed at optimizing memory allocation on NVIDIA GPUs, specifically designed to enhance performance and efficiency for large-scale machine learning models. This project leverages reinforcement learning to dynamically manage GPU resources, reducing memory usage, minimizing idle times, and improving overall resource utilization.

## Project Overview

The primary goal of GPU Mem Genius is to develop an intelligent agent capable of making real-time adjustments to memory allocation and batch configurations. By using a reinforcement learning (RL) approach, this agent learns to maximize GPU efficiency across various models and frameworks.

### Key Features

- **Dynamic Memory Management**: Real-time memory allocation adjustments to optimize GPU and CPU utilization.
- **Reinforcement Learning-based Optimization**: The agent learns optimal configurations autonomously to maximize resource efficiency.
- **Extensive Baseline and Benchmark Analysis**: Data-driven metrics collection and benchmarking across models.
- **Support for Various ML Frameworks**: Compatible with PyTorch, TensorFlow, and other popular deep learning frameworks.

## Setup and Installation

### Prerequisites

- **Python 3.8+**
- **Poetry** for dependency management
- **NVIDIA GPU with Tesla A100** or similar for testing
- **VS Code** with Remote - SSH extension for remote development

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YourUsername/gpu-mem-genius.git
   cd gpu-mem-genius
