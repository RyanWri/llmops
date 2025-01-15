# Predicting Memory Usage of the DQN Agent

## Overview
The memory usage of the DQN agent can be predicted by analyzing two key components:
1. **Input size to the agent**: Derived from the replay buffer and affected by the sample size.
2. **Neural network operations**: Memory required for storing weights, activations, and gradients during forward and backward passes.

This section outlines the approach to forecasting memory usage using the methodologies from the DNNMem paper and our understanding of the DQN architecture.

---

## Steps to Forecast Memory Usage

### 1. Input Size Calculation
The input to the DQN agent comes from the replay buffer, which samples a batch of experiences. The size of the input can be estimated as follows:
- **Batch Size (\(B\))**: The number of experiences sampled from the replay buffer.
- **Tuple Size (\(T\))**: The size of a single experience tuple, calculated in the replay buffer analysis.
- **Input Size**:
Input Size = B × T


For example:
- If \(B = 64\) and \(T = 37\) bytes (from replay buffer analysis), the input size is:
64 × 37 = 2368 bytes (~2.3 KB).


---

### 2. Neural Network Memory Usage
The DQN agent uses a neural network to approximate the Q-function. The memory usage of the network includes:
1. **Weights**: Memory required to store the model's parameters.
2. **Activations**: Intermediate outputs of each layer during forward passes.
3. **Gradients**: Memory required to store gradients during backward passes.

#### Weight Memory
- Determined by the number of parameters in the network and their data type (typically `float32`).
- For example, a fully connected layer with N_input inputs and N_output outputs has:

Weight Memory = (N_input × N_output + N_output) × sizeof(float32).


#### Activation and Gradient Memory
- **Activations**: Sum of the outputs from each layer during the forward pass.
- **Gradients**: Same size as activations, stored during backward passes for gradient calculation.

Using the DNNMem methodologies, we forecast:
- Memory for each layer based on input size and layer dimensions.
- Peak memory usage during training, which includes weights, activations, and gradients.

---

### 3. Sample Size Impact on Memory
The sample size (\(B\)) directly affects the input size to the neural network:
- **Larger Batch Sizes**: Increase memory usage for activations and gradients proportionally.
- **Smaller Batch Sizes**: Reduce memory requirements but may impact training stability.

---

### Methodology for Forecasting
1. **Replay Buffer Input**:
 - Predict input size based on batch size and tuple size.
 - Simulate sampling operations to validate memory predictions.

2. **Neural Network**:
 - Analyze the architecture (e.g., layer types, dimensions).
 - Estimate memory for weights, activations, and gradients layer by layer.
 - Combine results to determine total network memory usage.

3. **Dynamic Profiling**:
 - Use tools like PyTorch’s `torch.cuda.memory_allocated()` to monitor real-time memory usage.
 - Validate predictions with actual training runs.

4. **DNNMem Methodologies**:
 - Apply analytical models to predict memory usage across forward and backward passes.
 - Incorporate GPU memory fragmentation overheads for accurate forecasting.

---

## Assumptions
1. **Data Type**: All model parameters and data are stored as `float32`.
2. **Static Network**: The architecture remains fixed during training.
3. **Batch Size**: Fixed across training iterations unless explicitly changed.


### Links
1. [torch DQN](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
---


