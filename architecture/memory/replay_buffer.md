# Replay Buffer Memory Usage Forecast

## Overview
The replay buffer is a critical component in the Deep Q-Network (DQN) architecture, storing agent experiences for sampling during training. This document details how we analyze and forecast the memory usage of the replay buffer.

---

## Components of an Experience
Each experience tuple stored in the replay buffer consists of the following:

1. **State**: The observation from the environment.
2. **Action**: The action taken by the agent.
3. **New_State**: The subsequent observation after taking the action.
4. **Reward**: The immediate reward received.
5. **Is_Done**: A boolean indicating if the episode has ended.

---

## Memory Breakdown

The size of each tuple depends on the following components:

- **State**: Depends on the observation space of the environment.
  - **Classic Environments (e.g., CartPole)**: Low-dimensional vectors (e.g., positions, velocities).
  - **Atari Environments**: High-dimensional image data (e.g., \(84 \times 84\) grayscale frames).
- **Action**:
  - **Discrete Actions**: Stored as a single integer.
  - **Continuous Actions**: Stored as floating-point numbers or vectors.
- **New_State**: Same size as the `State`.
- **Reward**: A single floating-point value.
- **Is_Done**: A boolean stored as `uint8` or `bool`.

### Tuple Size Calculation
The total size of a tuple can be calculated as:

Size of tuple = sizeof(State) + sizeof(Action) + sizeof(New_State) + sizeof(Reward) + sizeof(Is_Done)


#### Example: Classic Environment (e.g., CartPole)
- **State**: 4 values (\(4 \times 4\) bytes).
- **Action**: 4 bytes.
- **New_State**: 4 values (\(4 \times 4\) bytes).
- **Reward**: 4 bytes.
- **Is_Done**: 1 byte.

**Total Size**: 
2 × (4 × 4) + 4 + 4 + 1 = 37 bytes


### Replay Buffer Capacity
For a buffer with \(10^6\) experiences:
Total memory = Size of tuple × Replay buffer capacity

**Example**: 
37 bytes × 10^6 = 37 MB


---

## Sampling Techniques

1. **Uniform Sampling**:
   - Each experience is equally likely to be sampled.
   - **Pros**: Simple to implement and computationally efficient.
   - **Cons**: Treats all experiences equally, potentially underutilizing important transitions.

2. **Prioritized Experience Replay (PER)**:
   - Experiences are sampled based on their temporal difference (TD) error.
   - **Pros**: Focuses training on transitions that significantly improve learning.
   - **Cons**: Adds computational overhead and increases memory usage for storing priorities.

---

## Assumptions
1. All environments use `float32` for numerical values unless otherwise specified.
2. Booleans (`is_done`) are stored as `uint8`.
3. Default replay buffer size is \(10^6\) experiences unless customized.

---
