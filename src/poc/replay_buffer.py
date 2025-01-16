import random
from collections import deque
import numpy as np
import sys


class ReplayBuffer:
    def __init__(self, capacity):
        """
        Initialize the replay buffer.
        Args:
            capacity (int): Maximum number of experiences to store.
        """
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        """
        Store an experience in the buffer.
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state after the action.
            done: Whether the episode ended.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.
        Args:
            batch_size (int): Number of experiences to sample.
        Returns:
            Tuple of arrays: (states, actions, rewards, next_states, dones).
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)

    def print_experience_types(self):
        """
        Print the types of each item in a stored experience tuple.
        """
        if len(self.buffer) == 0:
            print("Replay Buffer is empty.")
            return
        experience = self.buffer[0]  # Get the first stored experience
        print("Types of items in the experience tuple:")
        for idx, item in enumerate(experience):
            print(f"Item {idx + 1}: Type - {type(item)}, Size - {sys.getsizeof(item)}")
