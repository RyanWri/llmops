import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.model import DQN
from src.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, state_dim, action_dim, config):
        """
        Initialize the DQN Agent with the given configuration.
        Args:
            config (dict): Contains hyperparameters and settings.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon", 1.0)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.epsilon_min = config.get("epsilon_min", 0.1)
        self.lr = config.get("lr", 1e-4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks
        self.q_network = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        """
        Epsilon-greedy action selection.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_tensor = (
            torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()

    def train(self, batch_size, replay_buffer: ReplayBuffer):
        """
        Train the Q-network using a batch from the replay buffer.
        """
        if replay_buffer.get_current_size() < batch_size:
            return

        # Sample batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = (
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        )
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Compute Q-values
        q_values = self.q_network(states).gather(1, actions)

        # Compute target Q-values
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Compute loss and optimize
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update_target_network(self):
        """
        Update the target network weights.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
