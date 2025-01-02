import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, state):
        return self.fc(state)

class DQLAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.q_network = QNetwork(state_dim, action_dim).to("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
    
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)  # Explore
        state = torch.FloatTensor(state).unsqueeze(0).to(next(self.q_network.parameters()).device)
        with torch.no_grad():
            return torch.argmax(self.q_network(state)).item()  # Exploit
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
