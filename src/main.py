import gymnasium as gym
import torch
from agent import DQLAgent
from replay_buffer import ReplayBuffer
from metrics_exporter import MetricsExporter

# Initialize components
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQLAgent(state_dim, action_dim)
replay_buffer = ReplayBuffer(buffer_size=10000)
metrics_exporter = MetricsExporter()

# Training loop
episodes = 500
batch_size = 64
for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0

    for t in range(200):  # Max timesteps
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # Sample from buffer and train
        if len(replay_buffer.buffer) > batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(
                batch_size
            )
            states = torch.FloatTensor(states).to(agent.q_network.fc[0].weight.device)
            actions = torch.LongTensor(actions).to(agent.q_network.fc[0].weight.device)
            rewards = torch.FloatTensor(rewards).to(agent.q_network.fc[0].weight.device)
            next_states = torch.FloatTensor(next_states).to(
                agent.q_network.fc[0].weight.device
            )
            dones = torch.FloatTensor(dones).to(agent.q_network.fc[0].weight.device)

            q_values = (
                agent.q_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            )
            next_q_values = agent.q_network(next_states).max(1)[0]
            target = rewards + agent.gamma * next_q_values * (1 - dones)

            loss = torch.nn.functional.mse_loss(q_values, target)
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

            metrics_exporter.log_metric("loss", loss.item())

        if done:
            break

    # Log metrics
    metrics_exporter.log_metric("rewards", total_reward)
    metrics_exporter.log_gpu_stats()
    agent.update_epsilon()

print("Training completed!")
metrics = metrics_exporter.export()
print("Exported Metrics:", metrics)
