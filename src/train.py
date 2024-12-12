import torch
import torch.optim as optim
import torch.nn.functional as F


def train_policy_network(
    policy_net, optimizer, replay_buffer, batch_size, gamma, device
):
    if len(replay_buffer) < batch_size:
        return

    batch = replay_buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, device=device, dtype=torch.float32)
    actions = torch.tensor(actions, device=device, dtype=torch.long)
    rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
    next_states = torch.tensor(next_states, device=device, dtype=torch.float32)
    dones = torch.tensor(dones, device=device, dtype=torch.float32)

    logits = policy_net(states)
    action_probs = logits.gather(1, actions.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        next_logits = policy_net(next_states)
        next_action_values = torch.max(next_logits, dim=1).values
        targets = rewards + gamma * next_action_values * (1 - dones)

    loss = F.mse_loss(action_probs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train_agent(
    env, policy_net, replay_buffer, device, num_episodes, batch_size, gamma
):
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            state_tensor = torch.tensor(state, device=device, dtype=torch.float32)
            action_probs = policy_net(state_tensor)
            action = torch.multinomial(action_probs, num_samples=1).item()

            next_state, reward, done, _, _ = env.step(action)
            replay_buffer.push((state, action, reward, next_state, done))

            train_policy_network(
                policy_net, optimizer, replay_buffer, batch_size, gamma, device
            )

            state = next_state
            episode_reward += reward

        print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

    env.close()
