from src.env_setup import create_env
from src.policy_network import PolicyNetwork
from src.replay_buffer import ReplayBuffer
from src.train import train_agent
from src.gpu_logger import log_gpu_metrics
import torch
import threading


def main():
    env_name = "CartPole-v1"
    num_episodes = 500
    batch_size = 32
    gamma = 0.99
    buffer_capacity = 10000
    logging_interval = 5  # in seconds
    log_file = "/home/linuxu/rl-playground/logs/cartpole_gpu_metrics.csv"

    env = create_env(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = PolicyNetwork(state_dim, action_dim).to(device)
    replay_buffer = ReplayBuffer(buffer_capacity)

    # Start GPU logging in a separate thread
    log_metrics = log_gpu_metrics(interval=logging_interval, log_file=log_file)
    logging_thread = threading.Thread(
        target=lambda: [log_metrics() for _ in range(num_episodes)]
    )
    logging_thread.start()

    # Train the agent
    train_agent(env, policy_net, replay_buffer, device, num_episodes, batch_size, gamma)

    # Wait for the logging thread to finish
    logging_thread.join()


if __name__ == "__main__":
    main()
