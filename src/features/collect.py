import psutil
import torch
from collections import Counter
import math
import time
from scipy.stats import entropy
from src.atari_env import AtariEnv
from src.replay_buffer import ReplayBuffer
from src.features.game_complexity import calculate_complexity_scores


def get_network_architecture(dqn_agent):
    # need to use dnnmem to calculate layers and weights
    return None


def collect_static_features(
    env: AtariEnv, network_architecture, replay_buffer: ReplayBuffer
):
    state_dim, action_dim = env.get_dimensions().values()
    dqn_agent_mem = get_network_architecture(network_architecture)
    return {
        "agent_memory": dqn_agent_mem,
        "replay_buffer_size": replay_buffer.get_buffer_size(),
        "state_dimension": state_dim,
        "action_dimension": action_dim,
    }


def collect_dynamic_features(
    reward, num_steps, states, episode_duration, episode_number
):
    # needs to add states_entropy = get_state_entropy(states)
    epsilon_config = (0.1, 0.01, 0.0001)
    exploration_rate = get_exploration_rate(episode_number, epsilon_config)
    cpu_memory = get_cpu_memory()
    gpu_memory = get_gpu_memory()

    return {
        "episode_reward": reward,
        "episode_steps": num_steps,
        "episode_duration": episode_duration,
        "episode_exploration_rate": exploration_rate,
        "episode_states_entropy": 0,
        "cpu_memory": cpu_memory,
        "gpu_memory": gpu_memory,
    }


def get_timestamp():
    return time.time()


def get_game_complexity(game_id):
    return calculate_complexity_scores(game_id)


def get_replay_buffer_size(replay_buffer: ReplayBuffer):
    return replay_buffer.usage


def get_state_entropy(states):
    state_freq = Counter(states)
    freq_list = list(state_freq.values())
    return entropy(freq_list, base=2)


def get_exploration_rate(episode_number, epsilon_config):
    epsilon_min, epsilon_max, decay_rate = epsilon_config
    return epsilon_min + (epsilon_max - epsilon_min) * math.exp(
        -decay_rate * episode_number
    )


def get_cpu_memory():
    return psutil.Process().memory_info().rss  # Memory usage in bytes


def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated()
    return 0
