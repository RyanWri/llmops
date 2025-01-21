from collections import Counter
import math
import time
import gymnasium as gym
from scipy.stats import entropy
from src.replay_buffer import ReplayBuffer
from src.features.game_complexity import calculate_complexity_scores


def get_network_architecture(dqn_agent):
    # need to use dnnmem to calculate layers and weights
    return 1


def collect_static_features(game_id, network_architecture, buffer_size):
    dimensions = get_input_dimensions(game_id)
    dqn_agent_mem = get_network_architecture(network_architecture)
    return {
        "game_id": game_id,
        "agent_memory": dqn_agent_mem,
        "replay_buffer_size": buffer_size,
        "state_dimension": dimensions["state_dimensions"],
        "action_dimension": dimensions["action_space"],
    }


def collect_dynamic_features(reward, steps, states, episode_duration, episode_number):
    num_steps = len(steps)
    states_entropy = get_state_entropy(states)
    epsilon_config = (0.1, 0.01, 0.0001)
    exploration_rate = get_exploration_rate(episode_number, epsilon_config)

    return {
        "episode_reward": reward,
        "episode_steps": num_steps,
        "episode_duration": episode_duration,
        "episode_exploration_rate": 0,
        "episode_states_entropy": states_entropy,
    }


def get_timestamp():
    return time.time()


def get_game_complexity(game_id):
    return calculate_complexity_scores(game_id)


def get_input_dimensions(game_id):
    env = gym.make(game_id)
    state_dimensions = env.observation_space.shape
    action_space = env.action_space.n
    env.close()
    return {"state_dimensions": state_dimensions, "action_space": action_space}


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
