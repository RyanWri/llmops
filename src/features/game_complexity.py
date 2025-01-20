import gymnasium as gym
import ale_py
import numpy as np


def calculate_complexity_scores(game_id):
    gym.register_envs(ale_py)
    env = gym.make(game_id)
    env.reset()

    # State Space Complexity: can be approximated by the observation space shape (simplified for demonstration)
    atari_space_shape = [
        env.observation_space.shape[i] for i in range(len(env.observation_space.shape))
    ]
    state_space_size = np.prod(atari_space_shape)
    # Action Space Size: directly from the action space
    action_space_size = env.action_space.n

    # Reset environment to prepare for reward frequency analysis
    env.reset()
    terminated, truncated = False, False
    rewards = []
    while not terminated and not truncated:
        action = (
            env.action_space.sample()
        )  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

    # Reward Frequency: a simple measure could be the average reward per step
    reward_frequency = sum(rewards) / len(rewards) if rewards else 0

    env.close()

    # Normalize these values to a 1-5 scale for simplicity; you might want to adjust these based on empirical data
    state_space_score = min(
        5, max(1, int(state_space_size / 10000))
    )  # Example normalization
    action_space_score = min(
        5, max(1, action_space_size // 2)
    )  # Simplified normalization
    reward_frequency_score = min(
        5, max(1, int(reward_frequency * 10))
    )  # Example normalization

    return {
        "State Space Score": state_space_score,
        "Action Space Score": action_space_score,
        "Reward Frequency Score": reward_frequency_score,
    }


def get_rules_complexity(game_id):
    # This is a simplistic manual mapping, and you should adjust it based on your game analysis
    complexity_mapping = {
        "ALE/Pong-v5": 1,
        "ALE/MontezumasRevenge-v5": 5,
        "ALE/MsPacman-v5": 3,
        "ALE/Hero-v5": 4,
        # Add more games as needed
    }
    return complexity_mapping.get(game_id, 1)  # Default to 1 if game not listed


if __name__ == "__main__":
    # Example usage
    game_complexity = calculate_complexity_scores(
        "ALE/Hero-v5"
    )  # Use an actual game ID
    print(game_complexity)

    # Example usage
    rules_complexity = get_rules_complexity("ALE/Pong-v5")
    print("Rules Complexity Score:", rules_complexity)
