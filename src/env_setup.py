import gymnasium as gym


def create_env(env_name):
    return gym.make(env_name)
