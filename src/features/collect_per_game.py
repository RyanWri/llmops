import time
import numpy as np
import pandas as pd
from src.replay_buffer import ReplayBuffer
from src.atari_env import AtariEnv
from src.features.collect import collect_dynamic_features, collect_static_features


def run_episode(env: AtariEnv, episode_number: int):
    done = False
    total_reward = 0
    steps = 0
    states_visited = []
    start_time = time.time()

    while not done:
        # maybe the step should be from the buffer or the agent should predict
        obs, reward, done, info = env.step()
        total_reward += reward
        steps += 1
        states_visited.append(obs.flatten())  # You might want to abstract these states

    episode_duration = time.time() - start_time
    # dynamic features collected at the end of each episode
    dynamic_features = collect_dynamic_features(
        total_reward, steps, np.array(states_visited), episode_duration, episode_number
    )
    return dynamic_features


if __name__ == "__main__":
    game_id = "ALE/Pong-v5"
    env = AtariEnv(game_id, render_mode="rgb_array")
    replay_buffer = ReplayBuffer(buffer_size=10000, sample_size=100)

    # need to add agent later
    dql_agent = None

    # collect static features
    # static features can be collected once outside of the episode because they are static
    static_features = collect_static_features(env, dql_agent, replay_buffer)

    # collect dynamic features
    total_features = []
    num_of_episodes = 2
    for episode_number in range(num_of_episodes):
        # make sure env is reseted before running an episode
        env.reset()
        episode_features = run_episode(env, episode_number)
        total_features.append(episode_features)

    df = pd.DataFrame(total_features)
    game_name = game_id.replace("/", "-")
    filename = f"{game_name}.csv"
    df.to_csv(filename)
