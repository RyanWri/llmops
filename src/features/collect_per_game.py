import time
from src.features.collect import (
    collect_dynamic_features,
)


def run_pong_episode(env, agent, episode_number, buffer):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    states_visited = []
    start_time = time.time()

    while not done:
        action = agent.predict(state)  # Replace this with your actual prediction logic
        next_state, reward, done, _ = env.step(action)
        states_visited.append(next_state)  # You might want to abstract these states
        state = next_state
        total_reward += reward
        steps += 1

    episode_duration = time.time() - start_time
    dynamic_features = collect_dynamic_features(
        total_reward, steps, states_visited, episode_duration, episode_number
    )
    return dynamic_features
