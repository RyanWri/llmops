from src.poc.atari_env import AtariEnv
import random


def test_environment():
    # Initialize Atari environment
    game_name = "ALE/Pong-v5"  # You can switch this to "ALE/Hero-v5"
    env = AtariEnv(game_name)

    obs, info = env.reset()
    done = False
    print("Initial Observation:", obs)

    while not done:
        action = random.randint(0, env.env.action_space.n - 1)  # Take random action
        obs, reward, done, truncated, info = env.step(action)

        print(f"Action: {action}, Reward: {reward}, Done: {done}")
        env.render()  # Display the environment

    env.close()


if __name__ == "__main__":
    test_environment()
