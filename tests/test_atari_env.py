from src.atari_env import AtariEnv
import random


def test_environment():
    # Initialize Atari environment
    game_name = "ALE/Pong-v5"  # You can switch this to "ALE/Hero-v5"
    env = AtariEnv(game_name)

    obs, info = env.reset()
    print("Initial Observation:", obs)

    ale_pong_action_space = 6
    assert env.env.action_space.n == ale_pong_action_space, "Action space mismatch"

    action = random.randint(0, env.env.action_space.n - 1)  # Take random action
    obs, reward, done, truncated, info = env.step(action)

    assert obs.shape == (210, 160, 3), "Observation shape mismatch"

    env.close()
