import gymnasium as gym
import ale_py


class AtariEnv:
    def __init__(self, game_name, render_mode="human"):
        """
        Initialize the Atari environment.
        Args:
            game_name (str): Name of the Atari game (e.g., "ALE/Pong-v5").
            render_mode (str): Mode to render the environment ("human", "rgb_array").
        """
        gym.register_envs(ale_py)
        self.env = gym.make(game_name, render_mode=render_mode)

    def reset(self):
        """Reset the environment."""
        return self.env.reset()

    def step(self, action):
        """
        Perform an action in the environment.
        Args:
            action (int): Action to take.
        Returns:
            obs: The next state.
            reward: Reward received.
            done: Whether the episode is finished.
            info: Additional info from the environment.
        """
        return self.env.step(action)

    def render(self):
        """Render the current state of the environment."""
        self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()
