import gymnasium as gym
import ale_py


class AtariEnv:
    def __init__(self, game_name, render_mode=None, action_selector=None):
        gym.register_envs(ale_py)
        self.env = gym.make(game_name, render_mode=render_mode)
        self.action_selector = action_selector or self._random_action

    def _random_action(self, obs):
        return self.env.action_space.sample()

    def set_action_selector(self, action_selector):
        self.action_selector = action_selector

    def reset(self):
        return self.env.reset()

    def step(self):
        action = self.action_selector(None)
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def get_dimensions(self):
        return {
            "state_dim": self.env.observation_space.shape,
            "action_dim": self.env.action_space.n,
        }

    def close(self):
        self.env.close()
