import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tetris_client import TetrisClient

class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, host="127.0.0.1", port=8000):
        super().__init__()

        self.client = TetrisClient(host, port)

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(200, 100, 3), dtype=np.uint8
        )

        # Actions:
        # 0: left
        # 1: right
        # 2: rotate CCW
        # 3: rotate CW
        # 4: drop
        self.action_space = spaces.Discrete(5)

        self.prev_cleared = 0

    def reset(self, seed=None, options=None):
        self.client.start()
        is_over, cleared, obs = self.client.get_state()
        self.prev_cleared = cleared
        return obs, {}

    def step(self, action):
        if action == 0:
            self.client.move(-1)
        elif action == 1:
            self.client.move(1)
        elif action == 2:
            self.client.rotate(0, 1)
        elif action == 3:
            self.client.rotate(1, 1)
        elif action == 4:
            self.client.drop()

        is_over, cleared, obs = self.client.get_state()

        reward = cleared - self.prev_cleared
        self.prev_cleared = cleared

        terminated = is_over
        truncated = False

        info = {"lines_cleared": cleared}

        return obs, reward, terminated, truncated, info

    def render(self):
        _, _, obs = self.client.get_state()
        return obs

    def close(self):
        self.client.close()
