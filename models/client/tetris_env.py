import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tqdm import tqdm
from .tetris_client import TetrisClient

class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, host="127.0.0.1", port=10612, reward_type="train"):
        super().__init__()

        self.client = TetrisClient(host, port)

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(1, 100, 50), dtype=np.uint8
        )

        self.action_space = spaces.Discrete(5)
        self.steps = 0
        self.prev_cleared = 0
        self.prev_hole = 0
        self.prev_hight = 0
        self.reward_type = reward_type
    
    def _process_obs(self, obs):
        gray = np.dot(obs[...,:3], [0.2989, 0.5870, 0.1140])
        gray = cv2.resize(gray, (50, 100), interpolation=cv2.INTER_AREA)
        gray = gray.astype(np.uint8)
        gray = np.expand_dims(gray, axis=0)
        return gray

    def reset(self, seed=None, options=None):
        self.client.start()
        is_over, cleared, holes, hight, obs = self.client.get_state()
        self.prev_cleared = cleared
        self.prev_hole = holes
        self.prev_hight = hight
        self.steps = 0

        processed_obs = self._process_obs(obs)
        return processed_obs, {}

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
            self.client.move(0)

        is_over, cleared, holes, hight, obs = self.client.get_state()

        self.steps += 1

        cleared_delta = cleared - self.prev_cleared
        holes_delta = holes - self.prev_hole
        height_delta = hight - self.prev_hight

        if self.reward_type == "train":
            reward = (cleared_delta * 10) + 0.01
            reward -= (holes_delta * 0.1)
            # reward -= (height_delta * 0.1)
            reward -= 10 if is_over else 0
        elif self.reward_type == "eval":
            reward = cleared_delta + self.steps / 1000000

        self.prev_cleared = cleared
        self.prev_hole = holes
        self.prev_hight = hight
        terminated = is_over
        truncated = False

        processed_obs = self._process_obs(obs)

        info = {
            "lines_cleared": cleared,
            "reward": reward,
        }

        return processed_obs, reward, terminated, truncated, info

    def render(self):
        _, _, _, _, obs = self.client.get_state()
        return obs

    def close(self):
        self.client.close()
