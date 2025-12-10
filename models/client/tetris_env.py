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
        self.total_global_steps = 0
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
        is_over, cleared, holes, hight, bumpiness, pillar, y_pos, contact, obs = self.client.get_state()
        self.prev_cleared = cleared
        self.prev_hole = holes
        self.prev_hight = hight
        self.prev_bumpiness = bumpiness
        self.prev_pillar = pillar
        self.prev_y_pos = y_pos
        self.prev_contact = contact
        self.steps = 0

        processed_obs = self._process_obs(obs)
        return processed_obs, {}

    def step(self, action):
        if action == 0:
            self.client.move(-1)
        elif action == 1:
            self.client.move(1)
        elif action == 2:
            self.client.move(0)
        elif action == 3:
            self.client.rotate(0, 1)
        elif action == 4:
            self.client.rotate(1, 1)
        
        self.steps += 1
        is_over, cleared, holes, hight, bumpiness, pillar, y_pos, contact, obs = self.client.get_state()

        cleared_delta = cleared - self.prev_cleared
        holes_delta = holes - self.prev_hole
        bumpiness_delta = bumpiness - self.prev_bumpiness

        if self.reward_type == "train":
            reward = self._base_train_reward_v2(cleared_delta, holes_delta, hight, bumpiness_delta, pillar, y_pos, contact, is_over, action)
        elif self.reward_type == "eval":
            reward = self._base_eval_reward(cleared_delta)

        self.prev_cleared = cleared
        self.prev_hole = holes
        self.prev_hight = hight
        terminated = is_over
        truncated = False

        processed_obs = self._process_obs(obs)

        info = {
            "lines_cleared": cleared,
            "reward": reward,
            "steps": self.steps,
            "holes": holes,
            "height": hight,
            "bumpiness": bumpiness,
            "pillar": pillar,
            "y_pos": y_pos,
            "contact": contact,
        }

        if truncated:
            info["TimeLimit.truncated"] = True

        return processed_obs, reward, terminated, truncated, info

    def render(self):
        _, _, _, _, _, _, _, _, obs = self.client.get_state()
        return obs

    def close(self):
        self.client.close()

    def _base_train_reward(self, cleared_delta, holes_delta, height_delta, is_over):
        reward = (cleared_delta * 10) + 0.01
        reward -= (holes_delta * 0.1)
        reward -= 10 if is_over else 0
        return reward
    
    def _base_train_reward_v2(self, cleared_delta, holes_delta, height, bumpiness_delta, pillar, y_pos, contact, is_over, action):
        reward = 0
        reward -= 50 if is_over else 0
        reward += 0.01
        reward += (2 ** cleared_delta) * 8 if cleared_delta !=0 else 0
        reward += 100 if cleared_delta == 4 else 0
        reward += 0.002 if action == 2 else 0

        reward += 0.03 * (contact - 3) if contact > 3 else 0
        reward -= 0.05 * (3 - contact) if contact < 3 else 0
        reward -= 0.1 if bumpiness_delta > 4 else 0

        reward += -0.1 if y_pos > (height / 10 + 2) else 0.01

        return reward
    
    def _base_eval_reward(self, cleared_delta):
        return cleared_delta + 1 / 1000000
