import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tqdm import tqdm
from .tetris_client import TetrisClient
from .globel_constant import IMG_HEIGHT, IMG_WIDTH, MOVE_LEFT, MOVE_RIGHT, ROTATE_CW, ROTATE_CCW, DROP

class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, host="127.0.0.1", port=10612, reward_type="train"):
        super().__init__()

        self.client = TetrisClient(host, port)

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8
        )

        self.action_space = spaces.Discrete(5)
        self.steps = 0
        self.prev_cleared = 0
        self.prev_hole = 0
        self.prev_height = 0
        self.prev_action = None
        self.reward_type = reward_type

    def reset(self, seed=None, options=None):
        self.client.start()
        state = self.client.get_state()
        _, _, _, _, _, _, _, _, obs = state
        self._update_state(state)
        self.steps = 0

        return obs, {}

    def step(self, action):
        if action == MOVE_LEFT:
            self.client.move(-1)
        elif action == MOVE_RIGHT:
            self.client.move(1)
        elif action == ROTATE_CCW:
            self.client.rotate(0, 1)
        elif action == ROTATE_CW:
            self.client.rotate(1, 1)
        elif action == DROP:
            self.client.drop()
        else:
            self.client.move(0)
        
        state = self.client.get_state()
        is_over, cleared, holes, height, bumpiness, pillar, y_pos, contact, obs = state

        reward = self._calculate_reward(state, action, steps = self.steps)
        self._update_state(state)
        self.prev_action = action

        self.steps += 1
        terminated = is_over
        truncated = False

        info = {
            "lines_cleared": cleared,
            "reward": reward,
            "steps": self.steps,
            "holes": holes,
            "height": height,
            "bumpiness": bumpiness,
            "pillar": pillar,
            "y_pos": y_pos,
            "contact": contact,
        }

        if truncated:
            info["TimeLimit.truncated"] = True

        return obs, reward, terminated, truncated, info

    def render(self):
        _, _, _, _, _, _, _, _, obs = self.client.get_state()
        return obs

    def close(self):
        self.client.close()

    def _calculate_reward(self, state, action, steps):
        is_over, cleared, holes, height, bumpiness, pillar, y_pos, contact, _ = state

        cleared_delta = cleared - self.prev_cleared
        holes_delta = holes - self.prev_hole
        height_delta = height - self.prev_height
        bumpiness_delta = bumpiness - self.prev_bumpiness

        if self.reward_type == "train":
            reward = self._base_train_reward(cleared_delta, holes, holes_delta, height, bumpiness_delta, y_pos, contact, is_over, action, steps)
        elif self.reward_type == "eval":
            reward = self._base_eval_reward(cleared_delta)
        else:
            raise ValueError("Invalid reward type")

        return reward

    def _update_state(self, state):
        _, cleared, holes, height, bumpiness, pillar, y_pos, contact, _ = state
        self.prev_cleared = cleared
        self.prev_hole = holes
        self.prev_height = height
        self.prev_bumpiness = bumpiness
        self.prev_pillar = pillar
        self.prev_y_pos = y_pos
        self.prev_contact = contact

    def _base_train_reward(self, cleared_delta, holes, holes_delta, height, bumpiness_delta, y_pos, contact, is_over, action, steps):
        reward = 0
        reward += 0.1 * (1.2 ** ((height - holes) // 10))
        reward += 2 if action == DROP else 0
        reward -= y_pos * 0.5 if y_pos > (height / 10 + 1) else 0
        reward -= holes_delta * 0.6 if holes_delta > 0 else 0
        reward += cleared_delta * 100 * max(1, (steps // 200)) if cleared_delta > 0 else 0
        reward -= 50 if is_over else 0
        reward -= 200 if steps <= 50 and is_over else 0
        reward -= 50 if steps <= 100 and is_over else 0
        reward -= 2 if contact < 3 else 0
        reward -= 2 if contact < 2 else 0
        reward -= bumpiness_delta * 0.1 if bumpiness_delta > 3 else 0

        return reward
    
    def _base_eval_reward(self, cleared_delta):
        return cleared_delta + 1 / 1000000
